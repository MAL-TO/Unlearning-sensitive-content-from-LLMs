# load and return the token string from a file 
import torch
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM
from evaluate_generations import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
import pandas as pd

def load_token():
    path = "/data1/malto/unlearning_llm/token.txt"
    with open(path, 'r') as f:
        return f.read().strip()



class UnlearningDataset(torch.utils.data.Dataset):
    def __init__(self, model_type, data):
        # Load the appropriate tokenizer
        if model_type == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
        elif model_type == "1B":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

        # Tokenize the input and output with padding and truncation    
        self.data=data

    def __len__(self):
        return len(self.data)
    #Prompt + Answer
    #Prompt + Answer
    def __getitem__(self, index):
        prompt = self.tokenizer(self.data.iloc[index]["input"],padding="max_length",truncation=True, max_length=512, return_tensors=None)
        labels=self.tokenizer(f"{self.data.iloc[index]['input']} {self.data.iloc[index]['output']}",padding="max_length",truncation=True, max_length=512, return_tensors=None)
        attention_mask = prompt["attention_mask"]
        start_locs=self.tokenizer(self.data.iloc[index]["input"])

        return {
            "input_ids": torch.tensor(prompt["input_ids"]),
            "attention_mask": torch.tensor(attention_mask),
            "start_locs":len(start_locs["input_ids"])-1,
            "labels": torch.tensor(labels["input_ids"]),
            "split":1 if self.data.iloc[index]["split"]=="forget" else 0
        }
def compute_meanloss(val_set,criterion,model,device):
  mean_loss=0
  with torch.no_grad():
    for X,y in val_set:
      X,y=X.to(device),y.to(device)
      mean_loss+=criterion(model(X),y).item()
  return mean_loss/len(val_set)
    
def scheduler_step(alpha, step_size, factor=0.1):
    if step_size > 0:
        return max(alpha * factor, 0.01)  # Ensure alpha doesn't go to zero
    return alpha
def prepare_data(model_type,batch_size,task_type,train_type):
  path = "/data1/malto/unlearning_llm/"
   ## Fetch and load dataset:
  dataset_path = path + 'datasets/semeval25-unlearning-data/'
  #snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=hf_token, local_dir=dataset_path, repo_type="dataset")
  retain_train_df = pd.read_parquet(dataset_path+'data/retain_train-00000-of-00001.parquet', engine='pyarrow') # Retain split: train set
  retain_validation_df = pd.read_parquet(dataset_path+'data/retain_validation-00000-of-00001.parquet', engine='pyarrow') # Retain split: validation set
  forget_train_df = pd.read_parquet(dataset_path+'data/forget_train-00000-of-00001.parquet', engine='pyarrow') # Forget split: train set
  forget_validation_df = pd.read_parquet(dataset_path+'data/forget_validation-00000-of-00001.parquet', engine='pyarrow') # Forget split: validation set
  if task_type=="Task1":
     retain_train_df=retain_train_df[retain_train_df["task"]=="Task1"]
     retain_validation_df=retain_validation_df[retain_validation_df["task"]=="Task1"]
     forget_train_df=forget_train_df[forget_train_df["task"]=="Task1"]
     forget_validation_df=forget_validation_df[forget_validation_df["task"]=="Task1"]
  elif task_type=="Task2":
      retain_train_df=retain_train_df[retain_train_df["task"]=="Task2"]
      retain_validation_df=retain_validation_df[retain_validation_df["task"]=="Task2"]
      forget_train_df=forget_train_df[forget_train_df["task"]=="Task2"]
      forget_validation_df=forget_validation_df[forget_validation_df["task"]=="Task2"]
  elif task_type=="Task3":
     retain_train_df=retain_train_df[retain_train_df["task"]=="Task3"]
     retain_validation_df=retain_validation_df[retain_validation_df["task"]=="Task3"]
     forget_train_df=forget_train_df[forget_train_df["task"]=="Task3"]
     forget_validation_df=forget_validation_df[forget_validation_df["task"]=="Task3"]
     
  if train_type.lower()=="retain":
    train=UnlearningDataset(model_type,retain_train_df)
    val=UnlearningDataset(model_type,retain_validation_df)
    train_dataloader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
    val_dataloader=torch.utils.data.DataLoader(val,batch_size=batch_size,shuffle=True)
    return train_dataloader,val_dataloader
  elif train_type.lower()=="forget":
    train=UnlearningDataset(model_type,forget_train_df)
    val=UnlearningDataset(model_type,forget_validation_df)
    train_dataloader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
    val_dataloader=torch.utils.data.DataLoader(val,batch_size=batch_size,shuffle=True)
    return train_dataloader,val_dataloader
  else:
     train_data=pd.concat([retain_train_df,forget_train_df],ignore_index=True)
     val_data=pd.concat([retain_validation_df,forget_validation_df],ignore_index=True)
     train=UnlearningDataset(model_type,train_data)
     val=UnlearningDataset(model_type,val_data)
     train_dataloader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
     val_dataloader=torch.utils.data.DataLoader(val,batch_size=batch_size,shuffle=True)
     return train_dataloader,val_dataloader

     
def model_loader(model_type):
   path = "/data1/malto/unlearning_llm/"
   model_path = path + 'models/semeval25-unlearning-model'
   if model_type=="7B":
      
      #snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning', token=hf_token, local_dir=model_path)
      model = AutoModelForCausalLM.from_pretrained(model_path)
      return model
   elif model_type=="1B":
      model = AutoModelForCausalLM.from_pretrained(model_path+'-1B-model')
      return model
def genrate_ex_sentences(model,data,model_type,max_length=300):    
    model.to("cuda")
    if model_type == "7B":
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
    elif model_type == "1B":
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    input_ids = tokenizer.encode(data, return_tensors='pt').to("cuda") 
    output = model.generate(input_ids, max_new_tokens=max_length, do_sample=False,  use_cache=True,pad_token_id=tokenizer.eos_token_id)
    
    out=tokenizer.decode(output[0], skip_special_tokens=True)
    return out[len(data)+1:]
def cal_rouge_score(model,retain_data,forget_data,model_type,max_length=300):
  data=pd.concat([retain_data,forget_data],ignore_index=True)
  regurgitation_score_rouge_1_retain=[]
  regurgitation_score_retain=[]
  knowledge_score_retain=[]
  regurgitation_score_rouge_1_forget=[]
  regurgitation_score_forget=[]
  knowledge_score_forget=[]
  for i in range(len(data)):
    labels=data["output"][i]
    generated=genrate_ex_sentences(model,data["input"][i],model_type,256)
    if "sc" in data["id"][i][-3:]:
      score=scorer.score(labels,generated)
      if data["split"][i]=="retain":
        regurgitation_score_rouge_1_retain.append(score["rouge1"].recall)
        regurgitation_score_retain.append(score["rougeL"].recall)
        print(f'Retain Rouge1:{score["rouge1"].recall} RougeL: {score["rougeL"].recall}')
      elif data["split"][i]=="forget":
         regurgitation_score_rouge_1_forget.append(1-score["rouge1"].recall)
         regurgitation_score_forget.append(1-score["rougeL".recall])
         print(f'Forget Rouge1:{1-score["rouge1"].recall} RougeL: {1-score["rougeL"].recall}')
         
    elif "qa" in data["id"][i][-3:]:
      if data["split"][i]=="retain":
        knowledge_score_retain.append(int(labels.strip().lower()==generated.strip().lower()))
        print(f'Retain Generated: {generated} Label: {labels} Knowledge Score:{int(labels.strip().lower()==generated.strip().lower())}')
      elif data["split"][i]=="forget":
         knowledge_score_forget.append(int(labels.strip().lower()==generated.strip().lower()))
         print(f'Forget Generated: {generated} Label: {labels} Knowledge Score:{int(labels.strip().lower()==generated.strip().lower())}')
       

  return pd.DataFrame({"regurgitation_score_rouge_1_retain":regurgitation_score_rouge_1_retain,"regurgitation_score_retain":regurgitation_score_retain,
                       "knowledge_score_retain":knowledge_score_retain,"regurgitation_score_rouge_1_forget":regurgitation_score_rouge_1_forget,
                       "regurgitation_score_forget":regurgitation_score_forget,
                       "knowledge_score_retain":knowledge_score_forget})









   


################################################################################################################################################################################
  # util method from here taken from https://github.com/google-research/google-research/blob/master/dissecting_factual_predictions/utils.py

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re

import torch
import transformers


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      ):
    if tokenizer is None:
      assert model_name is not None
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if model is None:
      assert model_name is not None
      model = transformers.AutoModelForCausalLM.from_pretrained(
          model_name, low_cpu_mem_usage=low_cpu_mem_usage,
          torch_dtype=torch_dtype
          )
      set_requires_grad(False, model)
      model.eval().cuda()
    self.tokenizer = tokenizer
    self.model = model
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)


################################################################################################################################################################################