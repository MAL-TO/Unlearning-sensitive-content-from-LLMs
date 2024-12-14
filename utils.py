# load and return the token string from a file 
import torch
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
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
        self.data = data
        self.X = self.data["input"].apply(
            lambda x: self.tokenizer(x, padding="max_length", truncation=True, max_length=128, return_tensors=None)
        )
        self.y = self.data["output"].apply(
            lambda x: self.tokenizer(x, padding="max_length", truncation=True, max_length=128, return_tensors=None)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.X.iloc[index]["input_ids"])
        attention_mask = torch.tensor(self.X.iloc[index]["attention_mask"])
        labels = torch.tensor(self.y.iloc[index]["input_ids"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
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

def get_answer_loss(operation, batch, model, device="cuda:0"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask,labels  = (
        batch["input_ids"].unsqueeze(0).to(device),
        batch["attention_mask"].unsqueeze(0).to(device),
        batch["labels"].to(device)
    )
    outputs = model(input_ids,attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")


    # GA or GD.
    position_loss = loss_fct(outputs, labels)
    if operation == "ga":  # Negative the direction for GA.
        position_loss = -position_loss


    return position_loss


    
def GradientAscentTrainLoop(model,forget_set,retain_set,val_forget_set,val_retain_set,epoch,device,optimizer,alpha,gamma,train_type):
  """
  Training Loop that uses gradient ascent algorithm

  :param model: model used for training
  :param forget_set: forget set part of data set
  :param retain_Set retain set part of data set
  :param val_forget_set: forget set part of validation data set
  :param val_retain_Set retain set part of validation data set
  :param epoch: number of epochs 
  :param device: device for the training
  :param optimizer : optimizer used for training
  :param alpha (int): coefficent for forget loss
  :param beta gamma (int): coefficent for retain loss
  :param traintype: defining the train type 1 for gradient_difference ,2 for gradient_ascent

  :returns: trained model
  """
  model.to(device)
  model.train()
  if train_type==1:
    for forget_epoch in range(epoch):
      epoch_loss=0
      for forget,retain in zip(forget_set,retain_set):
        model.zero_grad()
        L_f=get_answer_loss("ga",forget,model,device)
        L_r=get_answer_loss("gd",retain,model,device)
        loss=alpha*L_f+gamma*L_r
        
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
      total_val_loss=0
      with torch.no_grad():
        for val_f,val_r in zip(val_forget_set,val_retain_set):
            val_L_f=get_answer_loss("ga",val_f,model,device)
            val_L_r=get_answer_loss("gd",val_r,model,device)
            val_loss=alpha*val_L_f+gamma*val_L_r
            total_val_loss+=val_loss.item()
      print(f"Epoch {forget_epoch}, Train Loss: {epoch_loss} Validation Loss: {total_val_loss:.4f}")
  elif train_type==2:
    for forget_epoch in range(epoch):
      epoch_loss=0
      for forget in forget_set:
        model.zero_grad()
        loss=get_answer_loss("ga",forget,model,device)
        
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
      total_val_loss=0
      with torch.no_grad():
        for val_f in val_forget_set:
              val_loss=get_answer_loss("ga",val_f,model,device)
              total_val_loss+=val_loss.item()
      print(f"Epoch {forget_epoch}, Train Loss: {epoch_loss} Validation Loss: {total_val_loss:.4f}")
  

  return model

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