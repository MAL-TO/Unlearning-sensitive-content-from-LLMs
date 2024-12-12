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