# load and return the token string from a file 
import torch
from datasets import load_dataset
import pandas as pd
def load_token():
    path = "/data1/malto/unlearning_llm/token.txt"
    with open(path, 'r') as f:
        return f.read().strip()

class UnlearningDataset(torch.utils.data.Dataset):
  def __init__(self,path,tofu):
    if tofu:
      self.dataset=load_dataset("locuslab/TOFU", path)
      self.dataset_X=pd.DataFrame(self.dataset["train"]["question"])
      self.dataset_Y=pd.DataFrame(self.dataset["train"]["answer"])
    # TO DO : when all data is available create same for the dataset for cahllenge

    
    
  def __len__(self):
    return len(self.dataset_X)
  def __getitem__(self, index):
    return self.dataset_X[0][index],self.dataset_Y[0][index]
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
    X, y,  = (
        batch["X"].to(device),
        batch["y"].to(device),
    )
    outputs = model(X)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")


    # GA or GD.
    position_loss = loss_fct(outputs, y)
    if operation == "ga":  # Negative the direction for GA.
        position_loss = -position_loss


    return position_loss


    
def GradientAscentTrainLoop(model,forget_set,validation_set,retain_set,epoch,device,optimizer,alpha,gamma):
  """
  Training Loop that uses gradient ascent algorithm

  :param model: model used for training
  :param forget_set: forget set part of data set
  :param validation_set: validation set part of data set
  :param retain_Set retain set part of data set
  :param alpha (int): The parameter α balances the noise injection and the fine-tuning term, α equal to 0 corresponds to simple fine-tuning
  :param epoch: number of epochs 
  :param device: device for the training
  :param criterion : loss function
  :param meanloss : meanloss function that calculates mean loss of validation set
  :param optimizer : optimizer used for training
  :param scheduler (int) : Adjusts  𝛼 dynamically

  :returns: trained model
  """
  model.to(device)
  model.train()
  for forget_epoch in range(epoch):
    epoch_loss=0
    for forget,retain in zip(forget_set,retain_set):
      model.zero_grad()
      L_f=get_answer_loss("ga",forget,model)
      L_r=get_answer_loss("gd",retain,model)
      loss=alpha*L_f+gamma*L_r
      epoch_loss+=loss.item()
      loss.backward()
      optimizer.step()



    print(f"Epoch {forget_epoch}/{epoch}, Train Loss: {epoch_loss} Validation Loss: {L_dv:.4f}")

  return model