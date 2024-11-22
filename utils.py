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

    
def GradientAscentTrainLoop(model,forget_set,validation_set,retain_set,alpha,epoch,device,criterion,meanloss,optimizer,scheduler):
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
  for forget_epoch in range(epoch):
    if validation_set!=None:
      if forget_epoch % 10 == 0:
        L_dv=meanloss(validation_set,criterion,model,device)
    epoch_loss=0
    for X,y in forget_set:
      X_forget,y_forget=X.to(device),y.to(device)
      X_retain,y_retain=next(iter(retain_set))
      X_retain,y_retain=X_retain.to(device),y_retain.to(device)
      model.zero_grad()
      L_f=criterion(model(X_forget),y_forget)
      L_r=criterion(model(X_retain),y_retain)
      loss=alpha*torch.relu((L_dv-L_f)**2)+(1-alpha)*L_r
      epoch_loss+=loss.item()
      loss.backward()
      optimizer.step()
      alpha = scheduler_step(alpha, scheduler)

    print(f"Epoch {forget_epoch}/{epoch}, Alpha: {alpha:.4f}, Train Loss: {epoch_loss} Validation Loss: {L_dv:.4f}")

  return model