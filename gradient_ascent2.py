
import torch
import wandb
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM


def gradient_ascent(current_model,batch, device):
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    ce=torch.nn.CrossEntropyLoss()
    loss_ce=ce(normal_outputs,batch["labels"].to(device))
    return -loss_ce
 

def GATrainingLoop(unlearnmodel,train_set,val_set,epoch,device,optimizer,project_name,config):
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
  wandb.init(
    # set the wandb project where this run will be logged
    project=project_name,

    # track hyperparameters and run metadata
    config=config
)
  if config["model_type"]=="1B":
      tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
  elif config["model_type"]=="7B":
      tokenizer=AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
  print(config)

  unlearnmodel.to(device) ##student
#challenge's pre trained model for retain set (good teacher)
  unlearnmodel.train()
  for forget_epoch in range(epoch):

    epoch_loss=0
    batch_no=1
    for batch in train_set:
        optimizer.zero_grad()
        loss=gradient_ascent(unlearnmodel,batch,device)
        wandb.log({"Loss":loss.item()})

        
        

        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_no} Batch Type:{batch['split']} Loss:{loss.item()}")
        batch_no+=1
    total_val_loss=0
    with torch.no_grad():
        for batch in val_set:
            val_loss=gradient_ascent(unlearnmodel,batch,device)
            wandb.log({"Val Loss":val_loss.item()})
            
            
            print(f"Batch Val Loss : {val_loss.item()}")
            total_val_loss+=val_loss.item()
    print(f"Epoch {forget_epoch+1}, Train Loss: {epoch_loss/len(train_set)} Validation Loss: {total_val_loss/len(val_set):.4f}")
    unlearnmodel.save_pretrained(f"{config["file_name"]}_epoch_{forget_epoch+1}")
    tokenizer.save_pretrained(f"{config["file_name"]}_epoch_{forget_epoch+1}")

  

  return unlearnmodel