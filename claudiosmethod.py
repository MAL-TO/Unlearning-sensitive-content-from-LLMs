
import torch
import wandb
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss




    
def ClaudioTrainLoop(unlearnmodel,fullmodel,pretrainedmodel,train_set,val_set,epoch,device,optimizer,project_name,config):
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
  unlearnmodel.to(device)
  pretrainedmodel.to(device)
  fullmodel.to(device)
  unlearnmodel.train()
  for forget_epoch in range(epoch):

    epoch_loss=0
    batch_no=1
    for batch in train_set:
        optimizer.zero_grad()
        if batch["split"][0]=="forget":
           loss=compute_kl(pretrainedmodel,unlearnmodel,batch,device)
        elif batch["split"][0]=="retain":
           loss=compute_kl(fullmodel,unlearnmodel,batch,device)
        

        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_no} Batch Type:{batch["split"]} Loss:{loss.item()}")
        wandb.log({"Batch Loss":loss.item()})
        batch_no+=1
    total_val_loss=0
    with torch.no_grad():
        for batch in val_set:
            if batch["split"][0]=="forget":
                val_loss=compute_kl(pretrainedmodel,unlearnmodel,batch,device)
            elif batch["split"][0]=="retain":
                val_loss=compute_kl(fullmodel,unlearnmodel,batch,device)
            wandb.log({"Batch Val Loss":val_loss.item()})
            print(f"Batch Val Loss : {val_loss.item()}")
            total_val_loss+=val_loss.item()
    print(f"Epoch {forget_epoch}, Train Loss: {epoch_loss/len(train_set)} Validation Loss: {total_val_loss/len(val_set):.4f}")

  

  return unlearnmodel