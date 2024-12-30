
import torch
import wandb
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
def cross_entropy(pretrained_model, current_model,full_model,batch, device):
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
    with torch.no_grad():
        full_model_outputs = full_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    l=torch.unsqueeze(batch["split"],-1)
    l=torch.unsqueeze(l,-1).to(device)
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_f = torch.nn.functional.softmax(full_model_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    out_teacher= (1-l)*prob_f + l*prob_p


    loss = -(out_teacher * torch.log(prob_q + 1e-12)).sum(-1).mean() 

    return loss

def kl_divergence(pretrained_model, current_model, full_model,batch, device):
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
    with torch.no_grad():
        full_model_outputs = full_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
    l=torch.unsqueeze(batch["split"],-1)
    l=torch.unsqueeze(l,-1).to(device)
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_f = torch.nn.functional.softmax(full_model_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    out_teacher= (1-l)*prob_f + l*prob_p


    loss = (out_teacher * (torch.log(out_teacher + 1e-12) - torch.log(prob_q + 1e-12))).sum(-1).mean()


    return loss
def newloss(pretrained_model, current_model,full_model,batch, device):
    tot_kl=0
    tot_ce=0


    

    for i in range(batch["split"].shape[0]):
        if batch["split"][i]==0:
            normal_outputs = current_model(
            batch["input_ids"][i].unsqueeze(0).to(device),
            attention_mask=batch["attention_mask"][i].unsqueeze(0).to(device),
            labels=batch["labels"][i].unsqueeze(0).to(device))
            with torch.no_grad():
                full_model_outputs = full_model(
                    batch["input_ids"][i].unsqueeze(0).to(device),
                    attention_mask=batch["attention_mask"][i].unsqueeze(0).to(device),
                    labels=batch["labels"][i].unsqueeze(0).to(device),
        )
            l=torch.unsqueeze(batch["split"],-1)
            l=torch.unsqueeze(l,-1).to(device)
            prob_f = torch.nn.functional.softmax(full_model_outputs.logits, -1)
            prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

            out_teacher= prob_f


            loss = (out_teacher * (torch.log(out_teacher + 1e-12) - torch.log(prob_q + 1e-12))).sum(-1).mean()
            tot_kl+=loss
        elif batch["split"][i]==1:
            normal_outputs = current_model(
            batch["input_ids"][i].unsqueeze(0).to(device),
            attention_mask=batch["attention_mask"][i].unsqueeze(0).to(device),
            labels=batch["labels"][i].unsqueeze(0).to(device))
            with torch.no_grad():
                pretrained_outputs = pretrained_model(
                batch["input_ids"][i].unsqueeze(0).to(device),
                attention_mask=batch["attention_mask"][i].unsqueeze(0).to(device),
                labels=batch["labels"][i].unsqueeze(0).to(device),
        )
            prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
            prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

            out_teacher= prob_p


            loss = -(out_teacher * torch.log(prob_q + 1e-12)).sum(-1).mean()
            tot_ce+=loss
    return tot_kl,tot_ce 









   




    
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
  unlearnmodel.to(device) ##student
  pretrainedmodel.to(device) #orginal OLBO 1B for forget set (bad teacher)
  fullmodel.to(device) #challenge's pre trained model for retain set (good teacher)
  unlearnmodel.train()
  for forget_epoch in range(epoch):

    epoch_loss=0
    batch_no=1
    for batch in train_set:
        optimizer.zero_grad()
        if config["loss"]=="kl":
           loss=kl_divergence(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
           wandb.log({"Loss":loss.item()})
        elif config["loss"]=="ce":
           loss=cross_entropy(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
           wandb.log({"Loss":loss.item()})
        elif config["loss"]=="mix":
            loss1=kl_divergence(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
            loss2=cross_entropy(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
            loss=config["alpha"]*loss1+config["gamma"]*loss2
            wandb.log({"Loss":loss.item()})
        elif config["loss"]=="mix2":
            kl,ce=newloss(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
            loss=config["alpha"]*kl+config["gamma"]*ce
            wandb.log({"Loss":loss.item()})

        
        

        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_no} Batch Type:{batch['split']} Loss:{loss.item()}")
        batch_no+=1
    total_val_loss=0
    with torch.no_grad():
        for batch in val_set:
            if config["loss"]=="kl":
                val_loss=kl_divergence(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
                wandb.log({"Val Loss":val_loss.item()})
            elif config["loss"]=="ce":
                val_loss=cross_entropy(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
                wandb.log({"Val Loss":val_loss.item()})
            elif config["loss"]=="mix":
                loss1=kl_divergence(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
                loss2=cross_entropy(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
                val_loss=config["alpha"]*loss1+config["gamma"]*loss2
                wandb.log({"Val Loss":loss.item()})
            elif config["loss"]=="mix2":
                kl,ce=newloss(pretrainedmodel,unlearnmodel,fullmodel,batch,device)
                val_loss=config["alpha"]*kl+config["gamma"]*ce
                wandb.log({"Val Loss":loss.item()})
            
            
            print(f"Batch Val Loss : {val_loss.item()}")
            total_val_loss+=val_loss.item()
    print(f"Epoch {forget_epoch}, Train Loss: {epoch_loss/len(train_set)} Validation Loss: {total_val_loss/len(val_set):.4f}")

  

  return unlearnmodel