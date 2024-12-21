
import torch
import wandb
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
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
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss




    
def GradientDifferenceTrainLoop(model,forget_set,retain_set,val_forget_set,val_retain_set,epoch,device,optimizer,alpha,gamma,project_name,config):
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
  model.to(device)
  model.train()
  for forget_epoch in range(epoch):

    epoch_loss=0
    batch_no=1
    for forget,retain in zip(forget_set,retain_set):
        optimizer.zero_grad()
        L_f=get_answer_loss("ga",forget,model,device)
        L_r=get_answer_loss("gd",retain,model,device)
        loss=alpha*L_f+gamma*L_r

        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_no} Loss:{loss.item()}")
        wandb.log({"Batch Loss":loss.item()})
        batch_no+=1
    total_val_loss=0
    with torch.no_grad():
        for val_f,val_r in zip(val_forget_set,val_retain_set):
            val_L_f=get_answer_loss("ga",val_f,model,device)
            val_L_r=get_answer_loss("gd",val_r,model,device)
            val_loss=alpha*val_L_f+gamma*val_L_r
            wandb.log({"Batch Val Loss":val_loss.item()})
            print(f"Batch Val Loss : {val_loss.item()}")
            total_val_loss+=val_loss.item()
    print(f"Epoch {forget_epoch}, Train Loss: {epoch_loss/len(forget_set)} Validation Loss: {total_val_loss/len(val_forget_set):.4f}")

  

  return model
def GradientAscentTrainingLoop(model,forget_set,val_forget_set,epoch,device,optimizer,project_name,config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=project_name,

    # track hyperparameters and run metadata
    config=config
      )
    for forget_epoch in range(epoch):
      epoch_loss=0
      batch_no=1
      for forget in forget_set:
        model.zero_grad()
        loss=get_answer_loss("ga",forget,model,device)
        
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_no} Loss:{loss.item()}")
        wandb.log({"Batch Loss":loss.item()})
        batch_no+=1
      total_val_loss=0
      with torch.no_grad():
        for val_f in val_forget_set:
              val_loss=get_answer_loss("ga",val_f,model,device)
              wandb.log({"Batch Val Loss":val_loss.item()})
              print(f"Batch Val Loss : {val_loss.item()}")
              total_val_loss+=val_loss.item()
      print(f"Epoch {forget_epoch}, Train Loss: {epoch_loss} Validation Loss: {total_val_loss:.4f}")
    return model
   