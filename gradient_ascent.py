
import torch
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
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device)
    )
    outputs = model(input_ids,attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")


    # GA or GD.
    position_loss = loss_fct(outputs["logits"].permute(0,2,1), labels)
    if operation == "ga":  # Negative the direction for GA.
        position_loss = -position_loss


    return position_loss.mean()


    
def GradientDifferenceTrainLoop(model,forget_set,retain_set,val_forget_set,val_retain_set,epoch,device,optimizer,alpha,gamma):
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

  

  return model
def GradientAscentTrainingLoop(model,forget_set,val_forget_set,epoch,device,optimizer):
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
   