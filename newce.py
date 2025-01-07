def cross_entropy2(pretrained_model, current_model, full_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        full_model: The full original model for reference.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    # Compute the outputs for the current model
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
        
    losses=[]
    loss = -(out_teacher * torch.log(prob_q + 1e-12)).sum(-1)
    # Compute the loss
    for x in range(batch["input_ids"].shape[0]):
        
        position_weight = torch.zeros_like(batch["input_ids"][x])
        position_weight[batch["start_locs"][x]:]=1
        position_weight=position_weight.to(device)
        position_weight = position_weight / position_weight.sum()
        print(position_weight)
        one_loss = (position_weight * loss[x]).sum(-1)
        losses.append(one_loss)

    final_loss = torch.stack(losses).mean()
    



    return final_loss