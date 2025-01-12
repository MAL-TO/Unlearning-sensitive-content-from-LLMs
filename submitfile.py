import torch
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM

def unlearning(model_path,output_path,forget_path,retain_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    class UnlearningDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer,data):
            # Load the appropriate tokenizer
            self.tokenizer = tokenizer
            

            # Tokenize the input and output with padding and truncation    
            self.data=data
        

            
            

        def __len__(self):
            return len(self.data)
        #Prompt + Answer
        #Prompt + Answer
        def __getitem__(self, index):
            prompt = self.tokenizer(self.data.iloc[index]["input"],padding="max_length",truncation=True, max_length=512, return_tensors=None)
            labels=self.tokenizer(f"{self.data.iloc[index]['input']} {self.data.iloc[index]['output']}",padding="max_length",truncation=True, max_length=512, return_tensors=None)
            attention_mask = prompt["attention_mask"]
            start_locs=self.tokenizer(self.data.iloc[index]["input"])


            return {
                "input_ids": torch.tensor(prompt["input_ids"]),
                "attention_mask": torch.tensor(attention_mask),
                "start_locs":len(start_locs["input_ids"])-1,
                "labels": torch.tensor(labels["input_ids"]),
                "split":1 if self.data.iloc[index]["split"]=="forget" else 0,

            }
    #Preparing data
    retain_df = pd.read_parquet(f"{retain_path}/retain.parquet", engine='pyarrow') 
    forget_df = pd.read_parquet(f"{forget_path}/forget.parquet", engine='pyarrow') 
    train_data=pd.concat([retain_df,forget_df],ignore_index=True)
    dataset=UnlearningDataset(tokenizer,train_data)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True) 
    unlearn_model=AutoModelForCausalLM.from_pretrained(model_path)
    good_teacher=AutoModelForCausalLM.from_pretrained(model_path)
    optimizer=torch.optim.SGD(unlearn_model.parameters(),lr=0.0001)
    device="cuda" if torch.cuda.is_available() else "cpu"
    def kl_divergence(current_model,good_teacher,batch, device):
        normal_outputs = current_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        with torch.no_grad():
            good_teacher_outputs = good_teacher(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
        # P: pretrained model; Q: current model.
        l=torch.unsqueeze(batch["split"],-1)
        l=torch.unsqueeze(l,-1)
        bad_teacher=torch.normal(mean = 0, 
                                        std = 1, 
                                        size = good_teacher_outputs.logits.shape).cuda() + torch.ones(good_teacher_outputs.logits.shape[-1]).cuda()
        prob_p = torch.nn.functional.softmax(bad_teacher.to(device), -1)
        prob_f = torch.nn.functional.softmax(good_teacher_outputs.logits, -1)
        prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

        out_teacher= (1-l.to(device))*prob_f + l.to(device)*prob_p


        loss = (out_teacher * (torch.log(out_teacher + 1e-12) - torch.log(prob_q + 1e-12))).sum(-1).mean()

        return loss

    unlearn_model.to(device) 
    good_teacher.to(device) 
    unlearn_model.train()
    good_teacher.eval()
    for forget_epoch in range(2):
        for batch in dataloader:
            optimizer.zero_grad()
            
            loss=kl_divergence(unlearn_model,good_teacher,batch,device)
            loss.backward()
            optimizer.step()
    unlearn_model.save_pretrained(output_path) 
    tokenizer.save_pretrained(output_path)

