import torch
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM


def unlearning(retain_path,forget_path,model_path,output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    class UnlearningDataset(torch.utils.data.Dataset):
        def __init__(self,data,tokenizer):
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
                "split":1 if self.data.iloc[index]["split"]=="forget" else 0
            }
    #Preparing data
    retain_df = pd.read_parquet(retain_path, engine='pyarrow') 
    forget_df = pd.read_parquet(forget_path, engine='pyarrow') 
    train_data=pd.concat([retain_df,forget_df],ignore_index=True)
    dataset=UnlearningDataset(train_data,tokenizer)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True) #TODO Ask claudio for batch size
    unlearn_model=AutoModelForCausalLM.from_pretrained(model_path)
    bad_teacher = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
    good_teacher=AutoModelForCausalLM.from_pretrained(model_path)
    optimizer=torch.optim.SGD(unlearn_model.parameters(),lr=0.000006)
    device="cuda" if torch.cuda.is_available() else "cpu"
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

    unlearn_model.to(device) ##student
    bad_teacher.to(device) #orginal OLBO 1B for forget set (bad teacher)
    good_teacher.to(device) #challenge's pre trained model for retain set (good teacher)
    unlearn_model.train()
    bad_teacher.eval()
    good_teacher.eval()
    for forget_epoch in range(1):
        for batch in dataloader:
            print("Batch")
            optimizer.zero_grad()
            loss=cross_entropy(bad_teacher,unlearn_model,good_teacher,batch,device)
            loss.backward()
            optimizer.step()
    unlearn_model.save_pretrained(output_path) #TODO Ask could we save the model at every epoch and then overwrite it bcs of the time limt?
    tokenizer.save_pretrained(output_path)

    
def main():
    path = "/data1/malto/unlearning_llm/"
    model_path = path + 'models/semeval25-unlearning-model'
    dataset_path = path + 'datasets/semeval25-unlearning-data/'
    retain_path=dataset_path+'data/retain_train-00000-of-00001.parquet'
    forget_path=dataset_path+'data/forget_train-00000-of-00001.parquet'
    model_path="/home/amunis/Unlearning-sensitive-content-from-LLMs/orginal_model"
    output_path='/home/amunis/Unlearning-sensitive-content-from-LLMs/unlearn_model'
    unlearning(retain_path,forget_path,model_path,output_path)

if __name__=="__main__":
    main()