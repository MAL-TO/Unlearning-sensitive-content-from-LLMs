import torch
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os



def unlearning(retain_path, forget_path, model_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

    class UnlearningDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.tokenizer = tokenizer
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            prompt = self.tokenizer(self.data.iloc[index]["input"], padding="max_length", truncation=True, max_length=512, return_tensors=None)
            labels = self.tokenizer(f"{self.data.iloc[index]['input']} {self.data.iloc[index]['output']}", padding="max_length", truncation=True, max_length=512, return_tensors=None)
            attention_mask = prompt["attention_mask"]
            start_locs = self.tokenizer(self.data.iloc[index]["input"])
            
            # All data in a specific dataloader will have the same split value
            return {
                "input_ids": torch.tensor(prompt["input_ids"]),
                "attention_mask": torch.tensor(attention_mask),
                "start_locs": len(start_locs["input_ids"]) - 1,
                "labels": torch.tensor(labels["input_ids"]),
                "split": 1 if "forget" in self.data.iloc[index].get("split", "") else 0
            }
        
    def create_dataloader(data_df, tokenizer, batch_size=2, shuffle=True):
        """Helper function to create dataloaders for specific datasets"""
        dataset = UnlearningDataset(data_df, tokenizer)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Preparing separate dataloaders
    retain_df = pd.read_parquet(retain_path, engine='pyarrow')
    retain_df['split'] = 'retain'
    forget_df = pd.read_parquet(forget_path, engine='pyarrow')
    forget_df['split'] = 'forget'
    
    forget_dataloader = create_dataloader(forget_df, tokenizer)
    retain_dataloader = create_dataloader(retain_df, tokenizer)

    unlearn_model = AutoModelForCausalLM.from_pretrained(model_path)
    #bad_teacher = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
    good_teacher = AutoModelForCausalLM.from_pretrained(model_path)
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.000006)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def random_prediction_generator(logits_shape, device, noise_std=0.1):
        # Create uniform base probabilities
        vocab_size = logits_shape[-1]
        base_probs = torch.ones(logits_shape, device=device) / vocab_size
        
        # Add Gaussian noise
        noise = torch.randn(logits_shape, device=device) * noise_std
        
        # Add noise and ensure still valid probabilities
        noisy_probs = base_probs + noise
        noisy_probs = torch.clamp(noisy_probs, min=1e-10)
        noisy_probs = noisy_probs / noisy_probs.sum(dim=-1, keepdim=True)
        
        return noisy_probs

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

    # def compute_new_loss(current_model, full_model, batch, device):
    #     normal_outputs = current_model(
    #         batch["input_ids"].to(device),
    #         attention_mask=batch["attention_mask"].to(device),
    #     )

    #     split_mask = batch["split"].to(device)
    #     split_mask = split_mask.unsqueeze(-1).unsqueeze(-1)

    #     prob_f = torch.nn.functional.softmax(full_model_outputs.logits, -1)
    #     prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)
        
    #     # Generate random predictions for forget set
    #     random_probs = random_prediction_generator(prob_f.shape, device)
        
    #     # Combine predictions based on split
    #     out_teacher = (1-l)*prob_f + l*random_probs

    #     loss = -(out_teacher * torch.log(prob_q + 1e-12)).sum(-1).mean()
    #     return loss

    def compute_new_loss(current_model, full_model, batch, device):
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
            full_model_outputs = full_model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )


        # P: pretrained model; Q: current model.
        l=torch.unsqueeze(batch["split"],-1)
        l=torch.unsqueeze(l,-1).to(device)
        prob_f = torch.nn.functional.softmax(full_model_outputs.logits, -1)
        prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

        random_probs = torch.normal(mean = 0, 
                                    std = 1, 
                                    size = full_model_outputs.logits.shape).cuda() + torch.ones(full_model_outputs.logits.shape[-1]).cuda()
        random_probs = torch.nn.functional.softmax(random_probs, -1)
        out_teacher = (1-l)*prob_f + l*random_probs


        loss = -(out_teacher * torch.log(prob_q + 1e-12)).sum(-1).mean() 

        return loss
        

    def train_epochs(dataloader, num_epochs, phase_name):
        print(f"\nStarting {phase_name} training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            batch_losses = []
            for batch_idx, batch in enumerate(dataloader):
                print(f"Batch {batch_idx + 1}/{len(dataloader)}")
                optimizer.zero_grad()
                loss = compute_new_loss(unlearn_model, good_teacher, batch, device)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                print(f"loss {loss}")

            
            avg_epoch_loss = sum(batch_losses)/len(batch_losses)
            print(f"{phase_name} Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    # Move models to device
    unlearn_model.to(device)
    #bad_teacher.to(device)
    good_teacher.to(device)
    
    unlearn_model.train()
    #bad_teacher.eval()
    good_teacher.eval()

    # First phase: Train on forget set for 6 epochs
    train_epochs(forget_dataloader, 1, "Forget Set")
    
    # Save the model after forget set training
    forget_model_path = os.path.join(output_dir, 'forget_only_model')
    os.makedirs(forget_model_path, exist_ok=True)
    print("\nSaving model after forget set training...")
    unlearn_model.save_pretrained(forget_model_path)
    tokenizer.save_pretrained(forget_model_path)
    print("Forget set model saved successfully!")

    # Second phase: Train on retain set for 2 epochs
    # train_epochs(retain_dataloader, 2, "Retain Set")
    
    # Save the final model after both trainings
    # final_model_path = os.path.join(output_dir, 'complete_model')
    # os.makedirs(final_model_path, exist_ok=True)
    # print("\nSaving final model after retain set training...")
    # unlearn_model.save_pretrained(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    print("Final model saved successfully!")

def main():
    path = "/data1/malto/unlearning_llm/"
    model_path = "/data1/malto/unlearning_llm/models/semeval25-unlearning-model-1B-model"
    dataset_path = path + 'datasets/semeval25-unlearning-data/'
    retain_path = dataset_path + 'data/retain_train-00000-of-00001.parquet'
    forget_path = dataset_path + 'data/forget_train-00000-of-00001.parquet'
    output_path = '/home/ebayat/Unlearning-sensitive-content-from-LLMs/unlearned_model'
    unlearning(retain_path, forget_path, model_path, output_path)

if __name__ == "__main__":
    main()