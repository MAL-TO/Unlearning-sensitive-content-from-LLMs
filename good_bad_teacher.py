'''
step 1:
Train Good Teacher on retain data
Train Bad Teacher on forget data
Both teachers are frozen after training
step 2:
For each input:
    Get predictions from Good Teacher
    Get predictions from Bad Teacher
    Get predictions from Student
    Calculate divergence between Student and both Teachers
    Update Student to move towards Good Teacher and away from Bad Teacher

Total_Loss = α * KL(Student||Good_Teacher) - β * KL(Student||Bad_Teacher) + γ * Task_Loss

1. Data Preparation

2. Teacher Training:
   Retain Data → Good Teacher
   Forget Data → Bad Teacher

3. Student Training:
   Input → Student → Predictions
         ↓
   Good Teacher → KL Divergence
         ↓
   Bad Teacher  → KL Divergence
         ↓
   Loss Calculation → Model Update

4. Evaluation:
   Test Data → Models → Metrics → Logging
'''
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from huggingface_hub import snapshot_download
from utils import *


class TeacherStudentUnlearning:
    def __init__(self, good_teacher, bad_teacher, student, config):
        self.good_teacher = good_teacher
        self.bad_teacher = bad_teacher
        self.student = student
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, dataloader, num_epochs):
        self.good_teacher.to(self.device)
        self.bad_teacher.to(self.device)
        self.student.to(self.device)

        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.config['lr'])

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                loss = self.training_step(batch, optimizer)
                total_loss += loss
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss}")
            
            if (epoch + 1) % self.config['training']['eval_frequency'] == 0:
                self.evaluate()
        
    def training_step(self, batch, optimizer):
        optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get predictions from all models
        with torch.no_grad():
            good_teacher_output = self.good_teacher(**batch)
            bad_teacher_output = self.bad_teacher(**batch)
        
        student_output = self.student(**batch)
        
        # Calculate losses
        loss = self.calculate_combined_loss(
            student_output.logits,
            good_teacher_output.logits,
            bad_teacher_output.logits,
            batch
        )
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def evaluate(self):
        self.student.eval()
        # todo: implement evaluation
        pass

    def save_checkpoint(self, path):
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'good_teacher_state_dict': self.good_teacher.state_dict(),
            'bad_teacher_state_dict': self.bad_teacher.state_dict(),
        }, path)
        

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.good_teacher.load_state_dict(checkpoint['good_teacher_state_dict'])
        self.bad_teacher.load_state_dict(checkpoint['bad_teacher_state_dict'])
    
    def calculate_combined_loss(self, student_logits, good_teacher_logits, bad_teacher_logits, batch):
        alpha = self.config['loss']['alpha']
        beta = self.config['loss']['beta']
        gamma = self.config['loss']['gamma']
        
        kl_good = self.kl_divergence(student_logits, good_teacher_logits)
        kl_bad = self.kl_divergence(student_logits, bad_teacher_logits)
        # todo: implement task loss calculation
        task_loss = self.calculate_task_loss(student_logits, batch)
        
        return alpha * kl_good - beta * kl_bad + gamma * task_loss
    
    @staticmethod
    def kl_divergence(student_logits, teacher_logits):
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
    
    def calculate_task_loss(self, student_logits, batch):
        pass

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.base_path = "/data1/malto/unlearning_llm"
        self.model_path = os.path.join(self.base_path, 'models/semeval25-unlearning-model-1B-model')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token = load_token()
        
        # Ensure model is downloaded
        self._ensure_model_exists()
    
    def _ensure_model_exists(self):
        """Ensure the model exists locally, download if it doesn't"""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}, downloading...")
            snapshot_download(
                repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning',
                token=self.token,
                local_dir=self.model_path
            )
    
    def initialize_teachers(self):
        print("Initializing good teacher...")
        good_teacher = self._load_model()
        
        print("Initializing bad teacher...")
        bad_teacher = self._load_model()
        
        return good_teacher, bad_teacher

    def initialize_student(self):
        print("Initializing student...")
        return self._load_model()

    def _load_model(self):
        """Helper method to load model with consistent parameters"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            return model
        except Exception as e:
            print(f"Error loading model from {self.model_path}")
            print(f"Error details: {str(e)}")
            raise

    def freeze_teachers(self, good_teacher, bad_teacher):
        print("Freezing teacher models...")
        for teacher in [good_teacher, bad_teacher]:
            for param in teacher.parameters():
                param.requires_grad = False

class TeacherTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_good_teacher(self,teacher, dataloader):
        teacher.to(self.device)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=self.config['training']['learning_rate'])
        
        for epoch in range(self.config['training']['num_epochs']):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Training Good Teacher - Epoch {epoch+1}"):
                loss = self.training_step(teacher, batch, optimizer)
                total_loss += loss
                
            avg_loss = total_loss / len(dataloader)
            print(f"Good Teacher Epoch {epoch+1} Average Loss: {avg_loss}")


    def train_bad_teacher(self, teacher, dataloader):
        teacher.to(self.device)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=self.config['training']['learning_rate'])
        
        for epoch in range(self.config['training']['num_epochs']):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Training Bad Teacher - Epoch {epoch+1}"):
                loss = self.training_step(teacher, batch, optimizer)
                total_loss += loss
                
            avg_loss = total_loss / len(dataloader)
            print(f"Bad Teacher Epoch {epoch+1} Average Loss: {avg_loss}")

    def training_step(self, model, batch, optimizer):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validate_teachers(self, good_teacher, bad_teacher, val_dataloader):
        good_teacher.eval()
        bad_teacher.eval()
        # Implement validation metrics
        pass

class DataManager:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.retain_data = load_dataset("locuslab/TOFU", 'retain')['train']
        self.forget_data = load_dataset("locuslab/TOFU", 'forget')['train']

    def create_dataloaders(self, batch_size=8):
        retain_dataset = UnlearningDataset(self.retain_data, self.tokenizer)
        forget_dataset = UnlearningDataset(self.forget_data, self.tokenizer)
        return DataLoader(retain_dataset, batch_size=batch_size, shuffle=True), DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    
class UnlearningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.questions = data['question']
        self.answers = data['answer']
            
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        text = f"{self.questions[idx]}{self.answers[idx]}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class ConfigManager:
    def __init__(self):
        # Base path from the server setup
        base_path = "/data1/malto/unlearning_llm"
        
        self.config = {
            'model': {
                'path': f"{base_path}/models/semeval25-unlearning-model-1B-model",
                'type': 'base_model',
            },
            'training': {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'num_epochs': 1,
                'eval_frequency': 1,
            },
            'loss': {
                'alpha': 1.0,  # Good Teacher weight
                'beta': 0.5,   # Bad Teacher weight
                'gamma': 1.0   # Task loss weight
            },
            'data': {
                'base_path': f"{base_path}/datasets/semeval25-unlearning-data",
                'max_length': 512
            }
        }
        
    def load_config(self, path):
        with open(path, 'r') as f:
            self.config = json.load(f)
        
    def save_config(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
    def update_config(self, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

def main():
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.config
    
    # Initialize components
    model_manager = ModelManager(config)
    data_manager = DataManager(config)
    
    # Load and prepare data
    retain_data, forget_data = data_manager.load_data()
    retain_loader, forget_loader = data_manager.create_dataloaders(retain_data, forget_data)
    
    # Initialize models
    good_teacher, bad_teacher = model_manager.initialize_teachers()
    student = model_manager.initialize_student()
    
    # Train teachers
    teacher_trainer = TeacherTrainer(config)
    teacher_trainer.train_good_teacher(good_teacher, retain_loader)
    teacher_trainer.train_bad_teacher(bad_teacher, forget_loader)
    
    # Freeze teachers
    model_manager.freeze_teachers(good_teacher, bad_teacher)
    
    # Initialize unlearning system
    unlearning = TeacherStudentUnlearning(good_teacher, bad_teacher, student, config)
    
    # Train student
    unlearning.train(retain_loader, config['training']['num_epochs'])
    
    # Save final models
    unlearning.save_checkpoint("final_checkpoint.pt")
    config_manager.save_config("config.json")