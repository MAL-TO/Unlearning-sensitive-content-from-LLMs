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
   Retain Data + Forget Data→ Good Teacher
   Retain Data → Bad Teacher

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
from datasets import load_dataset, concatenate_datasets
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
from evaluate_generations import rouge_scorer

class TeacherStudentUnlearning:
    def __init__(self, good_teacher, bad_teacher, student, config):
        self.good_teacher = good_teacher
        self.bad_teacher = bad_teacher
        self.student = student
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['path'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train_student(self, retain_loader, forget_loader, num_epochs):
        self.good_teacher.to(self.device)
        self.bad_teacher.to(self.device)
        self.student.to(self.device)

        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.config['lr'])

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            retain_loss = self.train_retain_step(retain_loader, optimizer)
            forget_loss = self.train_forget_step(forget_loader, optimizer)

            print(f"Retain Loss: {retain_loss:.4f}")
            print(f"Forget Loss: {forget_loss:.4f}")

            if (epoch + 1) % self.config["training"]['eval_frequency'] == 0:
                self.evaluate()
    
    def train_retain_step(self, retain_loader, optimizer):
        self.student.train()
        total_loss = 0

        for batch in tqdm(retain_loader, desc="Training Retain"):
            optimizer.zero_grad()

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                good_teacher_output = self.good_teacher(**batch)
            
            student_outputs = self.student(**batch)
            loss = self.calculate_retain_loss(
                student_outputs.logits,
                good_teacher_output.logits,
                batch
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(retain_loader)
    
    def train_forget_step(self, forget_loader, optimizer):
        self.student.train()
        total_loss = 0

        for batch in tqdm(forget_loader, desc="Training Forget"):
            optimizer.zero_grad()

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                bad_teacher_output = self.bad_teacher(**batch)
            
            student_outputs = self.student(**batch)
            loss = self.calculate_forget_loss(
                student_outputs.logits,
                bad_teacher_output.logits,
                batch
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(forget_loader)

    def evaluate(self, retain_loader=None, forget_loader=None):
        """
        Evaluates the student model's performance on both retain and forget data.
        """
        if retain_loader is None or forget_loader is None:
            return
                
        self.student.eval()
        results = {
            'retain': {
                'regurgitation-score': [],
                'knowledge-score': [],
                'kl_div_good_teacher': [],
                'perplexity': []
            },
            'forget': {
                'regurgitation-score': [],
                'knowledge-score': [],
                'kl_div_bad_teacher': [],
                'perplexity': []
            }
        }
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        # Evaluate on retain data
        print("\nEvaluating on retain data...")
        with torch.no_grad():
            for batch in tqdm(retain_loader, desc="Retain Evaluation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get predictions from student and good teacher
                student_output = self.student(**batch)
                good_teacher_output = self.good_teacher(**batch)
                
                # Calculate metrics
                self._calculate_metrics(
                    student_output, 
                    good_teacher_output, 
                    batch, 
                    scorer, 
                    results['retain']
                )
        
        # Evaluate on forget data
        print("\nEvaluating on forget data...")
        with torch.no_grad():
            for batch in tqdm(forget_loader, desc="Forget Evaluation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get predictions from student and bad teacher
                student_output = self.student(**batch)
                bad_teacher_output = self.bad_teacher(**batch)
                
                # Calculate metrics
                self._calculate_metrics(
                    student_output, 
                    bad_teacher_output, 
                    batch, 
                    scorer, 
                    results['forget']
                )
        
        # Compute and print averages
        self._print_evaluation_results(results)
        
        return results

    def _calculate_metrics(self, student_output, teacher_output, batch, scorer, results_dict):
        """Helper method to calculate metrics for both retain and forget evaluation"""
        # Calculate KL divergence
        kl_div = self.kl_divergence(
            student_output.logits,
            teacher_output.logits
        ).item()
        
        # Calculate perplexity
        perplexity = torch.exp(student_output.loss).item()
        
        # Generate text for ROUGE and knowledge score
        generated_ids = self.student.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_new_tokens=self.config['max_new_tokens'],
            do_sample=False
        )
        
        # Decode outputs
        generated_text = self.tokenizer.batch_decode(
            generated_ids[:, batch['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        labels = self.tokenizer.batch_decode(
            batch['input_ids'],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    
        # Calculate ROUGE and knowledge scores
        for pred, target in zip(generated_text, labels):
            rouge_scores = scorer.score(target, pred)
            results_dict['regurgitation-score'].append(rouge_scores['rougeL'].recall)
            results_dict['knowledge-score'].append(
                int(pred.strip().lower() == target.strip().lower())
            )
        
        # Store KL divergence and perplexity
        if 'kl_div_good_teacher' in results_dict:
            results_dict['kl_div_good_teacher'].append(kl_div)
        else:
            results_dict['kl_div_bad_teacher'].append(kl_div)
        results_dict['perplexity'].append(perplexity)

    def _print_evaluation_results(self, results):
        """Helper method to print evaluation results"""
        print("\nEvaluation Results:")
        
        for data_type in ['retain', 'forget']:
            print(f"\n{data_type.upper()} DATA:")
            metrics = results[data_type]
            
            print(f"Regurgitation Score: {np.mean(metrics['regurgitation-score']):.4f}")
            print(f"Knowledge Score: {np.mean(metrics['knowledge-score']):.4f}")
            print(f"Perplexity: {np.mean(metrics['perplexity']):.4f}")
            
            if 'kl_div_good_teacher' in metrics:
                print(f"KL Divergence from Good Teacher: {np.mean(metrics['kl_div_good_teacher']):.4f}")
            else:
                print(f"KL Divergence from Bad Teacher: {np.mean(metrics['kl_div_bad_teacher']):.4f}")

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
    
    def calculate_retain_loss(self, student_logits, teacher_logits, batch):
        """Calculate loss for retain step - student should match good teacher"""
        alpha = self.config['retain']['alpha']  # Weight for teacher matching
        gamma = self.config['retain']['gamma']  # Weight for task loss
        
        # KL divergence to match good teacher
        kl_loss = self.kl_divergence(student_logits, teacher_logits)
        
        # Task-specific loss
        task_loss = self.calculate_task_loss(student_logits, batch)
        
        return alpha * kl_loss + gamma * task_loss
    
    def calculate_forget_loss(self, student_logits, teacher_logits, batch):
        """Calculate loss for forget step - student should diverge from bad teacher"""
        beta = self.config['forget']['beta']    # Weight for teacher divergence
        gamma = self.config['forget']['gamma']  # Weight for task loss
        
        # Negative KL divergence to move away from bad teacher
        kl_loss = self.kl_divergence(student_logits, teacher_logits)
        
        # Task-specific loss (with lower weight for forget data)
        task_loss = self.calculate_task_loss(student_logits, batch)
        
        return -beta * kl_loss + gamma * task_loss
    
    @staticmethod
    def kl_divergence(student_logits, teacher_logits):
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
    
    def calculate_task_loss(self, student_logits, batch):
        """
        Calculate task-specific loss using cross-entropy between student predictions and labels
        """
        # Get labels by shifting input_ids left
        labels = batch['input_ids'].clone()
        labels = labels[:, 1:].contiguous()  # Remove first token
        student_logits = student_logits[:, :-1, :].contiguous()  # Remove last prediction
        
        # Create loss mask from attention mask if available
        if 'attention_mask' in batch:
            loss_mask = batch['attention_mask'][:, 1:].contiguous()
        else:
            loss_mask = torch.ones_like(labels)
            
        # Calculate cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = student_logits.view(-1, student_logits.size(-1))
        shift_labels = labels.view(-1)
        
        # Calculate loss only on non-padded tokens
        per_token_loss = loss_fct(shift_logits, shift_labels)
        per_token_loss = per_token_loss.view(labels.size())
        masked_loss = per_token_loss * loss_mask
        
        # Average loss over non-padded tokens
        loss = masked_loss.sum() / loss_mask.sum()
        
        return loss

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
        """
        Validate both teachers' performance on validation data
        """
        good_teacher.eval()
        bad_teacher.eval()
        
        metrics = {
            'good_teacher': {'loss': 0, 'perplexity': 0, 'accuracy': 0},
            'bad_teacher': {'loss': 0, 'perplexity': 0, 'accuracy': 0}
        }
        
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Evaluate good teacher
                good_outputs = good_teacher(**batch)
                good_loss = good_outputs.loss
                metrics['good_teacher']['loss'] += good_loss.item()
                metrics['good_teacher']['perplexity'] += torch.exp(good_loss).item()
                
                # Evaluate bad teacher
                bad_outputs = bad_teacher(**batch)
                bad_loss = bad_outputs.loss
                metrics['bad_teacher']['loss'] += bad_loss.item()
                metrics['bad_teacher']['perplexity'] += torch.exp(bad_loss).item()
                
                # Calculate accuracy for both teachers
                good_logits = good_outputs.logits
                bad_logits = bad_outputs.logits
                
                # Get predictions (assuming we're doing next token prediction)
                good_preds = torch.argmax(good_logits, dim=-1)
                bad_preds = torch.argmax(bad_logits, dim=-1)
                
                # Calculate accuracy (ignoring padding tokens)
                labels = batch['input_ids']
                padding_mask = batch['attention_mask']
                
                good_correct = ((good_preds == labels) * padding_mask).sum().item()
                bad_correct = ((bad_preds == labels) * padding_mask).sum().item()
                total_tokens = padding_mask.sum().item()
                
                metrics['good_teacher']['accuracy'] += good_correct / total_tokens
                metrics['bad_teacher']['accuracy'] += bad_correct / total_tokens
        
        # Average metrics across batches
        for teacher in metrics:
            for metric in metrics[teacher]:
                metrics[teacher][metric] /= num_batches
        
        # Log results
        print("\nValidation Results:")
        print("Good Teacher:")
        print(f"  Loss: {metrics['good_teacher']['loss']:.4f}")
        print(f"  Perplexity: {metrics['good_teacher']['perplexity']:.4f}")
        print(f"  Accuracy: {metrics['good_teacher']['accuracy']:.4f}")
        
        print("\nBad Teacher:")
        print(f"  Loss: {metrics['bad_teacher']['loss']:.4f}")
        print(f"  Perplexity: {metrics['bad_teacher']['perplexity']:.4f}")
        print(f"  Accuracy: {metrics['bad_teacher']['accuracy']:.4f}")
        
        return metrics

class DataManager:
    '''
    Data Manager class to handle data loading and preparation
    output: retain_loader, forget_loader, all_loader
    '''
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.retain_data = load_dataset("locuslab/TOFU", 'retain90')['train'] # this has to be changed based on what we need now will give an error
        self.forget_data = load_dataset("locuslab/TOFU", 'forget01')['train']
        self.all_data = concatenate_datasets([self.retain_data, self.forget_data])
    
    def load_data(self):
        return self.retain_data, self.forget_data

    def create_dataloaders(self, batch_size=8):
        retain_dataset = UnlearningDataset(self.retain_data, self.tokenizer)
        forget_dataset = UnlearningDataset(self.forget_data, self.tokenizer)
        all_dataset = retain_dataset + forget_dataset
        return DataLoader(retain_dataset, batch_size=batch_size, shuffle=True), DataLoader(forget_dataset, batch_size=batch_size, shuffle=True), DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
    
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
                'num_epochs': 10,
                'eval_frequency': 1,
                'lr': 1e-5
            },
            'retain': {
                'alpha': 1.0,  # Good Teacher weight
                'gamma': 1.0   # Task loss weight
            },
            'forget': {
                'beta': 0.5,   # Bad Teacher weight
                'gamma': 1.0   # Task loss weight
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
    retain_data, forget_data, all_data = data_manager.load_data()
    retain_loader, forget_loader, all_loader = data_manager.create_dataloaders(retain_data, forget_data, all_data)
    
    # Initialize models
    good_teacher, bad_teacher = model_manager.initialize_teachers()
    student = model_manager.initialize_student()
    
    # Train teachers
    teacher_trainer = TeacherTrainer(config)
    teacher_trainer.train_good_teacher(good_teacher, all_loader)
    teacher_trainer.train_bad_teacher(bad_teacher, retain_loader)
    
    # Freeze teachers
    model_manager.freeze_teachers(good_teacher, bad_teacher)
    
    # Initialize unlearning system
    unlearning = TeacherStudentUnlearning(good_teacher, bad_teacher, student, config)
    
    # Train student
    unlearning.train_student(retain_loader, forget_loader, config['training']['num_epochs'])
    
    # Save final models
    unlearning.save_checkpoint("final_checkpoint.pt")
    config_manager.save_config("config.json")