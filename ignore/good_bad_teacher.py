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
in the new changed theTeacherTrainer class will no longer be used for training the teachers.
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
import wandb

class TeacherStudentUnlearning:
    def __init__(self, good_teacher, bad_teacher, student, config):
        self.good_teacher = good_teacher
        self.bad_teacher = bad_teacher
        self.student = student
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wandb.init(
            project="llm-unlearning",
            config={
                "architecture": "teacher_student",
                "good_teacher": config['model']['good_teacher']['path'],
                "bad_teacher": config['model']['bad_teacher']['model_id'],
                "learning_rate": config['training']['learning_rate'],
                "epochs": config['training']['num_epochs'],
                "retain_alpha": config['retain']['alpha'],
                "retain_gamma": config['retain']['gamma'],
                "forget_beta": config['forget']['beta'],
                "forget_gamma": config['forget']['gamma']
            }
        )
        model_name = 'allenai/OLMo-1B-0724-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train_student(self, retain_loader, forget_loader, validation_loader, num_epochs):
        self.good_teacher.to(self.device)
        self.bad_teacher.to(self.device)
        self.student.to(self.device)
        validation_frequency = self.config['training']['validation_frequency']

        optimizer = torch.optim.SGD(self.student.parameters(), lr=self.config['training']['learning_rate'])
        
        # Initialize tracking variables for metrics and history
        training_history = {
            'retain_losses': [],
            'forget_losses': [],
            'val_losses': [],
            'val_metrics': []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training steps
            forget_loss = self.train_forget_step(forget_loader, optimizer)
            retain_loss = self.train_retain_step(retain_loader, optimizer)

            # Log training metrics
            wandb.log({
                "epoch": epoch + 1,
                "retain_loss": retain_loss,
                "forget_loss": forget_loss,
                "total_loss": retain_loss + forget_loss
            })

            # Store training history
            training_history['retain_losses'].append(retain_loss)
            training_history['forget_losses'].append(forget_loss)
            
            # Validate only at specified frequency
            if (epoch + 1) % validation_frequency == 0:
                print(f"Running validation at epoch {epoch+1}")
                val_loss, val_metrics = self.validate(validation_loader)
                
                # Store validation results
                training_history['val_losses'].append(val_loss)
                training_history['val_metrics'].append(val_metrics)

                # Log validation metrics
                wandb.log({
                    "val_loss": val_loss,
                    "val_perplexity": val_metrics['perplexity'],
                    "val_good_teacher_agreement": val_metrics['good_teacher_agreement'],
                    "val_bad_teacher_divergence": val_metrics['bad_teacher_divergence']
                })
                
                # Print validation results
                print(f"Validation Loss: {val_loss:.4f}")
                print("Validation Metrics:", val_metrics)
            else:
                # Store None for epochs where we don't validate
                training_history['val_losses'].append(None)
                training_history['val_metrics'].append(None)
            
            # Print epoch results
            print(f"Retain Loss: {retain_loss:.4f}")
            print(f"Forget Loss: {forget_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print("Validation Metrics:", val_metrics)
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % self.config.get('training', {}).get('checkpoint_frequency', 5) == 0:
                checkpoint_path = os.path.join(self.config['checkpoints_dir'], f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    metrics=val_metrics
                )

                # Add wandb artifact logging for checkpoints
                artifact = wandb.Artifact(
                    f"checkpoint_epoch_{epoch+1}", 
                    type="model",
                    description=f"Model checkpoint from epoch {epoch+1}"
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
                print(f"Checkpoint saved for epoch {epoch+1}")
                    
        return training_history
    
    def train_student_new(self, retain_loader, forget_loader, validation_loader, num_epochs):
        # Move models to device
        self.good_teacher.to(self.device)
        self.bad_teacher.to(self.device)
        self.student.to(self.device)
        
        # Ensure teachers are in eval mode and gradients are disabled
        self.good_teacher.eval()
        self.bad_teacher.eval()
        
        validation_frequency = self.config['training']['validation_frequency']
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.config['training']['learning_rate'])
        
        # Initialize tracking variables
        training_history = {
            'combined_losses': [],
            'retain_losses': [],
            'forget_losses': [],
            'val_losses': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_retain_losses = []
            epoch_forget_losses = []
            
            # Set student to training mode
            self.student.train()
            
            # Use tqdm for progress bar
            total_batches = min(len(retain_loader), len(forget_loader))
            pbar = tqdm(zip(retain_loader, forget_loader), total=total_batches)
            
            for retain_batch, forget_batch in pbar:
                try:
                    optimizer.zero_grad()
                    
                    # Move batches to device and get batch sizes
                    retain_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
                    forget_batch = {k: v.to(self.device) for k, v in forget_batch.items()}
                    
                    retain_size = retain_batch['input_ids'].size(0)
                    forget_size = forget_batch['input_ids'].size(0)
                    total_size = retain_size + forget_size
                    
                    # Calculate proportional weights
                    retain_weight = retain_size / total_size
                    forget_weight = forget_size / total_size
                    
                    # Retain loss calculation
                    with torch.no_grad():
                        good_teacher_output = self.good_teacher(**retain_batch)
                    student_retain_output = self.student(**retain_batch)
                    retain_loss = self.calculate_retain_loss(
                        student_retain_output.logits,
                        good_teacher_output.logits,
                        retain_batch
                    )
                    
                    # Forget loss calculation
                    with torch.no_grad():
                        bad_teacher_output = self.bad_teacher(**forget_batch)
                    student_forget_output = self.student(**forget_batch)
                    forget_loss = self.calculate_forget_loss(
                        student_forget_output.logits,
                        bad_teacher_output.logits,
                        forget_batch
                    )
                    
                    # Combine losses with proportional weighting
                    combined_loss = (retain_weight * retain_loss + 
                                forget_weight * forget_loss)
                    
                    # Backward pass and optimization
                    combined_loss.backward()
                    optimizer.step()
                    
                    # Store losses for monitoring
                    epoch_retain_losses.append(retain_loss.item())
                    epoch_forget_losses.append(forget_loss.item())
                    
                    # Update progress bar
                    pbar.set_description(f"Loss: {combined_loss.item():.4f}")
                    
                except RuntimeError as e:
                    print(f"Error in batch processing: {str(e)}")
                    continue
            
            # Calculate epoch averages
            avg_retain_loss = sum(epoch_retain_losses) / len(epoch_retain_losses)
            avg_forget_loss = sum(epoch_forget_losses) / len(epoch_forget_losses)
            
            # Store in training history
            training_history['retain_losses'].append(avg_retain_loss)
            training_history['forget_losses'].append(avg_forget_loss)
            
            # Validation step
            if (epoch + 1) % validation_frequency == 0 and validation_loader is not None:
                self.student.eval()
                val_loss, val_metrics = self.validate(validation_loader)
                training_history['val_losses'].append(val_loss)
                training_history['val_metrics'].append(val_metrics)
                
                print(f"\nEpoch {epoch+1} Results:")
                print(f"Average Retain Loss: {avg_retain_loss:.4f}")
                print(f"Average Forget Loss: {avg_forget_loss:.4f}")
                print(f"Validation Loss: {val_loss:.4f}")
                print("Validation Metrics:", val_metrics)
            
            # Save checkpoint if needed
            if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self.save_checkpoint(
                    path=os.path.join(self.config['checkpoints_dir'], f'checkpoint_epoch_{epoch+1}.pt'),
                    epoch=epoch,
                    val_loss=val_loss if 'val_loss' in locals() else None,
                    metrics=val_metrics if 'val_metrics' in locals() else None
                )
        
        return training_history


    def validate(self, validation_loader):
        """Perform validation and calculate metrics"""
        self.student.eval()
        total_loss = 0
        total_retain_kl = 0
        total_forget_kl = 0
        num_batches = 0
        
        metrics = {
            'perplexity': 0,
            'good_teacher_agreement': 0,
            'bad_teacher_divergence': 0
        }
        
        with torch.no_grad():
            for batch in validation_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get predictions from all models
                student_output = self.student(**batch)
                good_teacher_output = self.good_teacher(**batch)
                bad_teacher_output = self.bad_teacher(**batch)
                
                # Calculate losses
                task_loss = student_output.loss
                retain_kl = self.kl_divergence(
                    student_output.logits,
                    good_teacher_output.logits
                )
                forget_kl = self.kl_divergence(
                    student_output.logits,
                    bad_teacher_output.logits
                )
                
                # Update running totals
                total_loss += task_loss.item()
                total_retain_kl += retain_kl.item()
                total_forget_kl += forget_kl.item()
                
                # Calculate perplexity
                metrics['perplexity'] += torch.exp(task_loss).item()
                
                # Calculate agreement/divergence scores
                metrics['good_teacher_agreement'] += self.calculate_agreement(
                    student_output.logits,
                    good_teacher_output.logits
                )
                metrics['bad_teacher_divergence'] += self.calculate_agreement(
                    student_output.logits,
                    bad_teacher_output.logits
                )
                
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return avg_loss, metrics
    
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
        
        
        # Compute and print averages
        self._print_evaluation_results(results)
        
        if results:
            wandb.log({
                "eval/retain_regurgitation_score": np.mean(results['retain']['regurgitation-score']),
                "eval/retain_knowledge_score": np.mean(results['retain']['knowledge-score']),
                "eval/retain_perplexity": np.mean(results['retain']['perplexity']),
                "eval/forget_regurgitation_score": np.mean(results['forget']['regurgitation-score']),
                "eval/forget_knowledge_score": np.mean(results['forget']['knowledge-score']),
                "eval/forget_perplexity": np.mean(results['forget']['perplexity'])
            })
            
            if 'kl_div_good_teacher' in results['retain']:
                wandb.log({
                    "eval/retain_kl_div": np.mean(results['retain']['kl_div_good_teacher']),
                    "eval/forget_kl_div": np.mean(results['forget']['kl_div_bad_teacher'])
                })
                
        return results
    
    def finish(self):
        """Clean up wandb run"""
        wandb.finish()

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

    def save_checkpoint(self, path, epoch, val_loss, metrics):
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'good_teacher_state_dict': self.good_teacher.state_dict(),
            'bad_teacher_state_dict': self.bad_teacher.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.good_teacher.load_state_dict(checkpoint['good_teacher_state_dict'])
        self.bad_teacher.load_state_dict(checkpoint['bad_teacher_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['metrics']
    
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
        
        return beta * kl_loss + gamma * task_loss
    
    @staticmethod
    def kl_divergence(student_logits, teacher_logits):
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
    
    def calculate_agreement(self, logits1, logits2):
        """
        Calculate token-level agreement between two sets of prediction logits
        
        Args:
            logits1: First set of model logits (typically from student)
            logits2: Second set of model logits (typically from teacher)
            
        Returns:
            float: Agreement score between 0 and 1, where 1 means perfect agreement
        """
        # Get the most likely token at each position
        preds1 = torch.argmax(logits1, dim=-1)  # Shape: [batch_size, sequence_length]
        preds2 = torch.argmax(logits2, dim=-1)  # Shape: [batch_size, sequence_length]
        
        # Calculate percentage of matching predictions
        return (preds1 == preds2).float().mean().item()
    
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
        self.model_path = config['model']['good_teacher']['path']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token = load_token()
        
        # Get bad teacher config and convert torch_dtype from string to actual dtype
        self.bad_teacher_config = config['model']['bad_teacher']
        if isinstance(self.bad_teacher_config['torch_dtype'], str):
            self.bad_teacher_config['torch_dtype'] = getattr(torch, self.bad_teacher_config['torch_dtype'])
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
        bad_teacher = self.load_bad_teacher()
        
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
    
    def load_bad_teacher(self):
        """Load a tiny pre-trained model as bad teacher"""
        try:
            # Load with minimal configuration for memory efficiency
            model = AutoModelForCausalLM.from_pretrained(
                self.bad_teacher_config['model_id'],
                torch_dtype=self.bad_teacher_config['torch_dtype'],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            return model
        except Exception as e:
            print(f"Error loading bad teacher from {self.bad_teacher_config['model_id']}")
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
    Data Manager class to handle data loading and preparation from parquet files
    output: retain_loader, forget_loader, validation_loader
    '''
    def __init__(self, config):
        self.config = config
        self.base_path = "/data1/malto/unlearning_llm"
        self.dataset_path = os.path.join(self.base_path, 'datasets/semeval25-unlearning-data/')
        
        model_name = 'allenai/OLMo-1B-0724-hf'
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
            
        # Load all datasets
        self.retain_train = pd.read_parquet(
            os.path.join(self.dataset_path, 'data/retain_train-00000-of-00001.parquet'), 
            engine='pyarrow'
        )
        self.retain_validation = pd.read_parquet(
            os.path.join(self.dataset_path, 'data/retain_validation-00000-of-00001.parquet'), 
            engine='pyarrow'
        )
        self.forget_train = pd.read_parquet(
            os.path.join(self.dataset_path, 'data/forget_train-00000-of-00001.parquet'), 
            engine='pyarrow'
        )
        self.forget_validation = pd.read_parquet(
            os.path.join(self.dataset_path, 'data/forget_validation-00000-of-00001.parquet'), 
            engine='pyarrow'
        )
    
    def load_data(self):
        """Return the raw dataframes"""
        return {
            'retain_train': self.retain_train,
            'retain_validation': self.retain_validation,
            'forget_train': self.forget_train,
            'forget_validation': self.forget_validation
        }

    def create_dataloaders(self, batch_size=8):
        """Create DataLoader objects for training and validation"""
        # Create datasets
        retain_train_dataset = UnlearningDataset(self.retain_train, self.tokenizer)
        retain_val_dataset = UnlearningDataset(self.retain_validation, self.tokenizer)
        forget_train_dataset = UnlearningDataset(self.forget_train, self.tokenizer)
        forget_val_dataset = UnlearningDataset(self.forget_validation, self.tokenizer)
        
        # Create dataloaders
        retain_train_loader = DataLoader(
            retain_train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        retain_val_loader = DataLoader(
            retain_val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        forget_train_loader = DataLoader(
            forget_train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        forget_val_loader = DataLoader(
            forget_val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return retain_train_loader, forget_train_loader, retain_val_loader, forget_val_loader
    
class UnlearningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract input and output from the dataframe
        self.inputs = data['input'].tolist()
        self.outputs = data['output'].tolist()
            
        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set special tokens for formatting
        self.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
        self.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Format input with special tokens
        text = f"{self.bos_token}{self.inputs[idx]}{self.outputs[idx]}{self.eos_token}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input IDs and attention mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels by shifting input_ids right
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left by 1
        labels[-1] = self.tokenizer.pad_token_id  # Last token predicts padding
        
        # Mask out padding tokens in labels with -100
        labels = labels.masked_fill(attention_mask == 0, -100)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class ConfigManager:
    def __init__(self):
        # Base path from the server setup
        base_path = "/data1/malto/unlearning_llm"
        
        self.config = {
            'model': {
                'good_teacher': {
                    'path': f"{base_path}/models/semeval25-unlearning-model-1B-model",
                    'type': 'base_model',
                },
                'bad_teacher': {
                    'model_id': 'EleutherAI/pythia-70m',
                    'torch_dtype': 'float16'
                }
            },
            'training': {
                'num_epochs': 1,
                'learning_rate': 1e-5,
                'checkpoint_frequency': 5, # needs to be changed
                'validation_frequency': 1  # Minimum change to qualify as an improvement
            },
            'checkpoints_dir': '/data1/malto/unlearning_llm/modelss', 
            'validation': {
                'batch_size': 16,
                'metrics': ['perplexity', 'agreement', 'divergence']
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
            },
            'max_new_tokens': 256 
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
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Initialize wandb
        wandb.init(
            project="llm-unlearning",
            config=config
        )
        
        # Keep rest of initialization
        model_manager = ModelManager(config)
        data_manager = DataManager(config)
        
        # Load and prepare data
        retain_train_loader, forget_train_loader, retain_val_loader, forget_val_loader = data_manager.create_dataloaders(batch_size=8)
        
        # Initialize models
        good_teacher, bad_teacher = model_manager.initialize_teachers()
        student = model_manager.initialize_student()
        
        # Freeze teachers
        model_manager.freeze_teachers(good_teacher, bad_teacher)
        
        # Initialize unlearning system
        unlearning = TeacherStudentUnlearning(good_teacher, bad_teacher, student, config)
        
        # Train student
        training_history = unlearning.train_student(retain_train_loader, forget_train_loader, retain_val_loader, config['training']['num_epochs'])
        
        # Final evaluation
        final_results = unlearning.evaluate(retain_val_loader, forget_val_loader)
        
        # Log final results summary
        wandb.run.summary.update({
            "final_retain_regurgitation": np.mean(final_results['retain']['regurgitation-score']),
            "final_retain_knowledge": np.mean(final_results['retain']['knowledge-score']),
            "final_forget_regurgitation": np.mean(final_results['forget']['regurgitation-score']),
            "final_forget_knowledge": np.mean(final_results['forget']['knowledge-score']),
            "training_epochs": config['training']['num_epochs'],
            "best_validation_loss": min(filter(None, training_history['val_losses']))
        })

        # Save final results to a file
        with open(os.path.join(config['checkpoints_dir'], 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
            
        # Log final results file as artifact
        final_results_artifact = wandb.Artifact(
            "final_results", 
            type="results",
            description="Final evaluation results"
        )
        final_results_artifact.add_file(os.path.join(config['checkpoints_dir'], 'final_results.json'))
        wandb.log_artifact(final_results_artifact)
        
        # Save config
        config_manager.save_config("config.json")
        
        # Clean up wandb
        unlearning.finish()
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        wandb.finish(exit_code=1)
        raise e