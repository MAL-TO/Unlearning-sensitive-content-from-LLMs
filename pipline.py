from good_bad_teacher import ConfigManager, DataManager, ModelManager, TeacherStudentUnlearning
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from gradient_ascent import *
path = "/data1/malto/unlearning_llm/"



def main():
    # Initialize configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    model=model_loader(config["model_type"])
    if config["optimizer"]=="adam":
        optimizer=torch.optim.Adam(model.parameters(),config["lr"])
    elif config["optimizer"]=="sgd":
        optimizer=torch.optim.SGD(model.parameters(),config["lr"])
    
    
    if config["train_type"]=="gtbt":
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
        
        # Train teachers (no teaching required since the models are already trained)
        # teacher_trainer = TeacherTrainer(config)
        # teacher_trainer.train_good_teacher(good_teacher, all_loader)
        # teacher_trainer.train_bad_teacher(bad_teacher, retain_loader)
        
        # Freeze teachers
        model_manager.freeze_teachers(good_teacher, bad_teacher)
        
        # Initialize unlearning system
        unlearning = TeacherStudentUnlearning(good_teacher, bad_teacher, student, config)
        
        # Train student
        unlearning.train_student(retain_loader, forget_loader, config['training']['num_epochs'])
        
        # Save final models
        unlearning.save_checkpoint("final_checkpoint.pt")
        config_manager.save_config("config.json")
    elif config["train_type"]=="gd":
        retain_t_dataloader,retain_v_dataloader,forget_t_dataloader,forget_v_dataloader=prepare_data(config["model_type"],config["batch_size"])

        final_model=GradientDifferenceTrainLoop(model,forget_t_dataloader,retain_t_dataloader,forget_v_dataloader,retain_v_dataloader,config["epochs"],config["device"]
                                            ,optimizer,config["alpha"],config["gamma"])
    elif config["train_type"]=="ga":
        retain_t_dataloader,retain_v_dataloader,forget_t_dataloader,forget_v_dataloader=prepare_data(config["model_type"],config["batch_size"])

        final_model=GradientAscentTrainingLoop(model,forget_t_dataloader,forget_v_dataloader,config["epochs"],config["device"]
                                            ,optimizer)

        