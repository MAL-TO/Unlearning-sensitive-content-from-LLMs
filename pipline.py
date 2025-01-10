from good_bad_teacher import ConfigManager, DataManager, ModelManager, TeacherStudentUnlearning
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from gradient_ascent import *
from claudiosmethod import *
path = "/data1/malto/unlearning_llm/"



def main():
    # Initialize configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    if config['extra_model']=='true':
        model = AutoModelForCausalLM.from_pretrained(config["model_path"])
        print("Model is loaded")
    else:
        model=model_loader(config["model_type"])
    good_teacher=model_loader(config["model_type"])
    if config["optimizer"]=="adam":
        optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    elif config["optimizer"]=="sgd":
        optimizer=torch.optim.SGD(model.parameters(),lr=config["lr"])
    elif config["optimizer"]=="adamw":
        optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"])
    if config["model_type"]=="7B":
        path = "/data1/malto/unlearning_llm/"
        good_teacher_path = path + 'models/semeval25-unlearning-model'
    else:
        path = "/data1/malto/unlearning_llm/"
        good_teacher_path = path + 'models/semeval25-unlearning-model'+'-1B-model'
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
        train_set,val_set=prepare_data(config["model_type"],config["batch_size"],config["task_type"],config["train_type_data"],good_teacher_path,config["device"])

        final_model=GradientDifferenceTrainLoop(model,train_set,val_set,config["epochs"],config["device"]
                                            ,optimizer,config["project_name"],config)
        final_model.save_pretrained(config["file_name"])
        tokenizer.save_pretrained(config["file_name"])

    elif config["train_type"]=="cl":
        train_set,val_set=prepare_data(config["model_type"],config["batch_size"],config["task_type"],config["train_type_data"])

        
        final_model=ClaudioTrainLoop(model,good_teacher,train_set,val_set,config["epochs"],config["device"],optimizer,config["project_name"],config)




if __name__ == "__main__":
    main()