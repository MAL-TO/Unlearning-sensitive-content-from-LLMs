from good_bad_teacher import ConfigManager, DataManager, ModelManager, TeacherStudentUnlearning
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from gradient_ascent import *
from claudiosmethod import *
import sys
from kl_minimization import *
from gradient_ascent2 import *
from gradient_difference import *
path = "/data3/csavelli/unlearning_llm/models/"



def main():
    # take what is written after the command for the name of the configuration file
    
    args = sys.argv[1:]

    # Initialize configuration
    with open(args[0], 'r') as file:
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
    train_set,val_set=prepare_data(config["model_type"],config["batch_size"],config["task_type"],config["train_type_data"])

    if config["train_type"]=="gtbt":
        final_model=ClaudioTrainLoop(model,good_teacher,train_set,val_set,config["epochs"],config["device"],optimizer,config["project_name"],config)
    elif config["train_type"]=="kl":
        final_model=KlMinTrainingLoop(model,good_teacher,train_set,val_set,config["epochs"],config["device"],optimizer,config["project_name"],config)
    elif config["train_type"]=="ga":
        final_model=GATrainingLoop(model,train_set,val_set,config["epochs"],config["device"],optimizer,config["project_name"],config)
        
    elif config["train_type"]=="gd":
        final_model=GDTrainingLoop(model,train_set,val_set,config["epochs"],config["device"],optimizer,config["project_name"],config)






if __name__ == "__main__":
    main()