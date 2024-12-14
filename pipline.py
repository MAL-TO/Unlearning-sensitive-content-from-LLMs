from utils import GradientAscentTrainLoop
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(train_type,model_type):
    # Initialize configuration
    path = "/data1/malto/unlearning_llm/"

    if train_type=="1B":
        ## Fetch and load model:
        model_path = path + 'models/semeval25-unlearning-model'
        #snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning', token=hf_token, local_dir=model_path+'-1B-model')
        model = AutoModelForCausalLM.from_pretrained(model_path+'-1B-model')
    elif train_type=="7B":
        ## Fetch and load model:
        model_path = path + 'models/semeval25-unlearning-model'
        #snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning', token=hf_token, local_dir=model_path+'-1B-model')
        model = AutoModelForCausalLM.from_pretrained(model_path)

    
    if train_type=="gtbt":
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
    elif train_type=="ga":

        GradientAscentTrainLoop()
