"lr":learning_Rate
"optimizer":optimizer (sgd,adam,adamw)
"model_type":(1B or 7B)
"epochs":epoch number,
"device": device type,
"batch_size":batch size,
"project_name": project name for wandb,
"file_name":output model file name,
"task_type": task types (Task1,Task2,Task3,All),
"train_type_data": train data type (All,forget,retain),
"extra_model":(true,false) (If you trained a model before and you want to apply more epochs, make true, unless make false),
"model_path":If your extra_model is true, give the path of model (If false, this space won't affect),
"bad_teacher":(random,false) If you give random the model uses random logits as bad teacher, if you give false it will use orginal olmo 1B model as bad teacher