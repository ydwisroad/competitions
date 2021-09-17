class PretrainConfig:
    train_file= 'mlm_data.csv'
    validation_file = 'mlm_data.csv'
    validation_split_percentage= 5
    pad_to_max_length= True
    model_name_or_path= 'roberta-base'
    config_name= 'roberta-base'
    tokenizer_name= 'roberta-base'
    use_slow_tokenizer= True
    per_device_train_batch_size= 8
    per_device_eval_batch_size= 8
    learning_rate= 5e-5
    weight_decay= 0.0
    num_train_epochs= 1 # change to 5
    max_train_steps= None
    gradient_accumulation_steps= 1
    lr_scheduler_type= 'constant_with_warmup'
    num_warmup_steps= 0
    output_dir= 'output'
    seed= 2021
    model_type= 'roberta'
    max_seq_length= None
    line_by_line= False
    preprocessing_num_workers= 4
    overwrite_cache= True
    mlm_probability= 0.15

class FineTuneConfig:
    fold = 5
    epochs = 200
    max_len = 250
    batch_size = 16

class PredictConfig:
    max_len = 250
    batch_size = 16




