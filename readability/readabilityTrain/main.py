import pandas as pd
import numpy as np

import argparse
import logging
import math
import os
import time
import math

import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed
)
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from tqdm import tqdm, trange
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from config import PretrainConfig,FineTuneConfig,PredictConfig
from utils import *
from HelperClasses import *
from model import *


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

def saveDataWithTextsOnly(trainCsv, testCsv):
    train = pd.read_csv(trainCsv)
    test = pd.read_csv(testCsv)

    mlm_data = train[['excerpt']]
    mlm_data = mlm_data.rename(columns={'excerpt': 'text'})
    mlm_data.to_csv('mlm_data.csv', index=False)

    mlm_data_val = test[['excerpt']]
    mlm_data_val = mlm_data_val.rename(columns={'excerpt': 'text'})
    mlm_data_val.to_csv('mlm_data_val.csv', index=False)

def stage1pretrain():
    logger.info("stage1pretrain starts")
    config = PretrainConfig()
    if config.train_file is not None:
        extension = config.train_file.split(".")[-1]
        assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
    if config.validation_file is not None:
        extension = config.validation_file.split(".")[-1]
        assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    saveDataWithTextsOnly("../../data/commonlitreadability/train.csv","../../data/commonlitreadability/test.csv")

    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if config.seed is not None:
        set_seed(config.seed)

    data_files = {}
    if config.train_file is not None:
        data_files["train"] = config.train_file
    if config.validation_file is not None:
        data_files["validation"] = config.validation_file
    extension = config.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)

    if config.config_name:
        modelconfig = AutoConfig.from_pretrained(config.config_name)
    elif config.model_name_or_path:
        modelconfig = AutoConfig.from_pretrained(config.model_name_or_path)
    else:
        modelconfig = CONFIG_MAPPING[config.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=not config.use_slow_tokenizer)
    elif config.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=not config.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            config.model_name_or_path,
            from_tf=bool(".ckpt" in config.model_name_or_path),
            config=modelconfig,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(modelconfig)

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if config.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if config.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({config.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(config.max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not config.overwrite_cache,
    )

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        load_from_cache_file=not config.overwrite_cache,
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=config.mlm_probability)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    else:
        config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_train_steps,
    )

    total_batch_size = config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(config.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / config.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= config.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(config.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        perplexity = math.exp(torch.mean(losses))

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

    if config.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(config.output_dir, save_function=accelerator.save)

def fineTuneRun(fold, model, tokenizer, optimizer, scheduler, scaler, train_loader, valid_loader, fineTuneConfig):
    logger.info('start to run %s' % (fold))

    trainer = Trainer(model, optimizer, scheduler, scaler)
    train_time_list = []

    result_dict = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': np.inf
    }

    for epoch in range(fineTuneConfig.epochs):
        result_dict['epoch'] = epoch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic1 = time.time()

        result_dict = trainer.train(train_loader, valid_loader, epoch,
                                    result_dict, tokenizer, fold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tic2 = time.time()
        train_time_list.append(tic2 - tic1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model, tokenizer, optimizer, scheduler, scaler, train_loader, valid_loader,
    gc.collect()
    return result_dict

def stage2finetune():
    fineTuneConfig = FineTuneConfig()

    logger.info("stage2finetune starts")
    train = pd.read_csv('../../data/commonlitreadability/train.csv')
    test = pd.read_csv('../../data/commonlitreadability/test.csv')

    train = create_folds(train, num_splits=5)
    gc.enable()

    #model, tokenizer, optimizer, scheduler, scaler, train_loader, valid_loader
    model, tokenizer = make_model(model_name='../../data/bert/roberta-base/', num_labels=1)
    train_loader, valid_loader = make_loader(
        train, tokenizer, max_len=fineTuneConfig.max_len,
        batch_size=fineTuneConfig.batch_size, fold=fineTuneConfig.fold
    )

    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = fineTuneConfig.epochs * num_update_steps_per_epoch
    warmup_proportion = 0
    if warmup_proportion != 0:
        warmup_steps = math.ceil((max_train_steps * 2) / 100)
    else:
        warmup_steps = 0

    optimizer = make_optimizer(model, "AdamW")
    scheduler = make_scheduler(
        optimizer, decay_name='cosine_warmup',
        t_max=max_train_steps,
        warmup_steps=warmup_steps
    )
    scaler = None

    result_list = []
    for fold in range(5):
        logger.info('----')
        logger.info(f'FOLD: {fold}')
        result_dict = fineTuneRun(fold, model, tokenizer, optimizer, scheduler, scaler, train_loader, valid_loader, fineTuneConfig)
        result_list.append(result_dict)
        logger.info('----')

    [logger.info("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]

    oof = np.zeros(len(train))
    for fold in tqdm(range(5), total=5):
        model, tokenizer = make_model()
        model.load_state_dict(
            torch.load(f'model{fold}.bin')
        )
        model.cuda()
        model.eval()
        val_index = train[train.kfold == fold].index.tolist()
        train_loader, val_loader = make_loader(train, tokenizer, 250, 16, fold=fold)
        # scalar = torch.cuda.amp.GradScaler()
        scalar = None
        preds = []
        for index, data in enumerate(val_loader):
            input_ids, attention_mask, token_type_ids, labels = data['input_ids'], data['attention_mask'], data[
                'token_type_ids'], data['label']
            input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

            loss, logits = outputs[:2]
            preds += logits.cpu().detach().numpy().tolist()
        oof[val_index] = preds

    round(np.sqrt(mean_squared_error(train.target.values, oof)), 4)


def predictRun(fold, test):
    predictConfig = PredictConfig()
    model, tokenizer = make_model(
        model_name='../../data/bert/roberta-base/',
        num_labels=1
    )
    model.load_state_dict(
        torch.load(f'./model{fold}.bin')
    )
    test_loader = make_loader_pred(
        test, tokenizer, max_len=predictConfig.max_len,
        batch_size=predictConfig.batch_size
    )

    if torch.cuda.device_count() >= 1:
        model = model.cuda()
    else:
        print("use cpu currently")

    scaler = None

    evaluator = EvaluatorPred(model, scaler)

    test_time_list = []

    # torch.cuda.synchronize()
    tic1 = time.time()

    preds = evaluator.evaluate(test_loader, tokenizer)

    # torch.cuda.synchronize()
    tic2 = time.time()
    test_time_list.append(tic2 - tic1)

    del model, tokenizer, test_loader, scaler
    gc.collect()
    # torch.cuda.empty_cache()

    return preds

def stage3prediction():
    logger.info("stage3finetune starts")
    pred_df = pd.DataFrame()

    test = pd.read_csv('../../data/commonlitreadability/test.csv')
    test = create_folds(test, num_splits=5)
    for fold in tqdm(range(5)):
        pred_df[f'fold{fold}'] = predictRun(fold, test)

    sub = pd.read_csv('../../data/commonlitreadability/sample_submission.csv')
    sub['target'] = pred_df.mean(axis=1).values.tolist()
    sub.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    print("start to execute the main program ")
    stage1pretrain()
    #stage2finetune()
    #stage3prediction()
