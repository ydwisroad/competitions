import pandas as pd
import numpy as np
import torch
import logging
from glob import glob
import os
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from transformers import AutoConfig
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from tqdm import tqdm, trange
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from model import *
from HelperClasses import CommonLitDataset,CommonLitPredDataset

def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
    return data

def get_optimizer_params(model):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = 5e-5
    no_decay = ['bias', 'gamma', 'beta']
    group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
    group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
    group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.01,
         'lr': learning_rate / 2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.01,
         'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.01,
         'lr': learning_rate * 2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': learning_rate / 2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': learning_rate * 2.6},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr': 1e-3, "momentum": 0.99},
    ]
    return optimizer_parameters


def make_model(model_name='../../data/bert/roberta-base/', num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    config = AutoConfig.from_pretrained(model_name)
    config.update({'num_labels': num_labels})
    model = CommonLitModel(model_name, config=config)
    return model, tokenizer


def make_optimizer(model, optimizer_name="AdamW"):
    optimizer_grouped_parameters = get_optimizer_params(model)
    kwargs = {
        'lr': 5e-5,
        'weight_decay': 0.01,
        # 'betas': (0.9, 0.98),
        # 'eps': 1e-06
    }
    if optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "Adam":
        from torch.optim import Adam
        optimizer = Adam(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))

def make_scheduler(optimizer, decay_name='linear', t_max=None, warmup_steps=None):
    if decay_name == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    elif decay_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_type))
    return scheduler


def make_loader(
        data,
        tokenizer,
        max_len,
        batch_size,
        fold=0
):
    train_set, valid_set = data[data['kfold'] != fold], data[data['kfold'] == fold]
    train_dataset = CommonLitDataset(train_set, tokenizer, max_len)
    valid_dataset = CommonLitDataset(valid_set, tokenizer, max_len)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size // 2,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    return train_loader, valid_loader


def make_loader_pred(
        data,
        tokenizer,
        max_len,
        batch_size,
):
    test_dataset = CommonLitPredDataset(data, tokenizer, max_len, is_test=True)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size // 2,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=0
    )

    return test_loader

def config(fold=0):
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    epochs = 200
    max_len = 250
    batch_size = 16

    return epochs



