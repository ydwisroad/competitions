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

class CommonLitModel(nn.Module):
    def __init__(
            self,
            model_name,
            config,
            multisample_dropout=False,
            output_hidden_states=False
    ):
        super(CommonLitModel, self).__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        # self.regressor = nn.Linear(config.hidden_size*2, 1)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[1]
        sequence_output = self.layer_norm(sequence_output)

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(sequence_output))
            else:
                logits += self.regressor(dropout(sequence_output))

        logits /= len(self.dropouts)

        # calculate loss
        loss = None
        if labels is not None:
            # regression task
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(labels.dtype)
            loss = torch.sqrt(loss_fn(logits, labels.view(-1)))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
