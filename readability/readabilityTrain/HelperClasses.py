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

gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace('\n', '')
    tok = tokenizer.encode_plus(
        data,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([tokenizer.pad_token_id] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
    return curr_sent

class CommonLitDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        if 'excerpt' in self.data.columns:
            self.excerpts = self.data.excerpt.values.tolist()
        else:
            self.excerpts = self.data.text.values.tolist()
        self.targets = self.data.target.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        excerpt, label = self.excerpts[item], self.targets[item]
        features = convert_examples_to_features(
            excerpt, self.tokenizer,
            self.max_len, self.is_test
        )
        return {
            'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(features['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.double),
        }


class CommonLitPredDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt, label = self.excerpts[item], self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer,
                self.max_len, self.is_test
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'token_type_ids': torch.tensor(features['token_type_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.double),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer,
                self.max_len, self.is_test
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'token_type_ids': torch.tensor(features['token_type_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
            }

class LambOptimizer(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas=(0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0,
            clamp_value: float = 10,
            adam: bool = False,
            debias: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

class Trainer:
    def __init__(self, model, optimizer, scheduler, scalar=None, log_interval=1, evaluate_interval=1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scalar = scalar
        self.log_interval = log_interval
        self.evaluate_interval = evaluate_interval
        self.evaluator = Evaluator(self.model, self.scalar)

    def train(self, train_loader, valid_loader, epoch,
              result_dict, tokenizer, fold):
        count = 0
        losses = AverageMeter()
        self.model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], batch_data['attention_mask'], \
                                                                batch_data['token_type_ids'], batch_data['label']
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), labels.to(device)

            if self.scalar is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

            loss, logits = outputs[:2]
            count += labels.size(0)
            losses.update(loss.item(), input_ids.size(0))

            if self.scalar is not None:
                self.scalar.scale(loss).backward()
                self.scalar.step(self.optimizer)
                self.scalar.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))
                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_loader.sampler),
                                                                               100 * count / len(train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(losses.avg),
                ]
                logger.info(', '.join(ret))

            if batch_idx % self.evaluate_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader,
                    epoch,
                    result_dict,
                    tokenizer
                )
                if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
                    logger.info("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch,
                                                                                           result_dict['val_loss'][-1]))
                    result_dict["best_val_loss"] = result_dict['val_loss'][-1]
                    torch.save(self.model.state_dict(), f"model{fold}.bin")

        result_dict['train_loss'].append(losses.avg)
        return result_dict

class Evaluator:
    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar

    def worst_result(self):
        ret = {
            'loss': float('inf'),
            'accuracy': 0.0
        }
        return ret

    def result_to_str(self, result):
        ret = [
            'epoch: {epoch:0>3}',
            'loss: {loss: >4.2e}'
        ]
        for metric in self.evaluation_metrics:
            ret.append('{}: {}'.format(metric.name, metric.fmtstr))
        return ', '.join(ret).format(**result)

    def save(self, result):
        with open('result_dict.json', 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def load(self):
        result = self.worst_result
        if os.path.exists('result_dict.json'):
            with open('result_dict.json', 'r') as f:
                try:
                    result = json.loads(f.read())
                except:
                    pass
        return result

    def evaluate(self, data_loader, epoch, result_dict, tokenizer):
        losses = AverageMeter()

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], batch_data[
                    'attention_mask'], batch_data['token_type_ids'], batch_data['label']
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), labels.to(device)

                if self.scalar is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )

                loss, logits = outputs[:2]
                losses.update(loss.item(), input_ids.size(0))

        logger.info('----Validation Results Summary----')
        logger.info('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, losses.avg))

        result_dict['val_loss'].append(losses.avg)
        return result_dict


class EvaluatorPred:
    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar

    def evaluate(self, data_loader, tokenizer):
        preds = []
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids = batch_data['input_ids'], batch_data['attention_mask'], \
                                                            batch_data['token_type_ids']
                input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device)

                if self.scalar is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )

                logits = outputs[0].detach().cpu().numpy().squeeze().tolist()
                preds += logits
        return preds
