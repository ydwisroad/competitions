#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install transformers


# In[2]:


import argparse
import logging
import os
import random
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from seqeval.metrics import classification_report

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# In[3]:


from collections import OrderedDict

from transformers import BertConfig
from transformers import RobertaConfig
from transformers import AlbertConfig, ElectraConfig
from transformers import XLMConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig
from transformers import AutoConfig, PretrainedConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel, RobertaModel # DistilBertModel, XLMModel, AlbertModel
from transformers import BertPreTrainedModel # AlbertPreTrainedModel, DistilBertPreTrainedModel, XLMPreTrainedModel, ElectraPreTrainedModel

from typing import List, Optional


# In[5]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# ## Input related functions

# In[6]:


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, valid_mask, segment_ids, label_ids, start_ids, end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.valid_mask = valid_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.start_ids = start_ids
        self.end_ids = end_ids


# In[7]:


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
#         return ['O','B-TECH','I-TECH']
        return ['O','B-FUNC','I-FUNC','B-POWER','I-POWER','B-SAP','I-SAP','B-TECH','I-TECH','B-QUALIFICATION','I-QUALIFICATION']


# In[8]:


def get_entities(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return set(chunks)


# In[9]:


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


# In[10]:


def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


# In[11]:


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    span_labels = []
    for label in label_list:
        label = label.split('-')[-1]
        if label not in span_labels:
            span_labels.append(label)
    span_map = {label: i for i, label in enumerate(span_labels)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        valid_mask = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            for i, word_token in enumerate(word_tokens):
                if i == 0:
                    valid_mask.append(1)
                else:
                    valid_mask.append(0)
                tokens.append(word_token)
        label_ids = [label_map[label] for label in example.labels]
        entities = get_entities(example.labels)
        start_ids = [span_map['O']] * len(label_ids)
        end_ids = [span_map['O']] * len(label_ids)
        for entity in entities:
            start_ids[entity[1]] = span_map[entity[0]]
            end_ids[entity[-1]] = span_map[entity[0]]
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        start_ids += [pad_token_label_id]
        end_ids += [pad_token_label_id]
        valid_mask.append(1)
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]
            valid_mask.append(1)
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            valid_mask.append(1)
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            start_ids = [pad_token_label_id] + start_ids
            end_ids = [pad_token_label_id] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            valid_mask.insert(0, 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            start_ids = ([pad_token_label_id] * padding_length) + start_ids
            end_ids = ([pad_token_label_id] * padding_length) + end_ids
            valid_mask = ([0] * padding_length) + valid_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            start_ids += [pad_token_label_id] * padding_length
            end_ids += [pad_token_label_id] * padding_length
            valid_mask += [0] * padding_length
        while (len(label_ids) < max_seq_length):
            label_ids.append(pad_token_label_id)
            start_ids.append(pad_token_label_id)
            end_ids.append(pad_token_label_id)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length
        assert len(valid_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("valid_mask: %s", " ".join([str(x) for x in valid_mask]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("start_ids: %s", " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s", " ".join([str(x) for x in end_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          valid_mask=valid_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          start_ids=start_ids,
                          end_ids=end_ids)
        )
    return features


# In[12]:


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


# In[13]:


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)
    return dataset


# ## Training related functions

# In[14]:


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


# In[15]:


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)


# In[16]:


def valid_sequence_output(sequence_output, valid_mask, attention_mask):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask


# In[17]:


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum'):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        return self._forward_alg(emissions, tags, mask, reduction)

    def _forward_alg(self,
                     emissions: torch.Tensor,
                     tags: torch.LongTensor,
                     mask: Optional[torch.ByteTensor] = None,
                     reduction: str = 'sum',
                     ) -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].bool().all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].bool().all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].bool().all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].bool().all()
        mask = mask.bool()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].bool().all()
        mask = mask.bool()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)
        best_tags_list = [item + [-1] * (seq_length - len(item)) for item in best_tags_list]
        best_tags_list = torch.from_numpy(np.array(best_tags_list))
        return torch.LongTensor(best_tags_list).cuda()


# In[18]:


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None,
            decode=False,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if decode:
            tags = self.crf.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs

        return outputs  # (loss), scores


# In[19]:


class AutoModelForCrfNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )


# In[20]:


MODEL_FOR_CRF_NER_MAPPING = OrderedDict(
    [
        # (XLMConfig, XLMCrfForNer),
        # (DistilBertConfig, DistilBertCrfForNer),
        # (RobertaConfig, RobertaCrfForNer),
        # (CamembertConfig, RobertaCrfForNer),
        # (XLMRobertaConfig, RobertaCrfForNer),
        (BertConfig, BertCrfForNer),
        # (AlbertConfig, AlbertCrfForNer),
        # (ElectraConfig, ElectraCrfForNer),
    ]
)


# In[21]:


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CRF_NER_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), ())
TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


# In[22]:


# classifier_parameters = [(n,p) for (n,p) in model.named_parameters() if args.model_type not in n]


# In[23]:


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()
    classifier_parameters = model.classifier.named_parameters()
    crf_parameters = model.crf.named_parameters()
    args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
    args.classifier_lr = args.classifier_lr if args.classifier_lr else args.learning_rate
    args.crf_lr = args.crf_lr if args.crf_lr else args.learning_rate
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},

        {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in crf_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.crf_lr},
        {"params": [p for n, p in crf_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.crf_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # adversarial_training
    if args.adv_training == 'fgm':
        adv = FGM(model=model, param_name='word_embeddings')
    elif args.adv_training == 'pgd':
        adv = PGD(model=model, param_name='word_embeddings')

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      "labels": batch[4], }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[3] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if args.adv_training:
                adv.adversarial_training(args, inputs, optimizer)

            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                              prefix=global_step)
                        for key, value in results.items():
                            if isinstance(value, float) or isinstance(value, int):
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    if best_score < results['tech_f1']:
                        best_score = results['tech_f1']
                        output_dir = os.path.join(args.output_dir, "best_checkpoint")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[24]:


def f1_score(true_entities, pred_entities):
    """Compute the F1 score."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def precision_score(true_entities, pred_entities):
    """Compute the precision."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(true_entities, pred_entities):
    """Compute the recall."""
    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def classification_report(true_entities, pred_entities, digits=5):
    """Build a text report showing the main classification metrics."""
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    tech_f1=0.0
    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_correct = len(type_true_entities & type_pred_entities)
        nb_pred = len(type_pred_entities)
        nb_true = len(type_true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        if type_name=='TECH':
            tech_f1 = f1

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(true_entities, pred_entities),
                             recall_score(true_entities, pred_entities),
                             f1_score(true_entities, pred_entities),
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report, tech_f1


# In[25]:


# def custom_metric(true_entities, pred_entities):
#     nb_correct = len(true_entities & pred_entities)
#     nb_pred = len(pred_entities)
#     nb_true = len(true_entities)

#     p = nb_correct / nb_pred if nb_pred > 0 else 0
#     r = nb_correct / nb_true if nb_true > 0 else 0
#     score = 2 * p * r / (p + r) if p + r > 0 else 0

#     return p, r, score


# In[26]:


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    trues = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      "labels": batch[4],
                      "decode": True}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, tags = outputs[:2]
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = tags.detach().cpu().numpy()
            trues = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, tags.detach().cpu().numpy(), axis=0)
            trues = np.append(trues, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    label_map = {i: label for i, label in enumerate(labels)}

    trues_list = [[] for _ in range(trues.shape[0])]
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(trues.shape[0]):
        for j in range(trues.shape[1]):
            if trues[i, j] != pad_token_label_id:
                trues_list[i].append(label_map[trues[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    class_report, tech_f1 = classification_report(true_entities, pred_entities)
    results = {
        "loss": eval_loss,
        "f1": f1_score(true_entities, pred_entities),
        'report': class_report,
        'tech_f1': tech_f1
    }

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****\n".format(prefix))
        writer.write("***** Eval loss : {} *****\n".format(eval_loss))
        for key in sorted(results.keys()):
            if key == 'report_dict':
                continue
            logger.info("{} = {}".format(key, str(results[key])))
            writer.write("{} = {}\n".format(key, str(results[key])))
    return results, preds_list


# In[27]:


# def test_metric(label, pred, beta=2):
#     match_label = set()
#     manual_only = set()
#     pred_only = set()

#     match_label = set(label).intersection(pred)

#     pred_only = set(pred) - match_label
#     manual_only = set(label) - match_label

#     precision = len(match_label)/len(set(pred)) if len(set(pred)) else 0
#     recall = len(match_label)/len(set(label))
#     f1 = (1+beta**2)*precision*recall/((beta**2)*precision+recall) if ((beta**2)*precision+recall) else 0

#     return [list(match_label), list(manual_only), list(pred_only), precision, recall, f1]


# In[28]:


# def test_evaluation(args, model, tokenizer, labels, pad_token_label_id, mode="test", prefix='test'):
#     eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

#     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
#     # Note that DistributedSampler samples randomly
#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(eval_dataset,
#                                  sampler=eval_sampler,
#                                  batch_size=args.eval_batch_size)

#     # multi-gpu evaluate
#     if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
#         model = torch.nn.DataParallel(model)

#     # Eval!
#     logger.info("***** Running evaluation %s *****", prefix)
#     logger.info("  Num examples = %d", len(eval_dataset))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     preds = None
#     trues = None
#     model.eval()
#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         batch = tuple(t.to(args.device) for t in batch)

#         with torch.no_grad():
#             inputs = {"input_ids": batch[0],
#                       "attention_mask": batch[1],
#                       "valid_mask": batch[2],
#                       "labels": batch[4], }
#             if args.model_type != "distilbert":
#                 inputs["token_type_ids"] = (
#                     batch[2] if args.model_type in ["bert", "xlnet"] else None
#                 )  # XLM and RoBERTa don"t use segment_ids
#             outputs = model(**inputs)
#             tmp_eval_loss, logits = outputs[:2]
#             if args.n_gpu > 1:
#                 tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
#             eval_loss += tmp_eval_loss.item()
#         nb_eval_steps += 1
#         if preds is None:
#             preds = logits.detach().cpu().numpy()
#             trues = inputs["labels"].detach().cpu().numpy()
#         else:
#             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#             trues = np.append(trues, inputs["labels"].detach().cpu().numpy(), axis=0)

#     eval_loss = eval_loss / nb_eval_steps
#     preds = np.argmax(preds, axis=2)
#     label_map = {i: label for i, label in enumerate(labels)}

#     trues_list = [[] for _ in range(trues.shape[0])]
#     preds_list = [[] for _ in range(preds.shape[0])]

#     for i in range(trues.shape[0]):
#         for j in range(trues.shape[1]):
#             if trues[i, j] != pad_token_label_id:
#                 trues_list[i].append(label_map[trues[i][j]])
#                 preds_list[i].append(label_map[preds[i][j]])
                
#     # Save predictions
#     output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
#     with open(output_test_predictions_file, "w") as writer:
#         with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
#             example_id = 0
#             for line in f:
#                 if line.startswith("-DOCSTART-") or line == "" or line == "\n":
#                     writer.write(line)
#                     if not preds_list[example_id]:
#                         example_id += 1
#                 elif preds_list[example_id]:
#                     output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
#                     writer.write(output_line)
#                 else:
#                     logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
    
#     test = pd.read_csv('/content/drive/MyDrive/skill scraper/datasets/test_df.csv', index_col=0)
#     test['original_index'] = test['original_index'].str.lower()

#     test['end_idx'] = test['start_idx']+test['token'].apply(lambda x: len(x))

#     pred=[]
#     with open('/content/drive/MyDrive/skill scraper/model_crf/test_predictions.txt', 'r') as f:
#         for line in f:
#             if line=='\n':
#                 pass
#             else:
#                 pred.append([line.split()[0], line.split()[1]])

#     pred = pd.DataFrame(pred, columns=['pred_token', 'pred_tag'])

#     test['space']=False
#     for i in test.index:
#         if len(tokenizer.tokenize(test.loc[i, 'token']))==0:
#             test.loc[i, 'space']=True
#     test = test[test['space']==False]

#     test['pred_token'] = pred['pred_token'].values
#     test['pred_tag'] = pred['pred_tag'].values
#     pred['original_index'] = test['original_index'].values
#     pred['original_index'] = pred['original_index'].str.lower()

#     pred_tags = {}
#     for name_i, group_i in test.groupby('original_index'):
#         pred_tag = []
#         if group_i['pred_tag'].iloc[0] in ['SAP','TECH','SOFT']:
#             pred_tag.append(group_i['pred_token'].iloc[0])
#         for idx in range(1, len(group_i)):
#             if group_i['pred_tag'].iloc[idx] in ['SAP','TECH','SOFT']:
#                 if group_i['pred_tag'].iloc[idx] != group_i['pred_tag'].iloc[idx-1]:
#                     pred_tag.append(group_i['pred_token'].iloc[idx])
#                 elif group_i['pred_tag'].iloc[idx] == group_i['pred_tag'].iloc[idx-1]:
#                     pred_tag[-1] = pred_tag[-1]+(' ')*(group_i['start_idx'].iloc[idx]-group_i['end_idx'].iloc[idx-1])+group_i['pred_token'].iloc[idx]
#         pred_tags[name_i] = set(pred_tag)

#     labels = pd.read_csv('/content/drive/MyDrive/skill scraper/total_ground_truth.csv', index_col=0)
#     # labels = labels[(labels['company']=='sap')&(labels['work_area'].isin(['Consulting','Customer Support']))]
#     # print('labels shape: ', labels.shape)

#     model_pred_list = []
#     match_label_list = []
#     manual_only_list = []
#     pred_only_list = []
#     precision_list = []
#     recall_list = []
#     f1_list = []

#     for idx in labels.index:
#         label = labels.loc[idx, 'Skills']
#         label = label.split(',')
#         label = [i.lower().strip() for i in label]
#         label = [i for i in label if len(i)>0]
#         pred = list(pred_tags[labels.loc[idx, 'original_index']])
#         pred = [i.lower().strip() for i in pred]
#         match_label,manual_only,pred_only,precision,recall,f1 = test_metric(label, pred, beta=2)
#         model_pred_list.append(pred)
#         match_label_list.append(match_label)
#         manual_only_list.append(manual_only)
#         pred_only_list.append(pred_only)
#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     labels['pred_label'] = model_pred_list
#     labels['match_label'] = match_label_list
#     labels['manual_only'] = manual_only_list
#     labels['pred_only'] = pred_only_list
#     labels['precision'] = precision_list
#     labels['recall'] = recall_list
#     labels['f1'] = f1_list

#     # print(labels[['pred_label','match_label']])

#     return labels


# In[29]:


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_tuple = tuple(map(torch.stack, zip(*batch)))
    batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
    max_len = batch_lens.max().item()
    results = ()
    for item in batch_tuple:
        if item.dim() >= 2:
            results += (item[:, :max_len],)
        else:
            results += (item,)
    return results


# In[30]:


def loss_backward(args, loss, optimizer):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    loss.backward()

class FGM():
    def __init__(self, model, param_name, alpha=1.0):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha

    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)

    def backup_param_data(self):
        self.data = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                assert name in self.data
                param.data = self.data[name]
        self.data = {}

    def adversarial_training(self, args, inputs, optimizer):
        self.backup_param_data()
        self.adversarial()
        loss = self.model(**inputs)[0]
        loss_backward(args, loss, optimizer)
        self.restore_param_data()


class PGD():
    def __init__(self, model, param_name, alpha=0.3, epsilon=1.0, K=3):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.K = K

    def backup_param_data(self):
        self.data = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                param.data = self.data[name]

    def backup_param_grad(self):
        self.grad = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.grad[name] = param.grad.clone()

    def restore_param_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                param.grad = self.grad[name]


    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        eta = torch.clamp(param_data - self.data[param_name])
        norm = torch.norm(eta)
        if norm > self.epsilon:
            eta = self.epsilon * eta / norm
        return self.data[param_name] + eta

    def adversarial_training(self, args, inputs, optimizer):
        self.backup_param_data()
        self.backup_param_grad()
        for k in range(self.K):
            self.adversarial()
            if k != self.K - 1:
                self.model.zero_grad()
            else:
                self.restore_param_grad()
            loss = self.model(**inputs)[0]
            loss_backward(args, loss, optimizer)
        self.restore_param_data()


class FreeAT():
    def __init__(self, model, param_name, alpha=0.3, epsilon=1.0, K=3):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.K = K


# In[31]:


class args:
    data_dir = None
    model_type = None
    model_name_or_path = None
    output_dir = None
    labels = None
    config_name = ""
    tokenizer_name = ""
    cache_dir = "./cache"
    max_seq_length = 128
    do_train = True
    do_eval = True
    do_predict = True
    evaluate_during_training = True
    do_lower_case = True
    per_gpu_train_batch_size = 8
    per_gpu_eval_batch_size = 8
    gradient_accumulation_steps = 1
    loss_type = 'ce'
    learning_rate = 5e-5
    bert_lr = 5e-5
    classifier_lr = 5e-5
    crf_lr = 1e-3
    adv_training = None
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 3.0
    max_steps = -1
    warmup_steps = 0
    logging_steps = '0.1'
    overwrite_output_dir = True
    no_cuda = False
    seed = 42
    overwrite_cache = False
    local_rank = -1

args.data_dir = './datasets'
args.model_type = 'bert'
#args.model_name_or_path = './mlm_model_uncased_tech_line_by_line'
model_name_or_path="/Users/i052090/Downloads/segmentation/data/bertmany/bert-base-uncased"
args.output_dir = './model_crf_uncased_tech'
args.do_lower_case = True
args.labels = ''
args.overwrite_output_dir = True
args.do_train = True
args.do_eval = True
args.do_predict = True
args.evaluate_during_training = True
args.num_train_epochs = 100
args.max_seq_length = 512
args.logging_steps = '0.5'
args.per_gpu_train_batch_size = 8
args.per_gpu_eval_batch_size = 16
args.learning_rate = 5e-5
args.bert_lr = 5e-5
args.classifier_lr = 5e-5
args.crf_lr = 1e-3


# In[32]:


def main():

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO 
    )
    logger.warning(
        "device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    print('labels: ', labels)
    print('pad_token_label_id: ', pad_token_label_id)

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
         args.config_name if args.config_name else args.model_name_or_path,
         num_labels=num_labels,
         id2label={str(i): label for i, label in enumerate(labels)},
         label2id={label: i for i, label in enumerate(labels)},
         cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    # #####
    setattr(config, 'loss_type', args.loss_type)
    setattr(config, 'output_hidden_states', True)
    #####
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    print("Tokenizer arguments: %s", tokenizer_args)
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
         cache_dir=args.cache_dir if args.cache_dir else None,
         **tokenizer_args,
     )
    model = AutoModelForCrfNer.from_pretrained(
         args.model_name_or_path,
         from_tf= bool(".ckpt" in args.model_name_or_path),
         config=config,
         cache_dir=args.cache_dir if args.cache_dir else None,
     )
    # tokenizer = BertTokenizer.from_pretrained(
    #     "/Users/i052090/Downloads/segmentation/data/bertmany/bert-base-uncased/bert-base-uncased-vocab.txt",
    #     do_lower_case=True)
    #
    # model = AutoModelForCrfNer.from_pretrained(
    #     "/Users/i052090/Downloads/segmentation/data/bertmany/bert-base-uncased",
    #     cache_dir="./cache")

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    print("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # # Evaluation
    # if args.do_eval:
    #     tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
    #     checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
    #     model = AutoModelForSoftmaxNer.from_pretrained(checkpoint)
    #     model.to(args.device)
    #     results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", prefix='test')
    #     # Save predictions
    #     output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    #     with open(output_test_predictions_file, "w") as writer:
    #         with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
    #             example_id = 0
    #             for line in f:
    #                 if line.startswith("-DOCSTART-") or line == "" or line == "\n":
    #                     writer.write(line)
    #                     if not predictions[example_id]:
    #                         example_id += 1
    #                 elif predictions[example_id]:
    #                     output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
    #                     writer.write(output_line)
    #                 else:
    #                     logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
                        
    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "a") as writer:
    #         writer.write('***** Predict in dev dataset *****')
    #         writer.write("{} = {}\n".format('loss', str(results['loss'])))
    #         writer.write("{} = {}\n".format('precision', str(results['precision'])))
    #         writer.write("{} = {}\n".format('recall', str(results['recall'])))
    #         writer.write("{} = {}\n".format('f1', str(results['f1'])))
    #         # writer.write("{} = {}\n".format('report', str(results['report'])))

    # if args.do_predict:
    #     tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
    #     checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
    #     model = AutoModelForSoftmaxNer.from_pretrained(checkpoint)
    #     model.to(args.device)
    #     results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", prefix='test')
    #     # # Save results
    #     # output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    #     # with open(output_test_results_file, "w") as writer:
    #     #     writer.write('***** Predict in dev dataset *****')
    #     #     writer.write("{} = {}\n".format('report', str(results['report'])))

        


# In[33]:


main()


# In[ ]:





# In[ ]:





# In[34]:


# import pandas as pd

# sap_jobs = pd.read_csv('/content/drive/MyDrive/skill scraper/sap_jobs_with_labels.csv', index_col=0)
# aws_jobs = pd.read_csv('/content/drive/MyDrive/skill scraper/aws_jobs_with_labels.csv', index_col=0)
# ms_jobs = pd.read_csv('/content/drive/MyDrive/skill scraper/ms_jobs_with_labels.csv', index_col=0)
# google_jobs = pd.read_csv('/content/drive/MyDrive/skill scraper/google_jobs_with_labels.csv', index_col=0)

# sap_jobs['original_index'] =  sap_jobs['company_name']+'_'+sap_jobs.index.astype(str)
# aws_jobs['original_index'] = aws_jobs['company_name']+'_'+aws_jobs.index.astype(str)
# ms_jobs['original_index'] = ms_jobs['company_name']+'_'+ms_jobs.index.astype(str)
# google_jobs['original_index'] = google_jobs['company_name']+'_'+google_jobs.index.astype(str)

# jobs = pd.concat([sap_jobs, aws_jobs, ms_jobs, google_jobs], ignore_index=True)

# sap_labels = pd.read_csv('/content/drive/MyDrive/skill scraper/sap_job_ground_truth.csv')
# sap_labels.dropna(subset=['Requisition ID'], inplace=True)
# sap_labels['Requisition ID'] = sap_labels['Requisition ID'].astype('int')
# sap_labels.columns = [i.strip() for i in sap_labels.columns]
# sap_labels['company'] = 'sap'

# aws_labels = pd.read_csv('/content/drive/MyDrive/skill scraper/aws_job_ground_truth.csv')
# aws_labels.dropna(subset=['Requisition ID'], inplace=True)
# aws_labels['Requisition ID'] = aws_labels['Requisition ID'].astype('int')
# aws_labels.columns = [i.strip() for i in aws_labels.columns]
# aws_labels['company'] = 'aws'

# ms_labels = pd.read_csv('/content/drive/MyDrive/skill scraper/ms_job_ground_truth.csv')
# ms_labels.dropna(subset=['Requisition ID'], inplace=True)
# ms_labels['Requisition ID'] = ms_labels['Requisition ID'].astype('int')
# ms_labels.columns = [i.strip() for i in ms_labels.columns]
# ms_labels['company'] = 'microsoft'

# google_labels = pd.read_csv('/content/drive/MyDrive/skill scraper/google_job_ground_truth.csv')
# google_labels.dropna(subset=['Requisition ID'], inplace=True)
# google_labels['Requisition ID'] = google_labels['Requisition ID'].astype('int')
# google_labels.columns = [i.strip() for i in google_labels.columns]
# google_labels['company'] = 'google'
# google_labels = google_labels.merge(google_jobs[['work_area']], how='left', left_on='Requisition ID', right_index=True)

# sap_jobs['original_index'] = sap_jobs.index
# sap_labels = sap_labels.merge(sap_jobs[['original_index','reference']], left_on = 'Requisition ID', right_on = 'reference')
# sap_labels['original_index'] = sap_labels['company']+'_'+sap_labels['original_index'].astype(str)

# aws_labels['original_index'] = aws_labels['company']+'_'+aws_labels['Requisition ID'].astype(str)
# ms_labels['original_index'] = ms_labels['company']+'_'+ms_labels['Requisition ID'].astype(str)
# google_labels['original_index'] = google_labels['company']+'_'+google_labels['Requisition ID'].astype(str)

# labels = pd.concat([sap_labels, aws_labels, ms_labels, google_labels], ignore_index=True)

# labels.reset_index(inplace=True, drop=True)

# labels = labels.merge(jobs[['original_index', 'description_raw']], how='left', left_on='original_index', right_on = 'original_index')

# labels['original_index'] = labels['original_index'].str.lower()
# jobs['original_index'] = jobs['original_index'].str.lower()

# labels.to_csv('/content/drive/MyDrive/skill scraper/total_ground_truth.csv')


# In[35]:


# labels_result = pd.read_csv(os.path.join(args.output_dir, "best_checkpoint/labels_comparison.csv"), index_col=0)

# labels_result['precision'].mean(), labels_result['recall'].mean(), labels_result['f1'].mean()


# In[36]:


# labels_result.groupby('company')[['precision','recall','f1']].mean()


# In[37]:


# labels_result.groupby(['company', 'work_area'])[['precision','recall','f1']].mean()


# In[38]:


# data = pd.read_csv(os.path.join(args.output_dir, "best_checkpoint/labels_comparison.csv"), index_col=0)

# data['precision'].mean(), data['recall'].mean(), data['f1'].mean()


# In[39]:


# data.groupby('company')[['precision','recall','f1']].mean()


# In[40]:


# data.groupby(['company', 'work_area'])[['precision','recall','f1']].mean()


# In[ ]:




