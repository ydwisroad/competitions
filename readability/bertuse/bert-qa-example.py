#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import codecs
import os, re, json


# In[2]:


# https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip


# In[3]:


train = json.load(open('../../../data/nlpdata/cmrc2018_public/train.json'))
dev = json.load(open('../../../data/nlpdata/cmrc2018_public/dev.json'))


# In[4]:


train['data'][0]['paragraphs'][0]


# In[5]:


from transformers import BertTokenizerFast, BertForQuestionAnswering

tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm-ext')
model = BertForQuestionAnswering.from_pretrained('hfl/chinese-bert-wwm-ext')


# In[6]:


paragraphs = []
questions = []
start_positions = []
end_positions = []
for paragraph in train['data']:
    for qa in paragraph['paragraphs'][0]['qas']:
        paragraphs.append(paragraph['paragraphs'][0]['context'])
        questions.append(qa['question'])
        start_positions.append(qa['answers'][0]['answer_start'])
        end_positions.append(qa['answers'][0]['answer_start'] + len(qa['answers'][0]['text']))


# In[7]:


train_encodings = tokenizer(paragraphs, questions, 
                            return_tensors='pt', truncation=True, padding=True,
                           max_length=512)

train_encodings['start_positions'] = [train_encodings.char_to_token(idx, x) if train_encodings.char_to_token(idx, x) != None else -1
                                      for idx, x in enumerate(start_positions)]
train_encodings['end_positions'] = [train_encodings.char_to_token(idx, x-1) if train_encodings.char_to_token(idx, x-1) != None else -1
                                    for idx, x in enumerate(end_positions)]


# In[8]:


paragraphs = []
questions = []
start_positions = []
end_positions = []
for paragraph in dev['data']:
    for qa in paragraph['paragraphs'][0]['qas']:
        paragraphs.append(paragraph['paragraphs'][0]['context'])
        questions.append(qa['question'])
        start_positions.append(qa['answers'][0]['answer_start'])
        end_positions.append(qa['answers'][0]['answer_start'] + len(qa['answers'][0]['text']))


# In[9]:


val_encodings = tokenizer(paragraphs, questions, 
                            return_tensors='pt', truncation=True, padding=True,
                           max_length=512)

val_encodings['start_positions'] = [val_encodings.char_to_token(idx, x) if val_encodings.char_to_token(idx, x) != None else -1
                                      for idx, x in enumerate(start_positions) ]
val_encodings['end_positions'] = [val_encodings.char_to_token(idx, x-1) if val_encodings.char_to_token(idx, x) != None else -1
                                    for idx, x in enumerate(end_positions)]


# In[10]:


import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
    
import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)


# In[11]:


from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for idx, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        acc1 = ( (start_pred == start_positions).sum() / len(start_pred) ).item()
        acc2 = ( (end_pred == end_positions).sum() / len(start_pred) ).item()

        if idx % 10 == 0:
            print(loss.item(), acc1, acc2)
            with codecs.open('log.log', 'a') as up:
                up.write('{3}\t{0}\t{1}\t{2}\n'.format(loss.item(), acc1, acc2, 
                                                       str(epoch) + '/' + str(idx) +'/'+ str(len(train_loader))))

model.eval()


# In[ ]:


def predcit(doc, query):
    print(doc)
    print('提问：', query)
    item = tokenizer([doc, query], max_length=512, return_tensors='pt', truncation=True, padding=True,)
    with torch.no_grad():
        input_ids = torch.tensor(item['input_ids']).to(device).reshape(1, -1)
        attention_mask = torch.tensor(item['attention_mask']).to(device).reshape(1, -1)
        
        outputs = model(input_ids[:, :512], attention_mask[:, :512])
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
    
    try:
        start_pred = item.token_to_chars(0, start_pred)
        end_pred = item.token_to_chars(0, end_pred)
    except:
        return ''
    
    if start_pred.start > end_pred.end:
        return ''
    else:
        return doc[start_pred.start:end_pred.end]


# In[81]:


dev['data'][0]


# In[82]:


predcit(dev['data'][0]['paragraphs'][0]['context'],
       dev['data'][0]['paragraphs'][0]['qas'][0]['question'])


# In[83]:


predcit(dev['data'][0]['paragraphs'][0]['context'],
       dev['data'][0]['paragraphs'][0]['qas'][1]['question'])

