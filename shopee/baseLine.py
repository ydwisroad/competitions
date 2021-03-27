#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DATA_PATH = '../input/'
DATA_PATH = '../../data/shopee/simple/'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'

# In[2]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2, matplotlib.pyplot as plt
from tqdm import tqdm_notebook

import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors


def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


# In[3]:


COMPUTE_CV = True

test = pd.read_csv(DATA_PATH + TEST_CSV)
if len(test) > 3:
    COMPUTE_CV = False
else:
    print('this submission notebook will compute CV score, but commit notebook will not')

# COMPUTE_CV = False

if COMPUTE_CV:
    train = pd.read_csv(DATA_PATH + TRAIN_CSV)
    train['image'] = DATA_PATH + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    train_gf = cudf.read_csv(DATA_PATH + TRAIN_CSV)
else:
    train = pd.read_csv(DATA_PATH + TEST_CSV)
    train['image'] = DATA_PATH + 'test_images/' + train['image']
    train_gf = cudf.read_csv(DATA_PATH + TEST_CSV)

print('train shape is', train.shape)
train.head()

# # image hash

# In[4]:


tmp = train.groupby('image_phash').posting_id.agg('unique').to_dict()
train['oof_hash'] = train.image_phash.map(tmp)

# In[5]:


if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_hash'), axis=1)
    print('CV score for baseline =', train.f1.mean())

# # image CNN

# In[6]:


from PIL import Image

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset


class ShopeeImageDataset(Dataset):
    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path)


# In[7]:


imagedataset = ShopeeImageDataset(
    train['image'].values,
    transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

imageloader = torch.utils.data.DataLoader(
    imagedataset,
    batch_size=10, shuffle=False, num_workers=2
)


# In[8]:


class ShopeeImageEmbeddingNet(nn.Module):
    def __init__(self):
        super(ShopeeImageEmbeddingNet, self).__init__()

        model = models.resnet18(True)
        model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out


# In[9]:


get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints/')
get_ipython().system('cp ../input/pretrained-pytorch-models/resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/')

# In[10]:


DEVICE = 'cuda'

imgmodel = ShopeeImageEmbeddingNet()
imgmodel = imgmodel.to(DEVICE)

imagefeat = []
with torch.no_grad():
    for data in tqdm_notebook(imageloader):
        data = data.to(DEVICE)
        feat = imgmodel(data)
        feat = feat.reshape(feat.shape[0], feat.shape[1])
        feat = feat.data.cpu().numpy()

        imagefeat.append(feat)

# In[11]:


from sklearn.preprocessing import normalize

# l2 norm to kill all the sim in 0-1
imagefeat = np.vstack(imagefeat)
imagefeat = normalize(imagefeat)

# In[12]:


KNN = 50
if len(test) == 3: KNN = 2
model = NearestNeighbors(n_neighbors=KNN)
model.fit(imagefeat)

# In[13]:


preds = []
CHUNK = 1024 * 4

imagefeat = cupy.array(imagefeat)

print('Finding similar images...')
CTS = len(imagefeat) // CHUNK
if len(imagefeat) % CHUNK != 0: CTS += 1
for j in range(CTS):

    a = j * CHUNK
    b = (j + 1) * CHUNK
    b = min(b, len(imagefeat))
    print('chunk', a, 'to', b)

    distances = cupy.matmul(imagefeat, imagefeat[a:b].T).T
    # distances = np.dot(imagefeat[a:b,], imagefeat.T)

    for k in range(b - a):
        IDX = cupy.where(distances[k,] > 0.95)[0]
        # IDX = np.where(distances[k,]>0.95)[0][:]
        o = train.iloc[cupy.asnumpy(IDX)].posting_id.values
        preds.append(o)

# del imagefeat, imgmodel


# In[14]:


train['oof_cnn'] = preds

if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_cnn'), axis=1)
    print('CV score for baseline =', train.f1.mean())

# # title TFIDF

# In[15]:


# from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer(stop_words=None, binary=True, max_features=25000)
text_embeddings = model.fit_transform(train_gf.title).toarray()
print('text embeddings shape', text_embeddings.shape)

# In[16]:


preds = []
CHUNK = 1024 * 4

print('Finding similar titles...')
CTS = len(train) // CHUNK
if len(train) % CHUNK != 0: CTS += 1
for j in range(CTS):

    a = j * CHUNK
    b = (j + 1) * CHUNK
    b = min(b, len(train))
    print('chunk', a, 'to', b)

    # COSINE SIMILARITY DISTANCE
    # cts = np.dot( text_embeddings, text_embeddings[a:b].T).T
    cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T

    for k in range(b - a):
        # IDX = np.where(cts[k,]>0.7)[0]
        IDX = cupy.where(cts[k,] > 0.7)[0]
        o = train.iloc[cupy.asnumpy(IDX)].posting_id.values
        preds.append(o)

del model, text_embeddings

# In[17]:


train['oof_text'] = preds

if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_text'), axis=1)
    print('CV score for baseline =', train.f1.mean())


# In[18]:


def combine_for_sub(row):
    x = np.concatenate([row.oof_text, row.oof_cnn, row.oof_hash])
    return ' '.join(np.unique(x))


def combine_for_cv(row):
    x = np.concatenate([row.oof_text, row.oof_cnn, row.oof_hash])
    return np.unique(x)


# In[19]:


if COMPUTE_CV:
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    train['oof'] = train.apply(combine_for_cv, axis=1)
    train['f1'] = train.apply(getMetric('oof'), axis=1)
    print('CV Score =', train.f1.mean())

train['matches'] = train.apply(combine_for_sub, axis=1)

# In[20]:


train[['posting_id', 'matches']].to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub.head()

# In[ ]: