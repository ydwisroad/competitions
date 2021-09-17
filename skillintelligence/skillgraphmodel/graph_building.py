#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch


# In[2]:


skill_posting_relation = pd.read_csv('./data/skill_posting_relation_normalized.csv')
skill_normalized_df = pd.read_csv('./data/skill_normalized_df.csv')


# In[3]:


skill_posting_relation.skill_normalized.nunique()


# In[4]:


skill_posting_relation.posting_ID.nunique()


# In[5]:


posting = pd.read_csv('./data/posting.csv')
posting = posting[posting.posting_ID.isin(skill_posting_relation.posting_ID)]


# In[6]:


posting.info()


# In[7]:


"""Graph builder from pandas dataframes"""
from collections import namedtuple
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import dgl

__all__ = ['PandasGraphBuilder']

def _series_to_tensor(series):
    if is_categorical(series):
        return torch.LongTensor(series.cat.codes.values.astype('int64'))
    else:       # numeric
        return torch.FloatTensor(series.values)

class PandasGraphBuilder(object):
    def __init__(self):
        self.entity_tables = {}
        self.relation_tables = {}

        self.entity_pk_to_name = {}     # mapping from primary key name to entity name
        self.entity_pk = {}             # mapping from entity name to primary key
        self.entity_key_map = {}        # mapping from entity names to primary key values
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}
        self.relation_name_to_etype = {}
        self.relation_src_key = {}      # mapping from relation name to source key
        self.relation_dst_key = {}      # mapping from relation name to destination key

    def add_entities(self, entity_table, primary_key, name):
        entities = entity_table[primary_key].astype('category')
        if not (entities.value_counts() == 1).all():
            raise ValueError('Different entity with the same primary key detected.')
        # preserve the category order in the original entity table
        entities = entities.cat.reorder_categories(entity_table[primary_key].values)

        self.entity_pk_to_name[primary_key] = name
        self.entity_pk[name] = primary_key
        self.num_nodes_per_type[name] = entity_table.shape[0]
        self.entity_key_map[name] = entities
        self.entity_tables[name] = entity_table

    def add_binary_relations(self, relation_table, source_key, destination_key, name):
        src = relation_table[source_key].astype('category')
        src = src.cat.set_categories(
            self.entity_key_map[self.entity_pk_to_name[source_key]].cat.categories)
        dst = relation_table[destination_key].astype('category')
        dst = dst.cat.set_categories(
            self.entity_key_map[self.entity_pk_to_name[destination_key]].cat.categories)
        if src.isnull().any():
            raise ValueError(
                'Some source entities in relation %s do not exist in entity %s.' %
                (name, source_key))
        if dst.isnull().any():
            raise ValueError(
                'Some destination entities in relation %s do not exist in entity %s.' %
                (name, destination_key))

        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.relation_name_to_etype[name] = etype
        self.edges_per_relation[etype] = (src.cat.codes.values.astype('int64'), dst.cat.codes.values.astype('int64'))
        self.relation_tables[name] = relation_table
        self.relation_src_key[name] = source_key
        self.relation_dst_key[name] = destination_key

    def build(self):
        # Create heterograph
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        return graph


# In[8]:


# Build graph
graph_builder = PandasGraphBuilder()
graph_builder.add_entities(posting, 'posting_ID', 'posting')
graph_builder.add_entities(skill_normalized_df, 'skill_ID_normalized', 'skill')
graph_builder.add_binary_relations(skill_posting_relation, 'posting_ID', 'skill_ID_normalized', 'contains')
graph_builder.add_binary_relations(skill_posting_relation, 'skill_ID_normalized', 'posting_ID', 'contained-by')

g = graph_builder.build()


# In[9]:


# graph_builder.add_entities(skill_normalized_df, 'skill_ID_normalized', 'skill')
entities = skill_normalized_df['skill_ID_normalized'].astype('category')
print(entities)
entities = entities.cat.reorder_categories(skill_normalized_df['skill_ID_normalized'].values)
entities


# In[10]:


# graph_builder.add_binary_relations(skill_posting_relation, 'posting_ID', 'skill_ID_normalized', 'contains')
relation_table = skill_posting_relation
source_key = 'skill_ID_normalized'
destination_key = 'posting_ID'

src = relation_table[source_key].astype('category')
print(src)
src = src.cat.set_categories(entities.cat.categories)
print(src)
print(src.cat.codes.values.astype('int64'))

graph_id_dict = dict(zip(src.values, src.cat.codes))


# In[11]:


graph_builder.edges_per_relation


# In[12]:


g


# In[13]:


import tqdm
import dask.dataframe as dd

def train_test_split_random(df, user):
    df['train_mask'] = np.ones((len(df),), dtype=np.bool)
    df['val_mask'] = np.zeros((len(df),), dtype=np.bool)
    df['test_mask'] = np.zeros((len(df),), dtype=np.bool)
    df = dd.from_pandas(df, npartitions=10)
    def train_test_split(df):
        df_shuffled=df.sample(frac=1, random_state=1, axis=0)
        if df_shuffled.shape[0] > 1:
            df_shuffled.iloc[-1, -3] = False
            df_shuffled.iloc[-1, -1] = True
        if df_shuffled.shape[0] > 2:
            df_shuffled.iloc[-2, -3] = False
            df_shuffled.iloc[-2, -2] = True
        return df_shuffled
    #df = df.groupby(user, group_keys=False).apply(train_test_split).compute(scheduler='processes').sort_index()
    df = df.groupby(user, group_keys=False).apply(train_test_split).compute().sort_index()
    print(df[df[user] == df[user].unique()[0]])
    print(df[df[user] == df[user].unique()[-1]])
    return df['train_mask'].to_numpy().nonzero()[0],            df['val_mask'].to_numpy().nonzero()[0],            df['test_mask'].to_numpy().nonzero()[0]

def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        preserve_nodes=True)
    # remove the induced node IDs - should be assigned by model instead
    del train_g.nodes[utype].data[dgl.NID]
    del train_g.nodes[itype].data[dgl.NID]

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g

def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix

def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) /         (values.max(0, keepdims=True) - values.min(0, keepdims=True))


# In[14]:


#get_ipython().run_cell_magic('time', '', "\n# Train-validation-test split\ntrain_indices, val_indices, test_indices = train_test_split_random(skill_posting_relation, 'posting_ID')")
# Train-validation-test split
train_indices, val_indices, test_indices = train_test_split_random(skill_posting_relation, 'posting_ID')

# In[15]:


print(train_indices)
print(val_indices)
print(test_indices)


# In[16]:


# Build the graph with training interactions only
train_g = build_train_graph(g, train_indices, 'posting', 'skill', 'contains', 'contained-by')
assert train_g.out_degrees(etype='contains').min() > 0


# In[17]:


torch.bincount(train_g.out_degrees(etype='contains'))


# In[18]:


train_g


# In[19]:


# Build the user-item sparse matrix for validation and test set.
val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'posting', 'skill', 'contains')


# In[20]:


## Dump the graph and the datasets

dataset = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': None,
    'item-images': None,
    'user-type': 'posting',
    'item-type': 'skill',
    'user-to-item-type': 'contains',
    'item-to-user-type': 'contained-by'}


# In[21]:


with open('save_posting.p', 'wb') as f:
    pickle.dump(dataset, f)


# In[22]:


dataset_whole = {
    'train-graph': g,    #g
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': None,
    'item-images': None,
    'user-type': 'posting',
    'item-type': 'skill',
    'user-to-item-type': 'contains',
    'item-to-user-type': 'contained-by'}


# In[23]:


with open('save_posting_whole.p', 'wb') as f:
    pickle.dump(dataset_whole, f)


# In[24]:


torch.bincount(g.out_degrees(etype='contains'))


# ### Supervised setting

# In[25]:


skill_posting_relation = pd.read_csv('./data/skill_posting_relation_normalized.csv')
skill_normalized_df = pd.read_csv('./data/skill_normalized_df.csv')


# In[26]:


print(skill_posting_relation.skill_normalized.nunique())
print(skill_posting_relation.posting_ID.nunique())


# In[27]:


posting = pd.read_csv('./data/posting.csv')
posting = posting[posting.posting_ID.isin(skill_posting_relation.posting_ID)]


# In[28]:


posting.info()


# In[29]:


# graph_builder.add_entities(skill_normalized_df, 'skill_ID_normalized', 'skill')
entities = skill_normalized_df['skill_ID_normalized'].astype('category')
print(entities)
entities = entities.cat.reorder_categories(skill_normalized_df['skill_ID_normalized'].values)
entities


# In[30]:


# graph_builder.add_binary_relations(skill_posting_relation, 'posting_ID', 'skill_ID_normalized', 'contains')
source_key = 'skill_ID_normalized'
destination_key = 'posting_ID'

src = skill_posting_relation[source_key].astype('category')
print(src)
src = src.cat.set_categories(entities.cat.categories)
print(src)
print(src.cat.codes.values.astype('int64'))

graph_id_dict = dict(zip(src.values, src.cat.codes))


# In[31]:


skill_normalized_df['graph_ID'] = skill_normalized_df.skill_ID_normalized
skill_normalized_df['graph_ID'] = skill_normalized_df['graph_ID'].map(graph_id_dict)
skill_normalized_df

pos = []

for index, row in skill_normalized_df.iterrows():
    pos_temp = []
    for g in ['CNCF', 'roadmap', 'onet']:
        if not pd.isnull(row[g]) and row[g] != 'devops':
            group = skill_normalized_df[skill_normalized_df[g] == row[g]]
            group_id = list(group.graph_ID.values)
            group_id.remove(row.graph_ID)
            pos_temp += group_id
        if len(pos_temp) == 0 and row[g] == 'devops':
            group = skill_normalized_df[skill_normalized_df[g] == row[g]]
            group_id = list(group.graph_ID.values)
            group_id.remove(row.graph_ID)
            pos_temp += group_id
    if len(pos_temp) == 0:
        group = skill_normalized_df[skill_normalized_df['onet'] == 'devops']
        group_id = list(group.graph_ID.values)
        pos_temp += group_id
    pos.append(pos_temp) 


# In[32]:


skill_normalized_df['related_skill_ID'] = pos
skill_normalized_df.head()


# In[33]:


skill_normalized_df.related_skill_ID.str.len()


# In[34]:


skill_normalized_df[skill_normalized_df.related_skill_ID.str.len() == 1]


# In[35]:


related_skill_dict = pd.Series(skill_normalized_df.related_skill_ID.values,index=skill_normalized_df.graph_ID.values).to_dict()


# In[36]:


dataset_whole = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'related-skill': related_skill_dict,
    'item-texts': None,
    'item-images': None,
    'user-type': 'posting',
    'item-type': 'skill',
    'user-to-item-type': 'contains',
    'item-to-user-type': 'contained-by'}


# In[37]:
print("related_skill_dict ", related_skill_dict)

with open('save_posting_whole_sup.p', 'wb') as f:
    pickle.dump(dataset_whole, f)

