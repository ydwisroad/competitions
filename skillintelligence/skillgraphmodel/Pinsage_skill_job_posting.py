#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import files
# uploaded = files.upload()


# In[ ]:


# from google.colab import drive
# drive.mount("/content/drive")


# In[ ]:


# !pip install dgl-cu100     # For CUDA 10.0 Build
import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader

def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size, related_skill_dict):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size
        self.related_skill_dict = related_skill_dict

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            
            tails = []
            neg_tails = []

            for h in heads.numpy():
                related_S = self.related_skill_dict[h]
                i = int(np.random.randint(0, len(related_S), size = 1))
                t = related_S[i]
                n_i = int(np.random.randint(0, self.g.number_of_nodes(self.item_type), size = 1))
                while n_i in related_S:
                    n_i = int(np.random.randint(0, self.g.number_of_nodes(self.item_type), size = 1))
                tails.append(t)
                neg_tails.append(n_i)
            tails = torch.as_tensor(tails)
            neg_tails = torch.as_tensor(neg_tails)

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    #print(old_frontier)
                    #print(frontier)
                    #print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]

# def assign_textual_node_features(ndata, textset, ntype):
#     """
#     Assigns numericalized tokens from a torchtext dataset to given block.
#     The numericalized tokens would be stored in the block as node features
#     with the same name as ``field_name``.
#     The length would be stored as another node feature with name
#     ``field_name + '__len'``.
#     block : DGLHeteroGraph
#         First element of the compacted blocks, with "dgl.NID" as the
#         corresponding node ID in the original graph, hence the index to the
#         text dataset.
#         The numericalized tokens (and lengths if available) would be stored
#         onto the blocks as new node features.
#     textset : torchtext.data.Dataset
#         A torchtext dataset whose number of examples is the same as that
#         of nodes in the original graph.
#     """
#     node_ids = ndata[dgl.NID].numpy()

#     for field_name, field in textset.fields.items():
#         examples = [getattr(textset[i], field_name) for i in node_ids]

#         tokens, lengths = field.process(examples)

#         if not field.batch_first:
#             tokens = tokens.t()

#         ndata[field_name] = tokens
#         ndata[field_name + '__len'] = lengths

def assign_features_to_blocks(blocks, g, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    # assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    # assign_textual_node_features(blocks[-1].dstdata, textset, ntype)

class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return blocks


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn

def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def _init_input_modules(g, ntype, hidden_dims):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(
                data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m
  
    # if textset is not None:
    #     for column, field in textset.fields.items():
    #         if field.vocab.vectors:
    #             module_dict[column] = BagOfWordsPretrained(field, hidden_dims)
    #         else:
    #             module_dict[column] = BagOfWords(field, hidden_dims)

    return module_dict

class BagOfWordsPretrained(nn.Module):
     def __init__(self, field, hidden_dims):
         super().__init__()

         input_dims = field.vocab.vectors.shape[1]
         self.emb = nn.Embedding(
             len(field.vocab.itos), input_dims,
             padding_idx=field.vocab.stoi[field.pad_token])
         self.emb.weight[:] = field.vocab.vectors
         self.proj = nn.Linear(input_dims, hidden_dims)
         nn.init.xavier_uniform_(self.proj.weight)
         nn.init.constant_(self.proj.bias, 0)

         disable_grad(self.emb)

     def forward(self, x, length):
         """
         x: (batch_size, max_length) LongTensor
         length: (batch_size,) LongTensor
         """
         x = self.emb(x).sum(1) / length.unsqueeze(1).float()
         return self.proj(x)

class BagOfWords(nn.Module):
     def __init__(self, field, hidden_dims):
         super().__init__()

         self.emb = nn.Embedding(
             len(field.vocab.itos), hidden_dims,
             padding_idx=field.vocab.stoi[field.pad_token])
         nn.init.xavier_uniform_(self.emb.weight)

     def forward(self, x, length):
         return self.emb(x).sum(1) / length.unsqueeze(1).float()

class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """
    def __init__(self, full_graph, ntype, hidden_dims):
        super().__init__()

        self.ntype = ntype
        self.inputs = _init_input_modules(full_graph, ntype, hidden_dims)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith('__len'):
              # This is an additional feature indicating the length of the ``feature``
              # column; we shouldn't process this.
                continue

            module = self.inputs[feature]
            if isinstance(module, (BagOfWords, BagOfWordsPretrained)):
              # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + '__len']
                result = module(data, length)
            else:
                result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)

class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z

class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h

class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()

        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score


# In[ ]:


import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
from tqdm.auto import tqdm

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims, n_layers):
        super().__init__()

        self.proj = LinearProjector(full_graph, ntype, hidden_dims)
        self.sage = SAGENet(hidden_dims, n_layers)
        self.scorer = ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def load_model(g, item_ntype, device, args):
    # Model
    model = PinSAGEModel(g, item_ntype, args['hidden_dims'], args['num_layers']).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args['lr'])
    if args['retrain']:
        checkpoint = torch.load(args['save_path'] + '.pth', map_location=device)
    else:
        checkpoint = torch.load(args['save_path'], map_location=device)
   
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, opt, checkpoint['epoch']


def train(dataset, args):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    related_skill = dataset['related-skill']
    
    device = torch.device(args['device'])

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))

    # # Prepare torchtext dataset and vocabulary
    # fields = {}
    # examples = []
    # for key, texts in item_texts.items():
    #     fields[key] = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
    # for i in range(g.number_of_nodes(item_ntype)):
    #     example = torchtext.data.Example.fromlist(
    #         [item_texts[key][i] for key in item_texts.keys()],
    #         [(key, fields[key]) for key in item_texts.keys()])
    #     examples.append(example)
    # textset = torchtext.data.Dataset(examples, fields)
    # for key, field in fields.items():
    #     field.build_vocab(getattr(textset, key))
    #     #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')

    # Sampler
    batch_sampler = ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args['batch_size'], related_skill)
    neighbor_sampler = NeighborSampler(
        g, user_ntype, item_ntype, args['random_walk_length'],
        args['random_walk_restart_prob'], args['num_random_walks'], args['num_neighbors'],
        args['num_layers'])
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args['num_workers'])
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args['batch_size'],
        collate_fn=collator.collate_test,
        num_workers=args['num_workers'])
    dataloader_it = iter(dataloader)

    # Model
    if args['retrain']:
        print('Loading pretrained model...')
        model, opt, start_epoch = load_model(g, item_ntype, device, args)
        print('start epoch:', start_epoch)
    else:
        model = PinSAGEModel(g, item_ntype, args['hidden_dims'], args['num_layers']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args['lr'])
        start_epoch = 0


    # For each batch of head-tail-negative triplets...
    for epoch_id in tqdm(range(start_epoch, args['num_epochs'] + start_epoch)):
        model.train()
        for batch_id in tqdm(range(args['batches_per_epoch'])):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        # model.eval()
        # with torch.no_grad():
        #     item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args['batch_size'])
        #     h_item_batches = []
        #     for blocks in dataloader_test:
        #         for i in range(len(blocks)):
        #             blocks[i] = blocks[i].to(device)

        #         h_item_batches.append(model.get_repr(blocks))
        #     h_item = torch.cat(h_item_batches, 0)

        #     print(evaluation.evaluate_nn(dataset, h_item, args['k'], args['batch_size']))
        
        if args['eval_epochs'] and not (epoch_id+1) % args['eval_epochs']:
            model.eval()
            with torch.no_grad():
                item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args['batch_size'])
                h_item_batches = []
                for blocks in dataloader_test:
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)
                    h_item_batches.append(model.get_repr(blocks))
                h_item = torch.cat(h_item_batches, 0)
                torch.save(h_item,'embeddings_supervised' + str((epoch_id+1)) + 'epoch.pth')
#                 files.download('embeddings_' + str((epoch_id+1)) + 'epoch.pth')

        if args['save_epochs']:
            if not (epoch_id+1) % args['save_epochs']:
                torch.save({
                'epoch': (epoch_id+1),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
                        }, args['save_path'] + '_' + str((epoch_id+1)) + 'epoch.pth')
#                 files.download(args['save_path'] + '_' + str((epoch_id+1)) + 'epoch.pth')


# In[ ]:


# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[ ]:


args = {'random_walk_length':2,
        'random_walk_restart_prob':0.5,
        'num_random_walks':10,
        'num_neighbors':3,
        'num_layers':2,
        'hidden_dims':64,
        'batch_size':64,
        #'device':'cuda:0',
        'device':'cpu',
        'num_epochs':300,
        'batches_per_epoch':10000, #20000
        'num_workers':2,
        'lr':3e-5,
        'k':10,
        'save_epochs':5,
        'eval_epochs':5,
        'retrain':0,
        'save_path':'model',
        }
# Load dataset
with open('save_posting_whole_sup.p', 'rb') as f:
    dataset = pickle.load(f)
train(dataset, args)



# In[ ]:




