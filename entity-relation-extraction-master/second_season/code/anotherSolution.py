import os
import re
import math
import zipfile
import numpy as np
from collections import Counter, defaultdict
from itertools import permutations, chain
from gensim.models import Word2Vec

train_data_dir = '../rawdata/train1'
test_a_data_dir = '../rawdata/test1'
test_b_data_dir = '../rawdata/test1'

ENTITIES = [
    "Amount", "Anatomy", "Disease", "Drug",
    "Duration", "Frequency", "Level", "Method",
    "Operation", "Reason", "SideEff", "Symptom",
    "Test", "Test_Value", "Treatment"
]

RELATIONS = [
    "Test_Disease", "Symptom_Disease", "Treatment_Disease",
    "Drug_Disease", "Anatomy_Disease", "Frequency_Drug",
    "Duration_Drug", "Amount_Drug", "Method_Drug", "SideEff-Drug"
]


class Entity(object):
    def __init__(self, ent_id, category, start_pos, end_pos, text):
        self.ent_id = ent_id
        self.category = category
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.text = text

    def __gt__(self, other):
        return self.start_pos > other.start_pos

    def offset(self, offset_val):
        return Entity(self.ent_id,
                      self.category,
                      self.start_pos + offset_val,
                      self.end_pos + offset_val,
                      self.text)

    def __repr__(self):
        fmt = '({ent_id}, {category}, ({start_pos}, {end_pos}), {text})'
        return fmt.format(**self.__dict__)


class Entities(object):
    def __init__(self, ents):
        self.ents = sorted(ents)
        self.ent_dict = dict(zip([ent.ent_id for ent in ents], ents))

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.ents[key]
        else:
            return self.ent_dict.get(key, None)

    def __len__(self):
        return len(self.ents)

    def offset(self, offset_val):
        ents = [ent.offset(offset_val) for ent in self.ents]
        return Entities(ents)

    def vectorize(self, vec_len, cate2idx):
        res_vec = np.zeros(vec_len, dtype=int)
        for ent in self.ents:
            res_vec[ent.start_pos: ent.end_pos] = cate2idx[ent.category]
        return res_vec

    def find_entities(self, start_pos, end_pos):
        res = []
        for ent in self.ents:
            if ent.start_pos > end_pos:
                break
            sp, ep = (max(start_pos, ent.start_pos), min(end_pos, ent.end_pos))
            if ep > sp:
                new_ent = Entity(ent.ent_id, ent.category, sp, ep, ent.text[:(ep - sp)])
                res.append(new_ent)
        return Entities(res)

    def __add__(self, other):
        ents = self.ents + other.ents
        return Entities(ents)

    def merge(self):
        merged_ents = []
        for ent in self.ents:
            if len(merged_ents) == 0:
                merged_ents.append(ent)
            elif (merged_ents[-1].end_pos == ent.start_pos and
                  merged_ents[-1].category == ent.category):
                merged_ent = Entity(ent_id=merged_ents[-1].ent_id,
                                    category=ent.category,
                                    start_pos=merged_ents[-1].start_pos,
                                    end_pos=ent.end_pos,
                                    text=merged_ents[-1].text + ent.text)
                merged_ents[-1] = merged_ent
            else:
                merged_ents.append(ent)
        return Entities(merged_ents)


class Relation(object):
    def __init__(self, rel_id, category, ent1, ent2):
        self.rel_id = rel_id
        self.category = category
        self.ent1 = ent1
        self.ent2 = ent2

    @property
    def is_valid(self):
        return (isinstance(self.ent1, Entity) and
                isinstance(self.ent2, Entity) and
                [self.ent1.category, self.ent2.category] == re.split('[-_]', self.category))

    @property
    def start_pos(self):
        return min(self.ent1.start_pos, self.ent2.start_pos)

    @property
    def end_pos(self):
        return max(self.ent1.end_pos, self.ent2.end_pos)

    def offset(self, offset_val):
        return Relation(self.rel_id,
                        self.category,
                        self.ent1.offset(offset_val),
                        self.ent2.offset(offset_val))

    def __gt__(self, other_rel):
        return self.ent1.start_pos > other_rel.ent1.start_pos

    def __repr__(self):
        fmt = '({rel_id}, {category} Arg1:{ent1} Arg2:{ent2})'
        return fmt.format(**self.__dict__)


class Relations(object):
    def __init__(self, rels):
        self.rels = rels

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rels[key]
        elif isinstance(key, slice):
            return Relations(self.rels[key])

    def __add__(self, other):
        rels = self.rels + other.rels
        return Relations(rels)

    def find_relations(self, start_pos, end_pos):
        res = []
        for rel in self.rels:
            if start_pos <= rel.start_pos and end_pos >= rel.end_pos:
                res.append(rel)
        return Relations(res)

    def offset(self, offset_val):
        return Relations([rel.offset(offset_val) for rel in self.rels])

    @property
    def start_pos(self):
        return min([rel.start_pos for rel in self.rels])

    @property
    def end_pos(self):
        return max([rel.end_pos for rel in self.rels])

    def __len__(self):
        return len(self.rels)

    def __repr__(self):
        return self.rels.__repr__()


class TextSpan(object):
    def __init__(self, text, ents, rels, **kwargs):
        self.text = text
        self.ents = ents
        self.rels = rels

    def __getitem__(self, key):
        if isinstance(key, int):
            start, stop = key, key + 1
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self.text)
        else:
            raise ValueError('parameter should be int or slice')
        if start < 0:
            start += len(self.text)
        if stop < 0:
            stop += len(self.text)
        text = self.text[key]
        ents = self.ents.find_entities(start, stop).offset(-start)
        rels = self.rels.find_relations(start, stop).offset(-start)
        return TextSpan(text, ents, rels)

    def __len__(self):
        return len(self.text)


class Sentence(object):
    def __init__(self, doc_id, offset, text='', ents=[], rels=[], textspan=None):
        self.doc_id = doc_id
        self.offset = offset
        if isinstance(textspan, TextSpan):
            self.textspan = textspan
        else:
            self.textspan = TextSpan(text, ents, rels)

    @property
    def text(self):
        return self.textspan.text

    @property
    def ents(self):
        return self.textspan.ents

    @property
    def rels(self):
        return self.textspan.rels

    def abbreviate(self, max_len, ellipse_chars='$$'):
        if max_len <= len(ellipse_chars):
            return ''
        left_trim = (max_len - len(ellipse_chars)) // 2
        right_trim = max_len - len(ellipse_chars) - left_trim
        return self[:left_trim] + ellipse_chars + self[-right_trim:]

    def __getitem__(self, key):
        if isinstance(key, int):
            start, stop = key, key + 1
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self.text)
        else:
            raise ValueError('parameter should be int or slice')
        if start < 0:
            start += len(self.text)
        if stop < 0:
            stop += len(self.text)
        offset = self.offset + start
        textspan = self.textspan[start: stop]
        return Sentence(self.doc_id, offset, textspan=textspan)

    def __gt__(self, other):
        return self.offset > other.offse

    def __add__(self, other):
        if isinstance(other, str):
            return Sentence(doc_id=self.doc_id, offset=self.offset, text=self.text + other,
                            ents=self.ents, rels=self.rels)
        assert self.doc_id == other.doc_id, 'sentences should be from the same document'
        assert self.offset + len(self) <= other.offset, 'sentences should not have overlap'
        doc_id = self.doc_id
        text = self.text + other.text
        offset = self.offset
        ents = self.ents + other.ents.offset(len(self.text))
        rels = self.rels + other.rels.offset(len(self.text))
        return Sentence(doc_id=doc_id, offset=offset, text=text, ents=ents, rels=rels)

    def __len__(self):
        return len(self.textspan)


class Document(object):
    def __init__(self, doc_id, text, ents, rels):
        self.doc_id = doc_id
        self.textspan = TextSpan(text, ents, rels)

    @property
    def text(self):
        return self.textspan.text

    @property
    def ents(self):
        return self.textspan.ents

    @property
    def rels(self):
        return self.textspan.rels


class Documents(object):
    def __init__(self, data_dir, doc_ids=None):
        self.data_dir = data_dir
        self.doc_ids = doc_ids
        if self.doc_ids is None:
            self.doc_ids = self.scan_doc_ids()

    def scan_doc_ids(self):
        doc_ids = [fname.split('.')[0] for fname in os.listdir(self.data_dir)]
        doc_ids = [doc_id for doc_id in doc_ids if len(doc_id) > 0]
        return np.unique(doc_ids)

    def read_txt_file(self, doc_id):
        fname = os.path.join(self.data_dir, doc_id + '.txt')
        with open(fname, encoding='utf-8') as f:
            text = f.read()
        return text

    def parse_entity_line(self, raw_str):
        ent_id, label, text = raw_str.strip().split('\t')
        category, pos = label.split(' ', 1)
        pos = pos.split(' ')
        ent = Entity(ent_id, category, int(pos[0]), int(pos[-1]), text)
        return ent

    def parse_relation_line(self, raw_str, ents):
        rel_id, label = raw_str.strip().split('\t')
        category, arg1, arg2 = label.split(' ')
        arg1 = arg1.split(':')[1]
        arg2 = arg2.split(':')[1]
        ent1 = ents[arg1]
        ent2 = ents[arg2]
        return Relation(rel_id, category, ent1, ent2)

    def read_anno_file(self, doc_id):
        ents = []
        rels = []
        fname = os.path.join(self.data_dir, doc_id + '.ann')
        with open(fname, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('T'):
                ent = self.parse_entity_line(line)
                ents.append(ent)
        ents = Entities(ents)

        for line in lines:
            if line.startswith('R'):
                rel = self.parse_relation_line(line, ents)
                if rel.is_valid:
                    rels.append(rel)
        rels = Relations(rels)
        return ents, rels

    def __len__(self):
        return len(self.doc_ids)

    def get_doc(self, doc_id):
        text = self.read_txt_file(doc_id)
        ents, rels = self.read_anno_file(doc_id)
        doc = Document(doc_id, text, ents, rels)
        return doc

    def __getitem__(self, key):
        if isinstance(key, int):
            doc_id = self.doc_ids[key]
            return self.get_doc(doc_id)
        if isinstance(key, str):
            doc_id = key
            return self.get_doc(doc_id)
        if isinstance(key, np.ndarray) and key.dtype == int:
            doc_ids = self.doc_ids[key]
            return Documents(self.data_dir, doc_ids=doc_ids)


class SentenceExtractor(object):
    def __init__(self, sent_split_char, window_size, rel_types, filter_no_rel_candidates_sents=True):
        self.sent_split_char = sent_split_char
        self.window_size = window_size
        self.filter_no_rel_candidates_sents = filter_no_rel_candidates_sents
        self.rels_type_set = set()
        for rel_type in rel_types:
            self.rels_type_set.add(tuple(re.split('[-_]', rel_type)))

    def get_sent_boundaries(self, text):
        dot_indices = []
        for i, ch in enumerate(text):
            if ch == self.sent_split_char:
                dot_indices.append(i + 1)

        if len(dot_indices) <= self.window_size - 1:
            return [(0, len(text))]

        dot_indices = [0] + dot_indices
        if text[-1] != self.sent_split_char:
            dot_indices += [len(text)]

        boundries = []
        for i in range(len(dot_indices) - self.window_size):
            start_stop = (
                dot_indices[i],
                dot_indices[i + self.window_size]
            )
            boundries.append(start_stop)
        return boundries

    def has_rels_candidates(self, ents):
        ent_cates = set([ent.category for ent in ents])
        for pos_rel in permutations(ent_cates, 2):
            if pos_rel in self.rels_type_set:
                return True
        return False

    def extract_doc(self, doc):
        sents = []
        for start_pos, end_pos in self.get_sent_boundaries(doc.text):
            ents = []
            sent_text = doc.text[start_pos: end_pos]
            for ent in doc.ents.find_entities(start_pos=start_pos, end_pos=end_pos):
                ents.append(ent.offset(-start_pos))
            if self.filter_no_rel_candidates_sents and not self.has_rels_candidates(ents):
                continue
            rels = []
            for rel in doc.rels.find_relations(start_pos=start_pos, end_pos=end_pos):
                rels.append(rel.offset(-start_pos))
            sent = Sentence(doc.doc_id,
                            offset=start_pos,
                            text=sent_text,
                            ents=Entities(ents),
                            rels=Relations(rels))
            sents.append(sent)
        return sents

    def __call__(self, docs):
        sents = []
        for doc in docs:
            sents += self.extract_doc(doc)
        return sents


class EntityPair(object):
    def __init__(self, doc_id, sent, from_ent, to_ent):
        self.doc_id = doc_id
        self.sent = sent
        self.from_ent = from_ent
        self.to_ent = to_ent

    def __repr__(self):
        fmt = 'doc {}, sent {}, {} -> {}'
        return fmt.format(self.doc_id, self.sent.text, self.from_ent, self.to_ent)


class EntityPairsExtractor(object):
    def __init__(self, allow_rel_types, max_len=150, ellipse_chars='$$', pad=10):
        self.allow_rel_types = allow_rel_types
        self.max_len = max_len
        self.pad = pad
        self.ellipse_chars = ellipse_chars

    def extract_candidate_rels(self, sent):
        candidate_rels = []
        for f_ent, t_ent in permutations(sent.ents, 2):
            rel_cate = (f_ent.category, t_ent.category)
            if rel_cate in self.allow_rel_types:
                candidate_rels.append((f_ent, t_ent))
        return candidate_rels

    def make_entity_pair(self, sent, f_ent, t_ent):
        doc_id = sent.doc_id
        if f_ent.start_pos < t_ent.start_pos:
            left_ent, right_ent = f_ent, t_ent
        else:
            left_ent, right_ent = t_ent, f_ent
        start_pos = max(0, left_ent.start_pos - self.pad)
        end_pos = min(len(sent), right_ent.end_pos + self.pad)
        res_sent = sent[start_pos: end_pos]

        if len(res_sent) > self.max_len:
            res_sent = res_sent.abbreviate(self.max_len)
        f_ent = res_sent.ents[f_ent.ent_id]
        t_ent = res_sent.ents[t_ent.ent_id]
        return EntityPair(doc_id, res_sent, f_ent, t_ent)

    def __call__(self, sents):
        samples = []
        for sent in sents:
            for f_ent, t_ent in self.extract_candidate_rels(sent.ents):
                entity_pair = self.make_entity_pair(sent, f_ent, t_ent)
                samples.append(entity_pair)
        return samples


class Dataset(object):
    def __init__(self, entity_pairs, doc_ent_pair_ids=set(), word2idx=None, cate2idx=None, max_len=150):
        self.entity_pairs = entity_pairs
        self.doc_ent_pair_ids = doc_ent_pair_ids
        self.max_len = max_len
        self.word2idx = word2idx
        self.cate2idx = cate2idx

    def __len__(self):
        return len(self.entity_pairs)

    def build_vocab_dict(self, vocab_size=2000):
        counter = Counter()
        for ent_pair in self.entity_pairs:
            for char in ent_pair.sent.text:
                counter[char] += 1
        word2idx = dict()
        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1
        if vocab_size > 0:
            num_most_common = vocab_size - len(word2idx)
        else:
            num_most_common = len(counter)
        for char, _ in counter.most_common(num_most_common):
            word2idx[char] = word2idx.get(char, len(word2idx))
        self.word2idx = word2idx

    def vectorize(self, ent_pair):
        sent_vec = np.zeros(self.max_len, dtype='int')
        for i, c in enumerate(ent_pair.sent.text):
            sent_vec[i] = self.word2idx.get(c, 1)
        ents_vec = ent_pair.sent.ents.vectorize(vec_len=self.max_len, cate2idx=self.cate2idx)
        from_ent_vec = np.zeros(self.max_len, dtype='int')
        from_ent_vec[ent_pair.from_ent.start_pos: ent_pair.from_ent.end_pos] = 1
        to_ent_vec = np.zeros(self.max_len, dtype='int')
        to_ent_vec[ent_pair.to_ent.start_pos: ent_pair.to_ent.end_pos] = 1

        if (ent_pair.sent.doc_id, ent_pair.from_ent.ent_id, ent_pair.to_ent.ent_id) in self.doc_ent_pair_ids:
            label = 1
        else:
            label = 0
        return sent_vec, ents_vec, from_ent_vec, to_ent_vec, label

    def __getitem__(self, idx):
        sent_vecs, ents_vecs, from_ent_vecs, to_ent_vecs, ent_dists, labels = [], [], [], [], [], []
        entity_pairs = self.entity_pairs[idx]
        if not isinstance(entity_pairs, list):
            entity_pairs = [entity_pairs]
        for ent_pair in entity_pairs:
            sent_vec, ents_vec, from_ent_vec, to_ent_vec, label = self.vectorize(ent_pair)
            sent_vecs.append(sent_vec)
            ents_vecs.append(ents_vec)
            from_ent_vecs.append(from_ent_vec)
            to_ent_vecs.append(to_ent_vec)
            ent_dists.append(ent_pair.to_ent.start_pos - ent_pair.from_ent.end_pos)
            labels.append(label)

        sent_vecs = np.array(sent_vecs)
        ents_vecs = np.array(ents_vecs)
        from_ent_vecs = np.array(from_ent_vecs)
        to_ent_vecs = np.array(to_ent_vecs)
        ent_dists = np.array(ent_dists)
        labels = np.array(labels)

        return sent_vecs, ents_vecs, from_ent_vecs, to_ent_vecs, ent_dists, labels

def train_word_embeddings(entity_pairs, word2idx, *args, **kwargs):
    w2v_train_sents = []
    for ent_pair in entity_pairs:
        w2v_train_sents.append(list(ent_pair.sent.text))
    w2v_model = Word2Vec( w2v_train_sents,vector_size=300, window=3, workers=2 ) #*args, **kwargs)
    word2idx.update({w: i for i, w in enumerate(w2v_model.wv.index_to_key, start=len(word2idx))})
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    w2v_embeddings = np.zeros((len(word2idx), w2v_model.vector_size))
    for char, char_idx in word2idx.items():
        if char in w2v_model.wv:
            w2v_embeddings[char_idx] = w2v_model.wv[char]
    return word2idx, idx2word, w2v_embeddings

all_rel_types = set([tuple(re.split('[-_]' ,rel)) for rel in RELATIONS])
ent2idx = dict(zip(ENTITIES, range(1, len(ENTITIES) + 1)))

train_docs = Documents(train_data_dir)
test_docs = Documents(test_b_data_dir)

doc_ent_pair_ids = set()
for doc in train_docs:
    for rel in doc.rels:
        doc_ent_pair_id = (doc.doc_id, rel.ent1.ent_id, rel.ent2.ent_id)
        doc_ent_pair_ids.add(doc_ent_pair_id)

sent_extractor = SentenceExtractor(sent_split_char='。', window_size=2, rel_types=RELATIONS,
                                   filter_no_rel_candidates_sents=True)
train_sents = sent_extractor(train_docs)
test_sents = sent_extractor(test_docs)

train_sents[0].text

max_len = 150
ent_pair_extractor = EntityPairsExtractor(all_rel_types, max_len=max_len)
train_entity_pairs = ent_pair_extractor(train_sents)
test_entity_pairs = ent_pair_extractor(test_sents)

word2idx = {'<pad>': 0, '<unk>': 1}
word2idx, idx2word, w2v_embeddings = train_word_embeddings(
    entity_pairs=chain(train_entity_pairs, test_entity_pairs),
    word2idx=word2idx,
    size=100,
    iter=10
)

train_data = Dataset(train_entity_pairs, doc_ent_pair_ids, word2idx=word2idx, max_len=max_len, cate2idx=ent2idx)
test_data = Dataset(test_entity_pairs, word2idx=word2idx, max_len=max_len, cate2idx=ent2idx)


from keras import layers
from keras import backend as K
from keras.layers import Input, Embedding, Lambda
from keras.layers import Concatenate, Dense
from keras.layers import Conv1D, MaxPool1D, Flatten
from keras.models import Model

num_ent_classes = len(ENTITIES) + 1
ent_emb_size = 2
emb_size = w2v_embeddings.shape[-1]
vocab_size = len(word2idx)


def build_model():
    inp_sent = Input(shape=(max_len,), dtype='int32')
    inp_ent = Input(shape=(max_len,), dtype='int32')
    inp_f_ent = Input(shape=(max_len,), dtype='float32')
    inp_t_ent = Input(shape=(max_len,), dtype='float32')
    inp_ent_dist = Input(shape=(1,), dtype='float32')
    f_ent = Lambda(lambda x: K.expand_dims(x))(inp_f_ent)
    t_ent = Lambda(lambda x: K.expand_dims(x))(inp_t_ent)

    ent_embed = Embedding(num_ent_classes, ent_emb_size)(inp_ent)
    sent_embed = Embedding(vocab_size, emb_size, weights=[w2v_embeddings], trainable=False)(inp_sent)

    x = Concatenate()([sent_embed, ent_embed])
    x = Conv1D(64, 1, padding='same', activation='relu')(x)

    f_res = layers.multiply([f_ent, x])
    t_res = layers.multiply([t_ent, x])

    conv = Conv1D(64, 3, padding='same', activation='relu')
    f_x = conv(x)
    t_x = conv(x)
    f_x = layers.add([f_x, f_res])
    t_x = layers.add([t_x, t_res])

    f_res = layers.multiply([f_ent, f_x])
    t_res = layers.multiply([t_ent, t_x])
    conv = Conv1D(64, 3, padding='same', activation='relu')
    f_x = conv(x)
    t_x = conv(x)
    f_x = layers.add([f_x, f_res])
    t_x = layers.add([t_x, t_res])

    f_res = layers.multiply([f_ent, f_x])
    t_res = layers.multiply([t_ent, t_x])
    conv = Conv1D(64, 3, padding='same', activation='relu')
    f_x = conv(x)
    t_x = conv(x)
    f_x = layers.add([f_x, f_res])
    t_x = layers.add([t_x, t_res])

    conv = Conv1D(64, 3, activation='relu')
    f_x = MaxPool1D(3)(conv(f_x))
    t_x = MaxPool1D(3)(conv(t_x))

    conv = Conv1D(64, 3, activation='relu')
    f_x = MaxPool1D(3)(conv(f_x))
    t_x = MaxPool1D(3)(conv(t_x))

    f_x = Flatten()(f_x)
    t_x = Flatten()(t_x)

    x = Concatenate()([f_x, t_x, inp_ent_dist])
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model([inp_sent, inp_ent, inp_f_ent, inp_t_ent, inp_ent_dist], x)
    return model


tr_sent, tr_ent, tr_f_ent, tr_t_ent, tr_ent_dist, tr_y = train_data[:]

K.clear_session()
model = build_model()

model.compile('adam', loss='binary_crossentropy', metrics=['acc'])


model.fit(x=[tr_sent, tr_ent, tr_f_ent, tr_t_ent, tr_ent_dist],
          y=tr_y, batch_size=64, epochs=2)


te_sent, te_ent, te_f_ent, te_t_ent, te_ent_dist, te_y = test_data[:]
preds = model.predict(x=[te_sent, te_ent, te_f_ent, te_t_ent, te_ent_dist], verbose=1)

def generate_submission(preds, entity_pairs, threshold):
    doc_rels = defaultdict(set)
    for p, ent_pair in zip(preds, entity_pairs):
        if p >= threshold:
            doc_id = ent_pair.doc_id
            f_ent_id = ent_pair.from_ent.ent_id
            t_ent_id = ent_pair.to_ent.ent_id
            category = ent_pair.from_ent.category + '_' + ent_pair.to_ent.category
            category = category.replace('SideEff_Drug', 'SideEff-Drug')
            doc_rels[doc_id].add((f_ent_id, t_ent_id, category))
    submits = dict()
    tot_num_rels = 0
    for doc_id, rels in doc_rels.items():
        output_str = ''
        for i, rel in enumerate(rels):
            tot_num_rels += 1
            line = 'R{}\t{} Arg1:{} Arg2:{}\n'.format(i + 1, rel[2], rel[0], rel[1])
            output_str += line
        submits[doc_id] = output_str
    print('Total number of relations: {}. In average {} relations per doc.'.format(tot_num_rels, tot_num_rels / len(submits)))
    return submits

def output_submission(submit_file, submits, test_dir):
    dir_name = os.path.basename(submit_file)
    dir_name, _ = os.path.splitext(dir_name)
    with zipfile.ZipFile(submit_file, 'w') as zf:
        for doc_id, rels_str in submits.items():
            fname = '{}.ann'.format(doc_id)
            test_file = os.path.join(test_dir, fname)
            content = open(test_file, encoding='utf-8').read()
            content += rels_str
            zf.writestr(os.path.join(dir_name, fname), content)

submits = generate_submission(preds, test_entity_pairs, 0.5)

submit_file = 'submit_181205.zip'
output_submission(submit_file, submits, test_b_data_dir)

