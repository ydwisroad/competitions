#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:18:29 2018
@author: LHQ
"""
import os
import re

#from sklearn.externals import joblib
from joblib import *
from joblib import numpy_pickle

from conf import get_config
from util import BioTagger, load_object

tagger = BioTagger()


def read_data(data_dir):
    for fname in os.listdir(data_dir):
        test_path = os.path.join(data_dir, fname)
        with open(test_path) as f:
            text = f.read()
        yield fname, text


if __name__ == "__main__":
    config = get_config()

    feat = load_object(config.get("NORMAL", "feat"))

    test_dir = "data/test"
    result_dir = "data/submit"
    
    crf = numpy_pickle.load('data/models/crf.m')

    for fname, text in read_data(test_dir):
        print(fname)
        sents = [text]
        y = crf.predict(feat(sents))
        anns = tagger.seq_to_ind(y[0])
        anns = sorted(anns, key=lambda x:(x[1],x[2]))
        ann_fname = fname.replace(".txt", ".ann")
        save_path = os.path.join(result_dir, ann_fname)
        with open(save_path, 'w') as f:
            for i, (type_, s, e) in enumerate(anns):
                f.write("T{tid}\t{type_} {start} {end}\t{name}\n".format(tid=i,
                        type_=type_,
                        start=s,
                        end=e,
                        name=re.sub("\s", "", text[s:e])))
    
    
