#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-10-28

@author:Brook

"""
import os
from itertools import islice
from datetime import datetime

#https://github.com/huyanluanyu1949/tian-chi-ruijin-mmc

#https://github.com/TeamHG-Memex/sklearn-crfsuite/tree/master/sklearn_crfsuite
#https://github.com/joblib/joblib/tree/master/joblib

#import sklearn_crfsuite
from sklearn_crfsuite import *  #metrics
from sklearn_crfsuite import metrics
#from sklearn.externals import joblib
from joblib import *
from joblib import numpy_pickle

from util import load_object, load_corpus
from conf import get_config
from util.misc import time_count

@time_count
def train(conf):
    train_dir = conf.get("TRAIN", "train_dir")

    model_path = conf.get("NORMAL", "model_path")
    report_dir = conf.get("TRAIN", "report_dir")
    N = conf.getint("TRAIN", "valdata_num")

    feat = load_object(conf.get("NORMAL", 'feat'))

    #crf = sklearn_crfsuite.CRF(
    crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            )

    X, y = zip(*load_corpus(train_dir))
        
    X_train = feat(X[:-N])
    y_train = y[:-N]

    X_validate = feat(X[-N:])
    y_validate = y[-N:]

    crf.fit(X_train, y_train)
    
    numpy_pickle.dump(crf, model_path)

    # 性能测试
    y_pred = crf.predict(X_validate)

    labels = list(crf.classes_)
    labels.remove("O")
    
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    
    #report = metrics.flat_classification_report(
    #    y_validate, y_pred, labels=sorted_labels, digits=3
    #)
    #print(report)
    #feat_example = list(feat(X[:1]))[0][3]
    #report_name = "%s.txt" % datetime.now().strftime("%Y%m%d_%H%M")
    #report_path = os.path.join(report_dir, report_name)
    #with open(report_path, 'w') as f:
    #    f.write(report)
    #    f.write("\n\n\n=============================================")
    #    f.write("\n[feat example]\n")
    #    for k, v in feat_example.items():
    #        f.write("'%s': '%s'\n" % (k, v))


if __name__ == "__main__":
    conf = get_config()
    train(conf)

