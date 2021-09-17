#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-10-28

@author:Brook
"""
import re
import os
import csv
import json
from importlib import import_module


def load_corpus2(train_dir):
    for fname in os.listdir(train_dir):
        path = os.path.join(train_dir, fname)
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            words, tags = zip(*reader)
            yield "".join(words), tags


def load_corpus(train_dir):
    for fname in os.listdir(train_dir):
        path = os.path.join(train_dir, fname)
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            words, tags = zip(*reader)
            sent = "".join(words)
            previous_end = 0
            for m in re.finditer("。", sent):
                start, end = m.span()
                s = sent[previous_end:start+1]
                t = tags[previous_end:start+1]
                previous_end = end
                yield s, t


def load_object(path):
    """Load an object given its absolute object path, and return it.
    object can be a class, function, variable or an instance.
    """

    try:
        dot = path.rindex('.')
    except ValueError:
        raise ValueError("Error loading object '%s': not a full path" % path)

    module, name = path[:dot], path[dot+1:]
    mod = import_module(module)

    try:
        obj = getattr(mod, name)
    except AttributeError:
        raise NameError("Module '%s' doesn't define any object named '%s'" % (module, name))

    return obj


def read_ann(path, encoding="utf8"):
    """读取标注信息文件
    """
    with open(path, encoding=encoding) as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [row for row in reader]
    return rows

 
def read_text(path, encoding="utf8"):
    with open(path, encoding=encoding) as f:
        text = f.read()
    return text

