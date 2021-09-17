#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-10-29

@author:Brook
"""
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def time_count(func):
    def func_(*args, **kwargs):
        start = time.time()
        logging.info("%s 正在运行..." % func.__name__)
        result = func(*args, **kwargs)
        delta = time.time() - start
        m = delta // 60
        s = delta % 60
        logging.info("%s 运行结束，耗时：%d min %d s" % (func.__name__, m, s ))
        return result
    return func_

