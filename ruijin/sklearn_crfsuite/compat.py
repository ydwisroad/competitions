# -*- coding: utf-8 -*-
try:
    #from sklearn.base import BaseEstimator
    from sklearn_crfsuite.compat import BaseEstimator
except ImportError:
    class BaseEstimator(object):
        pass
