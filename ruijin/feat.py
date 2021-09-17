#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:25:05 2018

@author: brook
"""
import re
from pypinyin import lazy_pinyin, Style  # 拼音


class Feature:
    """crf特征
    """
    
    _NUM_ZH = tuple(list("一二三四五六七八九十百千万两"))
    _TIME_UNIT = tuple(list("年周月天时分秒dhm"))
    
    def _word2feat(self, word: str, relative_index):
        feature = {
            "%s:word.lower()" % relative_index: word.lower(),
            "%s:word.is_digit" % relative_index: word.isdigit(),
            "%s:word.is_title" % relative_index: word.istitle(),
            "%s:word.is_blank" % relative_index: bool(re.search("\s", word)),
            "%s:word.pinyin_inital" % relative_index: lazy_pinyin(word, style=Style.FIRST_LETTER)[0],
            "%s:word.is_zh_num" % relative_index: word in self._NUM_ZH,
            "%s:word.is_time_unit" % relative_index: word in self._TIME_UNIT,
                }
        return feature
    
    def word2feat(self, sent, i):
        """构造特征
        
        Args
        ----
        sent : 字符串， 句子
        i : 字在句子中的索引位置
        
        Returns
        -------
        feats : dict, 特征
        """
        feats = {}
        feats.update(self._word2feat(sent[i], "0"))
        sent_len = len(sent)

        if i == 0:
            feats["BOS"] = True

        if i > 0:
            feats.update(self._word2feat(sent[i-1], "-1"))

        if i > 1:
            feats.update(self._word2feat(sent[i-2], "-2"))

        if i > 2:
            feats.update(self._word2feat(sent[i-3], "-3"))
            
        if i == sent_len - 1:
            feats["EOS"] = True

        if i < sent_len - 1:
            feats.update(self._word2feat(sent[i+1], "+1"))

        if i < sent_len - 2:
            feats.update(self._word2feat(sent[i+2], "+2"))

        if i < sent_len - 3:
            feats.update(self._word2feat(sent[i+3], "+3"))
        
        feats["w-1:w"] = sent[max(0,i-1):i+1].lower()
        feats["w:w+1"] = sent[i:i+2].lower()

        feats["w-2:w"] = sent[max(i-2,0):i+1].lower()
        feats['w-1:w+1'] = sent[max(i-1,0):i+2].lower()
        feats["w:w+2"] = sent[i:i+3].lower()
        feats["bias"] = 1
            
        return feats
    
    def sents2feats(self, sents):
        for sent in sents:
            feats = [self.word2feat(sent, i) for i in range(len(sent))]
            yield feats

    def __call__(self, sents):
        for sent in sents:
            feats = [self.word2feat(sent, i) for i in range(len(sent))]
            yield feats



feature = Feature()

