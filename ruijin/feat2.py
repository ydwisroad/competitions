#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:25:05 2018

@author: brook
"""
import re
import json
from pypinyin import lazy_pinyin, Style  # 拼音


class Feature:
    """crf特征
    """
    _NUM_ZH = tuple(list("一二三四五六七八九十百千万两"))
    _TIME_UNIT = tuple(list("年周月天时分秒dhm"))
    
    def _cal_offset_index(self, sent, i, offset):
        """
            计算真实偏移量(剔除空格后的偏移)
        """
        cnt = 0
        if offset < 0:
            for j in range(i-1, -1, -1):
                if re.search("\S", sent[j]):
                    cnt += 1
                if cnt == abs(offset):
                    start = j
                    break
            else:
                 start = None
        elif offset > 0:
            for j in range(i+1, len(sent)):
                if not re.search("\s", sent[j]):
                    cnt += 1
                if cnt == offset:
                    start = j
                    break
            else:
                 start = None
        else:
             start = i
        return start

    def around_word(self, sent, i, offset_1, offset_2):
        assert offset_1 < offset_2, "offset_1 must be less than offset_2"
        index_1 = self._cal_offset_index(sent, i, offset_1)
        index_2 = self._cal_offset_index(sent, i, offset_2)
        if index_1 is None or index_2 is None:
            return {}
        else:
            key = "W%d : W%d" % (offset_1, offset_2)
            words = re.sub("\s", "", sent[index_1:index_2+1])
            return {key:words.lower()}

    def _word2feat(self, word: str, ref_ind):
        feature = {
            "%s:word.lower()" % ref_ind: word.lower(),
            "%s:word.is_digit" % ref_ind: word.isdigit(),
            "%s:word.is_title" % ref_ind: word.istitle(),
            "%s:word.is_letter" % ref_ind: bool(re.search("[a-zA-Z]", word)),
            "%s:word.is_blank" % ref_ind: bool(re.search("\s", word)),
            "%s:word.pinyin_inital" % ref_ind: lazy_pinyin(word, style=Style.FIRST_LETTER)[0],
            "%s:word.is_zh_num" % ref_ind: word in self._NUM_ZH,
            "%s:word.is_time_unit" % ref_ind: word in self._TIME_UNIT,
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

        #if i > 2:
        #    feats.update(self._word2feat(sent[i-3], "-3"))
            
        if i == sent_len - 1:
            feats["EOS"] = True

        if i < sent_len - 1:
            feats.update(self._word2feat(sent[i+1], "+1"))

        if i < sent_len - 2:
            feats.update(self._word2feat(sent[i+2], "+2"))

        #if i < sent_len - 3:
        #    feats.update(self._word2feat(sent[i+3], "+3"))
        
        feats.update(self.around_word(sent, i, -1, 0))
        feats.update(self.around_word(sent, i, 0, 1))

        feats.update(self.around_word(sent, i, -2, 0))
        feats.update(self.around_word(sent, i, -1, 1))
        feats.update(self.around_word(sent, i, 0, 2))

        feats.update(self.around_word(sent, i, -3, 0))
        feats.update(self.around_word(sent, i, 0, 3))
        
        #if i < sent_len -3:
        #    feats.update({"w:+3":sent[i+3].lower()})
        #if i < sent_len -4:
        #    feats.update({"w:+4":sent[i+4].lower()})
        #if i < sent_len -5:
        #    feats.update({"w:+5":sent[i+5].lower()})
        return feats
    
    def __call__(self, sents):
        for sent in sents:
            feats = [self.word2feat(sent, i) for i in range(len(sent))]
            yield feats


feature = Feature()



if __name__ == "__main__":

    s = "我是 一 个 粉 刷匠，粉刷本领强"
    i, f1, f2 = 6,-3,-1
    print(s)
    print(i, f1, f2)
    print(feature.around_word(s, i, f1,f2))
