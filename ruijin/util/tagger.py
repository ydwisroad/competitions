#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:34:18 2018
@author: LHQ
"""
import csv



class Seq:
    """
        这边的一大段逻辑主要功能是，如果出现两个实体的索引位置有交叉
        则选择PRIOS的权重最大(索引位置最小) 的来标记
    """
    
    PRIORS = ('Disease', 'Test', 'Reason', 'Level', 'Symptom', 'Anatomy', 'Drug', 'Test_Value', 'Method', 'Treatment', 'SideEff', 'Frequency', 'Amount')
    
    def __init__(self, length):
        self.len = length
        self.seq = ['O'] * length

    def __getitem__(self, i):
        return self.seq[i]

    def __setitem__(self,i, value):
        self.seq[i] = value

    def __len__(self):
        return len(self.seq)
        
    def get_type(self, index):
        ele = self.seq[index]
        if ele == "O":
            return None
        else:
            return ele[2:]
        
    def get_tag(self, index):
        return self.seq[index][0]
        
    def _clean_right(self, index):
        for i in range(index+1, self.len):
            if self.get_tag(i) != "I":
                break
            else:
                self.seq[i] = "O"
    def _clean_left(self, index):
        for i in range(index,-1,-1):
            if self.get_tag(i) == "B":
                self.seq[i] = "O"
                break
            else:
                self.seq[i] = "O"
                
    def clean_tag(self, index):
        if self.get_tag(index) == "B":
            self.seq[index] = "O"
            self._clean_right(index)
            
        elif self.get_tag(index) == "I":
            self.seq[index] = "O"
            self._clean_right(index)
            self._clean_left(index)
                
        else:
            pass
        
    def check_clean(self, type_, start, end):
        p2 = self.PRIORS.index(type_)
        for i in range(start, end):
            if self.get_tag(i) != "O":
                p1 = self.PRIORS.index(self.get_type(i))
                if p1 <= p2:
                    return False
                else:
                    print("%s -> %s" % (self.get_type(i), type_))
                    self.clean_tag(i)
                    return True
        else:
            return False


class Tagger:
    
    def ind_to_seq(self, seq_len, tag_indexes):
        raise NotImplemented
    
    def seq_to_ind(self, tag_seq):
        raise NotImplemented


class BioTagger(Tagger):
    """
    BIO标注法标注
    """

    def ind_to_seq(self, seq_len, tag_indexes): 
        """实体索引转换成BIO标注序列,若两个实体的索引重叠，只取其一
        
        Args
        ----
        seq_len : int, 原始文档的长度
        tag_indexes : iterable, 迭代 (实体类型, 实体起始索引, 实体结束索引)元组
        
        Returns
        -------
        tag_seq : list, bio标注序列
        """
#        tag_seq = ['O'] * seq_len
        tag_seq = Seq(seq_len)
        for type_, start, end in tag_indexes:
            if len(set(tag_seq[start:end])) > 1:
                if not tag_seq.check_clean(type_, start, end):
                    continue
            tag_seq[start] = "B-%s" % type_
            tag_seq[start+1:end] = ["I-%s" % type_] * (end-start-1)
        return tag_seq.seq

    def seq_to_ind(self, tag_seq):
        """BIO标注序列转换成索引格式
        Args
        ----
        tag_seq : iterable, bio标注序列
        
        Returns
        -------
        tag_ind : list, 列表的每个元素是个三元元组:
              (实体类别，实体起始索引, 实体结束索引) 
        """
        tag_ind = []
        cur_tag = "o"
        type_ = "non"
        start = 0
        for i, tag in enumerate(tag_seq):
            if tag == "O":
                if cur_tag == "i":
                    tag_ind.append((type_, start, i))
                start = i
                cur_tag = 'o'
                
            elif tag.startswith("B-"):
                if cur_tag == "i":
                    tag_ind.append((type_, start, i))
                start = i
                cur_tag = "b"
                type_ = tag.replace("B-", "")
                
            elif tag.startswith("I-"):
                cur_tag = 'i'
            else:
                raise ValueError("unknown tag: %s" % tag)
        return tag_ind
