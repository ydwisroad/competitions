#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-11-01

@author:Brook
"""
import csv
import re
import os
import json

from util import read_text, read_ann, BioTagger


if __name__ == "__main__":
    train_dir = "data/train_text"
    ann_dir = "data/train_ann"
    save_dir = "data/train"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    tagger = BioTagger()
    
    p = re.compile("\d+;\d+ ")
    for fname in os.listdir(train_dir):
        if not fname.endswith(".txt"):
            continue
        
        fn = fname.replace(".txt", "")
        text_path = os.path.join(train_dir, "%s.txt" % fn)
        ann_path = os.path.join(ann_dir, "%s.ann" % fn)

        text = read_text(text_path)
        text_len = len(text)
        ann_rows = read_ann(ann_path)
        
        ann_infos = []
        for row in ann_rows:
            # 实体类别和实体的起始和终止位置(以空格分割),出现分号表示实体跨行
            ann_info = row[1]
            # 合并跨行实体的索引
            ann_info = p.sub("", ann_info)
            # 按空格分割成实体类别，实体起始位置，实体结束位置
            type_, start, end = ann_info.split(" ")
            
            ann_infos.append((type_, int(start), int(end)))

        """采用IOB标注法标注"""
        tags = tagger.ind_to_seq(text_len, ann_infos)
            
        # 保存成json文件
        save_path = os.path.join(save_dir, "%s.train" % fn)
        
        with open(save_path, 'w') as f:
            writer = csv.writer(f, delimiter="\t")
            text = re.sub("\s", " ", text)  # tab 替换成空格
            for w, t in zip(text, tags):
                writer.writerow([w, t])


