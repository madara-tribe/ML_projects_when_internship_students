from natto import MeCab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

SAMPLE_AD_WORDS = '【値下げ！】弊社限定未公開物件！豊田駅徒歩８分 ライオンズガーデン明大前パラダイム' # For example

line=pd.read_csv("csv_file")  # Set traing data
word_group=pd.DataFrame(line.columns.drop("Unnamed: 0"),columns={ 'word'}).T

word_index_dic={v:i for i, v in enumerate(word_group.loc["word"])}  # 2838

word_split=[i for i in re.split('[【】！" "]', SAMPLE_AD_WORDS) if i!=""] # building name ('ライオンズガーデン明大前パラダイム')


word_dictionay=[]
for m in word_split:
    for i,v in word_index_dic.items():
        word_dictionay.append([m,i,v])   # Matrix that has three columns: building_name, word_group.loc["word"], index

dictionary_new = {}
for l,m,n in word_dictionay:
    if l.count(m):
        dictionary_new.setdefault(l, []).append(None)
    else:
        dictionary_new.setdefault(l, []).append(m) # dictionary of Building name split by re.split('[【】！" "]

pd.DataFrame(dictionary_new).T  # Make list of dictionary of Building name: 4 * 2838
