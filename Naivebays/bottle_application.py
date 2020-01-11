from bottle import route, run, request, HTTPResponse
# -*- coding: utf-8 -*-
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import re
from sklearn.externals import joblib
#mecab
from natto import MeCab
nm = MeCab()
print(nm)
m = MeCab("-Owakati")
MODEL_FILE='TrainedNaivebaysmodel.csv'
line=pd.read_csv("AS_WORDS_DATA.csv")
clf = joblib.load(MODEL_FILE)

@route('/housmart', method='POST')
def clean_name():
    building_name = request.json['building_name']
    print(building_name)
    word_group=pd.DataFrame(line.columns.drop("Unnamed: 0"),columns={ 'word'}).T
    word_index_dic={v:i for i, v in enumerate(word_group.loc["word"])}

    word_split=[i for i in re.split('[【】！" "]', building_name) if i!=""]
    word_dictionay=[]
    for m in word_split:
        for i,v in word_index_dic.items():
            word_dictionay.append([m,i,v])
    dictionary_new = {}
    for l,m,n in word_dictionay:
        if l.count(m):
            dictionary_new.setdefault(l, []).append(None)
        else:
            dictionary_new.setdefault(l, []).append(m)
    def attach_bit(x):
        indexs = [1 if x == None else 0]
        return indexs
    com_dictionary=pd.DataFrame(dictionary_new).T.applymap(attach_bit)

    clf.predict(com_dictionary)#予測
    for n,l in zip(clf.predict(com_dictionary),com_dictionary.index):
        if n==1:
            print(l)
run(host='localhost', port=8080, debug=True)
