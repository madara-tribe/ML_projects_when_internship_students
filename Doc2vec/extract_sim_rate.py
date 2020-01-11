import pandas as pd
import numpy as np
import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence
from collections import OrderedDict

def read_csv(csv_name):
    ll=pd.read_csv(csv_name)
    matrix = ll.drop("Unnamed: 0", axis=1)
    matrix = matrix.dropna()
    matrix = matrix.reset_index()
    matrix = matrix.drop("index", axis=1)
    return matrix

def show_most_sim_title(model, matrix):
    for i in model.docvecs.most_similar(matrix['title'][1001], topn=10):
        print(i[0],i[1])



def make_similar_rate_matrix(dic):
    ron=[]
    for idx in range(10):
        ron.append('title_{:02d}'.format(idx))
        ron.append('contents_{:02d}'.format(idx))
    dics=pd.DataFrame(dic,index =ron).T
    dics.to_csv('similar_rate.csv')

def make_csv():
    article_matrix = read_csv('new_jornal.csv')
    model = models.Doc2Vec.load('doc2vec.model')

    dic = OrderedDict()
    for number,value in enumerate(article_matrix['title']):
        for t,s in model.docvecs.most_similar(article_matrix['title'][number], topn=10):
            dic.setdefault(value, []).append(t)
            dic.setdefault(value, []).append(s)
    make_similar_rate_matrix(dic)

if __name__ == '__main__':
    make_csv()
