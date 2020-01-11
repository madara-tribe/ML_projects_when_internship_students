import pandas as pd
import numpy as np
import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence

def read_csv(csv_name):
    ll=pd.read_csv(csv_name)
    matrix = ll.drop("Unnamed: 0", axis=1)
    matrix = matrix.dropna()
    matrix = matrix.reset_index()
    matrix = matrix.drop("index", axis=1)
    return matrix


def create_dataset_sentence(matrix):
    for i,v in matrix.iterrows():
        for name, doc in zip([v['title']], [v['contents']]):
            yield to_words(doc, name)


def to_words(doc, name=''):
    mecab = MeCab.Tagger("-Ochasen")
    lines = mecab.parse(doc).splitlines()
    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    return LabeledSentence(words=words, tags=[name])


# In[10]:


def doc2vec_train(sentence):
    PASSING_PRECISION=199
    sentence=list(sentence)
    model = models.Doc2Vec(size=400, alpha=0.0015, sample=1e-4, min_count=1, workers=4)
    model.build_vocab(sentence)
    for x in range(50):
        print(x)
        model.train(sentence,total_examples=model.corpus_count, epochs=model.iter)
        ranks = []
        for doc_id in range(200):
            inferred_vector = model.infer_vector(sentence[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
            rank = [docid for docid, sim in sims].index(sentence[doc_id].tags[0])
            ranks.append(rank)
        print(collections.Counter(ranks))
        if collections.Counter(ranks)[0] >= PASSING_PRECISION:
            break
    return model



def save_trained_doc2vec_model():
    article_matrix = read_csv('new_jornal.csv')
    jornal_inf=[]
    for number,(i,v) in enumerate(matrix.iterrows()):
        for idd,tit, bun in zip([v['id']],[v['title']], [v['contents']]):
            jornal_inf.append([idd,str(number)+tit, bun])


    jornal_matrix = pd.DataFrame(jornal_inf, columns = ['id', 'title', 'contents'])
    dataset_sentence = create_dataset_sentence(jornal_matrix)
    model = doc2vec_train(dataset_sentence)
    model.save('doc2vec.model')



### check_model_performance ###

sample_word = to_words(sample_matrix['contents'][1000]).words
def show_similar_docs(model_name, sample_words, show_words = True):
    model = models.Doc2Vec.load(model_name)
    print(sample_matrix['contents'][1000])
    x = model.infer_vector(sample_words)
    most_similar_texts = model.docvecs.most_similar([x])
    for similar_text in most_similar_texts:
        print(similar_text[0])
    if show_words:
        for w in words:
            print(w + ':')
            result = [result[0] for result in model.most_similar(positive=w, topn=10)]
            print(result)


def show_doc2vec_sim_rate(model, sample_matrix):
    print('show similar title')
    model.docvecs.most_similar(sample_matrix['title'][1001], topn=10)
    model.most_similar(positive="文庫本", topn=10) # get most similar words: 'topn' indicate how many times
    print('show sim rate')
    model.docvecs.similarity(matrix['title'][1000], matrix['title'][1500])  #　similarity of compared 2 docs

if __name__ == '__main__':
    save_trained_doc2vec_model():
