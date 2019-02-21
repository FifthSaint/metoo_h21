# -*- coding: utf-8 -*-
"""
Spyder Editor
미투댓글 토픽모델링 분석
based on TM_code.py
2019.2.20
author: sage5th
"""

# * set directory *
import os

os.getcwd()
Path='C:\\Users\\hannews\\Documents\\2019 미투댓글 한21\\code'
os.chdir(Path)

# * reading data *
import pandas as pd

filename='Comments_Naver_select_래퍼산이.csv'
df = pd.read_csv(filename, encoding='cp949')

reply = df['content'].tolist()

# * parsing *
import konlpy
from konlpy.tag import Komoran
komoran = Komoran()

words_list = [komoran.nouns(" ".join(str(item).splitlines())) for item in reply] # remove empty lines which return error

# 1번 등장하는 단어 제거
from collections import defaultdict

frequency = defaultdict(int)
for text in words_list:
    for token in text:
        frequency[token] += 1
words_list_u = [[token for token in text if frequency[token] > 1] for text in words_list]

# 길이가 1인 단어 제거
for text in list(words_list_u):
    for token in text:
        if len(token) < 2:
            text.remove(token)

# * gensim LDA model 만들기 *
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora
from gensim.models import LdaModel

Theme  = 'SanE'
K = 12 # the number of the topics: women 50, gay 25
random_state = 1 # set random seed
iterations = 1000 # 
fname=Theme+"K"+str(K)+"R"+str(random_state)+"I"+str(iterations)

dictionary = corpora.Dictionary(words_list_u)
# save dictionary
dictionary.save(fname+'_dictionary.pkl')
# load dictionary
dictionary = corpora.Dictionary.load(fname+'_dictionary.pkl')

corpus = [dictionary.doc2bow(text) for text in words_list_u]
# save corpus
corpora.MmCorpus.serialize(fname+'_corpus.mm', corpus)
# load corpus
corpus = corpora.MmCorpus(fname+'_corpus.mm')

lda = LdaModel(corpus, 
               num_topics=K, 
               id2word=dictionary,
               random_state=random_state, 
               iterations=iterations)

lda.save(fname)
# load LDA
lda = LdaModel.load(fname)

# visualization
import pyLDAvis.gensim

prepared_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
vname="ldavis"+fname+".html"
pyLDAvis.save_html(prepared_data, vname)