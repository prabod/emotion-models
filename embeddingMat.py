import numpy as np
import pandas as pd
import csv
from nltk.tokenize import TweetTokenizer
import re
import datetime
import cPickle as pickle

maxSeqLength = 50
glove = pd.read_table("glove.twitter.27B.200d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)

vocab = []

def getvocab(sentence):
	global vocab
	vocab = vocab + map(lambda x: x.lower(),tknzr.tokenize(sentence))


trainData = pd.read_table("datasets/2018-E-c-En-train.txt", sep="\t", index_col=0, quoting=csv.QUOTE_NONE)
testData = pd.read_table("datasets/2018-E-c-En-test.txt", sep="\t", index_col=0, quoting=csv.QUOTE_NONE)
devData = pd.read_table("datasets/2018-E-c-En-dev.txt", sep="\t", index_col=0, quoting=csv.QUOTE_NONE)


trainData['Tweet'].apply(getvocab)
devData['Tweet'].apply(getvocab)

vocab = list(set(vocab))
print len(vocab)
embedding_matrix = np.zeros((len(vocab)+1, 200))
c=0
for i in range(len(vocab)):
	try:
		embedding_matrix[i] = np.array(glove.loc[vocab[i].lower()].tolist())
	except:
		c+=1
		pass
print c
print embedding_matrix.shape
print embedding_matrix[10]
print embedding_matrix[1]
print embedding_matrix[122]
print embedding_matrix[1012]
def getIds(sentence):
    indx=np.zeros((maxSeqLength), dtype='int32')
    # cleaned = cleanSentences(sentence)
    indexCounter = 0
    for word in tknzr.tokenize(sentence):
        if indexCounter < maxSeqLength:
            try:
                indx[indexCounter] = vocab.index(word.lower())
            except:
                indx[indexCounter] = len(vocab)
        indexCounter = indexCounter + 1
    return indx


x_train = trainData['Tweet'].apply(getIds)
x_train = np.array(x_train.tolist())
y_train = trainData.loc[:, trainData.columns != 'Tweet'].as_matrix()


x_test = devData['Tweet'].apply(getIds)
x_test = np.array(x_test.tolist())
y_test = devData.loc[:, devData.columns != 'Tweet'].as_matrix()

y_train = np.array(y_train.tolist())
y_test = np.array(y_test.tolist())

with open('preprocessed/x_train.p','wb') as f:
	pickle.dump(x_train,f)

with open('preprocessed/y_train.p','wb') as f:
	pickle.dump(y_train,f)

with open('preprocessed/x_test.p','wb') as f:
	pickle.dump(x_test,f)

with open('preprocessed/y_test.p','wb') as f:
	pickle.dump(y_test,f)

with open('preprocessed/vocab.p','wb') as f:
	pickle.dump(vocab,f)


with open('preprocessed/embedding_matrix.p','wb') as f:
	pickle.dump(embedding_matrix,f)