import re
import numpy as np
import cPickle as pickle
import csv
import pandas as pd
from keras.models import load_model
from nltk.tokenize import TweetTokenizer
import keras.backend as K
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

max_features = 20000
maxSeqLength = 35
maxlen = 35  # cut texts after this number of words (among top max_features most common words)
batch_size = 24

print('Loading data...')
glove = pd.read_table("glove.twitter.27B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
wordIndex = glove.index.tolist()
wordMatrix = glove.as_matrix()
wordMatrix = wordMatrix.astype('float32')

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

def getIds(sentence):
    indx=np.ones((maxSeqLength), dtype='int32')
    # cleaned = cleanSentences(sentence)
    indexCounter = 0
    for word in tknzr.tokenize(sentence):
        if indexCounter < maxSeqLength:
            try:
                indx[indexCounter] = wordIndex.index(word.lower())
            except:
                indx[indexCounter] = 0
        indexCounter = indexCounter + 1
    return indx

model = load_model('my_model_batch_25.h5',custom_objects={'multitask_loss': multitask_loss})

while True:
    tweet = input(">")
    print model.predict(np.array([getIds(tweet)])) > 0.5
    print model.predict(np.array([getIds(tweet)]))
    print getIds(tweet)
