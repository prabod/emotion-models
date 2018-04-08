'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

import numpy as np
import pandas as pd
import csv
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation, GRU
from keras.datasets import imdb
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import RMSprop,Adam
import keras.backend as K
from keras.utils import print_summary
from nltk.tokenize import TweetTokenizer
import re
import datetime
import cPickle as pickle
from attlayer import Attention
from attdeep import AttentionWeightedAverage
from keras.regularizers import l2, L1L2

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

max_features = 20000
maxSeqLength = 50
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 16
print('Loading data...')
# glove = pd.read_table("glove.twitter.27B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# wordIndex = glove.index.tolist()
# wordIndex = map(lambda x: str(x).decode('utf-8') ,wordIndex)
# wordMatrix = glove.as_matrix()
# wordMatrix = wordMatrix.astype('float32')

# tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

# def getIds(sentence):
#     indx=np.ones((maxSeqLength), dtype='int32')
#     # cleaned = cleanSentences(sentence)
#     indexCounter = 0
#     for word in tknzr.tokenize(sentence):
#         if indexCounter < maxSeqLength:
#             try:
#                 indx[indexCounter] = wordIndex.index(word.lower())
#             except:
#                 indx[indexCounter] = 0
#         indexCounter = indexCounter + 1
#     return indx


# trainData = pd.read_table("datasets/2018-E-c-En-train.txt", sep="\t", index_col=0, quoting=csv.QUOTE_NONE)
# testData = pd.read_table("datasets/2018-E-c-En-test.txt", sep="\t", index_col=0, quoting=csv.QUOTE_NONE)
# devData = pd.read_table("datasets/2018-E-c-En-dev.txt", sep="\t", index_col=0, quoting=csv.QUOTE_NONE)

# x_train = trainData['Tweet'].apply(getIds)
# x_train = np.array(x_train.tolist())
# y_train = trainData.loc[:, trainData.columns != 'Tweet'].as_matrix()


# x_test = devData['Tweet'].apply(getIds)
# x_test = np.array(x_test.tolist())
# y_test = devData.loc[:, devData.columns != 'Tweet'].as_matrix()

# y_train = np.array(y_train.tolist())
# y_test = np.array(y_test.tolist())

# with open('preprocessed/x_train.p','wb') as f:
# 	pickle.dump(x_train,f)

# with open('preprocessed/y_train.p','wb') as f:
# 	pickle.dump(y_train,f)

# with open('preprocessed/x_test.p','wb') as f:
# 	pickle.dump(x_test,f)

# with open('preprocessed/y_test.p','wb') as f:
# 	pickle.dump(y_test,f)

with open('preprocessed/x_train.p','rb') as f:
	x_train = pickle.load(f)

with open('preprocessed/y_train.p','rb') as f:
	y_train = pickle.load(f)

with open('preprocessed/x_test.p','rb') as f:
	x_test = pickle.load(f)

with open('preprocessed/y_test.p','rb') as f:
	y_test = pickle.load(f)

with open('preprocessed/embedding_matrix.p','rb') as f:
	embedding_matrix = pickle.load(f)

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
# np.savetxt('test.out', x_train, delimiter=',') 
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
tbCallBack = TensorBoard(log_dir=logdir, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


print('Build model...')
# model = Sequential()
# model.add(Embedding(input_dim=22053,
#                     output_dim=200,
#                     mask_zero=True,
#                     # weights=[embedding_matrix],
#                     input_length=maxSeqLength,
#                     trainable=True,
#             		embeddings_regularizer=l2(1e-6)
#                     	))
# # model.add(Activation('tanh'))
# # model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
# model.add(SpatialDropout1D(0.1))
# model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
# model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
# model.add(AttentionWeightedAverage())
# model.add(Dropout(0.5))
# model.add(Dense(11, activation='sigmoid'))



model_input = Input(shape=(maxSeqLength,), dtype='int32')
embed_reg = L1L2(l2=1e-8)
embed = Embedding(input_dim=19852,
                  output_dim=200,
                  mask_zero=True,
                  weights=[embedding_matrix],
                  input_length=maxSeqLength,
                  embeddings_regularizer=embed_reg,
                  trainable=True,
                  name='embedding')
x = embed(model_input)
x = Activation('tanh')(x)
embed_drop = SpatialDropout1D(0.1, name='embed_drop')
x = embed_drop(x)
lstm_0_output = Bidirectional(LSTM(512, return_sequences=True,recurrent_dropout=0.5, dropout=0.2,kernel_regularizer=embed_reg),  name="bi_lstm_0")(x)
lstm_1_output = Bidirectional(LSTM(512, return_sequences=True,recurrent_dropout=0.5, dropout=0.2,kernel_regularizer=embed_reg), name="bi_lstm_1")(lstm_0_output)
x = concatenate([lstm_1_output, lstm_0_output, x])
x = AttentionWeightedAverage(name='attlayer')(x)
x = Dropout(0.5)(x)
outputs = [Dense(11, activation='sigmoid', name='softmax')(x)]
model = Model(inputs=[model_input], outputs=outputs, name="DeepMoji")


# model = Sequential()
# model.add(Embedding(input_dim=19852,
#                     output_dim=50,
#                     mask_zero=True,
#                     # weights=[embedding_matrix],
#                     input_length=maxSeqLength,
#                     trainable=True,
#             		# embeddings_regularizer=l2(1e-6)
#                     	))
# # model.add(Activation('tanh'))
# # model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
# # model.add(SpatialDropout1D(0.1))
# model.add(LSTM(50))
# # model.add(LSTM(256, return_sequences=True))
# # model.add(AttentionWeightedAverage())
# # model.add(Dropout(0.5))
# model.add(Dense(11, activation='sigmoid'))



# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=Adam(clipnorm=1, lr=0.001),
              metrics=['accuracy'])
print_summary(model)
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[reduce_lr,tbCallBack])
model.save('my_model_batch_att_50_001.h5')
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
