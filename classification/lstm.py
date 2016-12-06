import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import cPickle as pickle
from collections import Counter

#load y
y = pickle.load(open('../data/columns.pickle'))['sex']
for i in xrange(len(y)):
	if y[i] == 'm':
		y[i] = 1
	else:
		y[i] = 0

#load X
# X = pickle.load(open('../data/words_as_frequencies.pickle'))
top_words = 10000 # words which occur more than 2000 times

text = pickle.load(open('../data/columns.pickle'))['text']
frequencies = {}
for entry in text:
	for word in entry:
		if word in frequencies:
			frequencies[word] = frequencies[word] + 1
		else:
			frequencies[word] = 1
fs = sorted(frequencies, key=frequencies.get)
fs.reverse()
fs = fs[:top_words]
fs_index=0
fs_dict = {}
for word in fs:
	fs_dict[word] = fs_index
	fs_index += 1
waf = []
for entry in text:
	e = []
	for word in entry:
		if word in fs_dict:
			e.append(fs_dict[word])
	waf.append(e)
X = waf


# truncate and pad input sequences
max_review_length = 400
X = sequence.pad_sequences(X, maxlen=max_review_length)

#create model architecture
embedding_vecor_length = 32
nodes = 512
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(nodes))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#begin training
print(model.summary())
print "topwords", top_words
print "maxrevlen", max_review_length
print "nodes", nodes
model.fit(X, y, nb_epoch=100, batch_size=256, callbacks=[early_stopping], validation_split = 0.3, shuffle=True, show_accuracy=True)




