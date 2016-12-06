import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
import copy

#import the data
import cPickle as pickle
initial_data = pickle.load(open('columns.pickle', 'rb'))
texts = [' '.join(x) for x in initial_data['text']]

#model as unigram
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=2)
X_train_counts = count_vect.fit_transform(texts)
print "Count shape", X_train_counts.shape

#transform via tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print "Tfidf shape", X_train_tfidf.shape

#optional: reduce dimensions using LDA
from sklearn.decomposition import LatentDirichletAllocation
lda_transformer = LatentDirichletAllocation(n_topics = 300)
X_train_lda = lda_transformer.fit_transform(X_train_tfidf)
pickle.dump(X_train_lda, open('x_lda.pickle', 'wb'))
print "LDA shape", X_train_lda.shape

#create classifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
clf = SGDClassifier(loss='hinge', penalty='l2', n_iter=5, random_state=88)
reg = SGDRegressor(penalty='l2', n_iter=5, random_state=88)

#split dataset
X_train = X_train_tfidf[:-10000]
X_test = X_train_tfidf[-10000:]

#performs classification and gives output confusion matrix and words with largest weights
def classify(s):
	y_train = initial_data[s][:-10000]
	y_test = initial_data[s][-10000:]
	clf.fit(X_train, y_train)
	print clf.score(X_test, y_test)
	print confusion_matrix(y_test, clf.predict(X_test))
	print [count_vect.get_feature_names()[x] for x in np.argsort(clf.coef_[0])[:10]]
	print [count_vect.get_feature_names()[x] for x in np.argsort(clf.coef_[0])[-10:]]
	return clf

# def regress(s):
# 	y_train = map(float,initial_data[s][:-10000])
# 	y_test = map(float,initial_data[s][-10000:])
# 	reg.fit(X_train, y_train)
# 	print reg.score(X_test, y_test)
# 	return reg

#reformats education column to fit categories described in the paper
def fix_education(s):
	if 'college' in s and 'dropped' in s:
		return 'highschool'
	elif 'high' in s:
		return 'highschool'
	elif 'space' in s or s=='':
		return "highschool"
	elif 'college' in s:
		return 'college'
	elif 'dropped' in s:
		return 'college'
	elif 'law' in s or 'med' in s or 'masters' in s or 'ph.d' in s:
		return 'grad school'
	else:
		return s

classify('sex')
