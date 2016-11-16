import cPickle as pickle
from gensim import corpora
from nltk.corpus import stopwords
from collections import defaultdict

initial_data = pickle.load(open('columns.pickle', 'rb'))
texts = initial_data['text']
frequency = defaultdict(int)
texts = [[token for token in text if frequency[token] > 1] for text in texts]
for text in texts:
    for token in text:
        frequency[token] += 1
dictionary = corpora.Dictionary(texts)
dictionary.save('text_indices.dict')
print(dictionary)
print(dictionary.token2id)
print("Dictionary saved. Press enter to continue.")
raw_input()
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('text_serialized.mm', corpus)

