import cPickle as pickle
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from collections import defaultdict
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


initial_data = pickle.load(open('columns.pickle', 'rb'))
texts = initial_data['text']
frequency = defaultdict(int)
print frequency
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]
dictionary = corpora.Dictionary(texts)
dictionary.save('text_indices.dict')
print(dictionary)
print(dictionary.token2id)
print("Dictionary saved. Press enter to continue.")
raw_input()
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('text_serialized.mm', corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=500)
corpus_lda = model[corpus_tfidf]

