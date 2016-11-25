import csv
import sys
import re
import string
import cPickle as pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
wnl = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
#open csv as list of lists
def open_csv():
	with open('profiles.csv', 'rU') as f:
	    reader = csv.reader(f)
	    data = list(list(rec) for rec in csv.reader(f, delimiter=','))
	    f.close()
	    return data[1:]

#removes html tags
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

#cleans up punctuation and newlines from text
def cleantext(text):
	text = cleanhtml(text)
	text = text.replace('\n', ' ')
	text = text.replace('\t', ' ')
	text = text.replace('\r', ' ')
	text = text.translate(None, string.punctuation)
	text = text.translate(None, string.digits)
	return text

#gets all written text for a user at index i
def get_index_text(data, index):
	output_text = ""
	for j in xrange(6, 16):
		output_text += cleantext(data[index][j])
		output_text += " "
	return output_text.strip()

#gets all formatted text for all users
def get_all_text(data):
	retvar = []
	for i in xrange(len(data)):
		print(str(i)+ '\r')
		text = get_index_text(data, i)
		retvar.append([wnl.lemmatize(word) for word in tokenizer.tokenize(text) if word not in stopwords.words('english')])
	return retvar

#gets a column from the data
def get_column(data, i):
	retvar = []
	for j in xrange(len(data)):
		retvar.append(data[j][i])
	return retvar

raw_data = open_csv()
column_dictionary = {}
column_dictionary['text'] = get_all_text(raw_data)
column_dictionary['age'] = get_column(raw_data, 0)
column_dictionary['body'] = get_column(raw_data, 1)
column_dictionary['diet'] = get_column(raw_data, 2)
column_dictionary['drinks'] = get_column(raw_data, 3)
column_dictionary['drugs'] = get_column(raw_data, 4)
column_dictionary['education'] = get_column(raw_data, 5)
column_dictionary['income'] = get_column(raw_data, 18)
column_dictionary['orientation'] = get_column(raw_data, 23)
column_dictionary['pets'] = get_column(raw_data, 24)
column_dictionary['religion'] = get_column(raw_data, 25)
column_dictionary['sex'] = get_column(raw_data, 26)
column_dictionary['smokes'] = get_column(raw_data, 28)

pickle.dump(column_dictionary, open("columns.pickle", 'wb'))


