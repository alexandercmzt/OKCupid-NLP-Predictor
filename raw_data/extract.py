import csv
import sys
import re
import string
import cPickle as pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
wnl = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
from collections import Counter
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
	#text = text.translate(None, string.punctuation)
	#text = text.translate(None, string.digits)
	return text

#gets all written text for a user at index i
def get_index_text(data, index):
	output_text = ""
	for j in xrange(6, 16):
		output_text += cleantext(data[index][j])
		output_text += ". "
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

def get_fscore(data):
	retvar = []
	for i in xrange(len(data)):
		print(str(i)+ '\r')
		text = get_index_text(data, i)
		text = tokenizer.tokenize(text)
		text = dict(Counter([x[1] for x in nltk.pos_tag(text)]))
		pos = sum([text[x] for x in text.keys() if x in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR', 'IN']])
		neg = sum([text[x] for x in text.keys() if x in ['PRP', 'PRP$', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'UH']])
		retvar.append(0.5*(pos - neg + 100))
	return retvar

def gender_preferential(data):
	retvar=[]
	for i in xrange(len(data)):
		print(str(i)+ '\r')
		text = get_index_text(data, i)
		text = tokenizer.tokenize(text)
		vec = [0.0 for _ in xrange(10)]
		for word in text:
			if word[-4:] == 'able': vec[0] +=1
			elif word[-2:] == 'al' : vec[1] +=1
			elif word[-3:] == 'ful' : vec[2] +=1
			elif word[-4:] == 'ible' : vec[3] +=1
			elif word[-2:] == 'ic' : vec[4] +=1
			elif word[-3:] == 'ive' : vec[5] +=1
			elif word[-4:] == 'less' : vec[6] +=1
			elif word[-2:] == 'ly' : vec[7] +=1
			elif word[-3:] == 'ous' : vec[8] +=1
			elif word.lower() == 'sorry' : vec[9] +=1
		if len(text) > 0:
			retvar.append([x/len(text) for x in vec])
		else:
			retvar.append(vec)
	return retvar


def factor_analysis(data):
	retvar=[]
	for i in xrange(len(data)):
		#print(str(i)+ '\r')
		text = get_index_text(data, i)
		text = tokenizer.tokenize(text)
		vec = [0 for _ in xrange(8)]
		conversation = ['know', 'people', 'think', 'person', 'tell', 'feel', 'friends', 'talk','new', 'talking', 'mean', 'ask', 'understand', 'feelings', 'care','thinking', 'friend', 'relationship', 'realize', 'question', 'answer','saying']
		home = ['woke', 'home', 'sleep', 'today', 'eat', 'tired', 'wake', 'watch','watched', 'dinner', 'ate', 'bed', 'day', 'house', 'tv', 'early', 'boring','yesterday', 'watching', 'sit']
		family = ['years', 'family', 'mother', 'children', 'father', 'kids', 'parents','old', 'year', 'child', 'son', 'married', 'sister', 'dad', 'brother','moved', 'age', 'young', 'months', 'three', 'wife', 'living', 'college','four', 'high', 'five', 'died', 'six', 'baby', 'boy', 'spend','Christmas']
		food_clothes = ['food', 'eating', 'weight', 'lunch', 'water', 'hair', 'life', 'white','wearing', 'color', 'ice', 'red', 'fat', 'body', 'black', 'clothes','hot', 'drink', 'wear', 'blue', 'minutes', 'shirt', 'green', 'coffee','total', 'store', 'shopping']
		romance = ['forget', 'forever', 'remember', 'gone', 'true', 'face', 'spent','times', 'love', 'cry', 'hurt', 'wish', 'loved']
		positive = ['absolutely', 'abundance', 'ace', 'active', 'admirable', 'adore', 'agree', 'amazing', 'appealing', 'attraction', 'bargain', 'beaming', 'beautiful', 'best', 'better', 'boost', 'breakthrough', 'breeze', 'brilliant', 'brimming', 'charming', 'clean', 'clear', 'colorful', 'compliment', 'confidence', 'cool', 'courteous', 'cuddly', 'dazzling', 'delicious', 'delightful', 'dynamic', 'easy', 'ecstatic', 'efficient', 'enhance', 'enjoy', 'enormous', 'excellent', 'exotic', 'expert', 'exquisite', 'flair', 'free', 'generous', 'genius', 'great', 'graceful', 'heavenly', 'ideal', 'immaculate', 'impressive', 'incredible', 'inspire', 'luxurious', 'outstanding', 'royal', 'speed', 'splendid', 'spectacular', 'superb', 'sweet', 'sure', 'supreme', 'terrific', 'treat', 'treasure', 'ultra', 'unbeatable', 'ultimate', 'unique', 'wow', 'zest']
		negative = ['wrong', 'stupid', 'bad', 'evil', 'dumb', 'foolish', 'grotesque', 'harm', 'fear', 'horrible', 'idiot', 'lame', 'mean', 'poor', 'heinous', 'hideous', 'deficient', 'petty', 'awful', 'hopeless', 'fool', 'risk', 'immoral', 'risky', 'spoil', 'spoiled', 'malign', 'vicious', 'wicked', 'fright', 'ugly', 'atrocious', 'moron', 'hate', 'spiteful', 'meager', 'malicious', 'lacking']
		emotional = ['aggressive', 'alienated', 'angry', 'annoyed', 'anxious', 'careful', 'cautious', 'confused', 'curious', 'depressed', 'determined', 'disappointed', 'discouraged', 'disgusted', 'ecstatic', 'embarrassed', 'enthusiastic', 'envious', 'excited', 'exhausted', 'frightened', 'frustrated', 'guilty', 'happy', 'helpless', 'hopeful', 'hostile', 'humiliated', 'hurt', 'hysterical', 'innocent', 'interested', 'jealous', 'lonely', 'mischievous', 'miserable', 'optimistic', 'paranoid', 'peaceful', 'proud', 'puzzled', 'regretful', 'relieved', 'sad', 'satisfied', 'shocked', 'shy', 'sorry', 'surprised', 'suspicious', 'thoughtful', 'undecided', 'withdrawn']
		vec = [0.0 for _ in xrange(8)]
		for word in text:
			if word in conversation: vec[0] +=1
			elif word in home: vec[1] +=1
			elif word in family: vec[2] +=1
			elif word in food_clothes: vec[3] +=1
			elif word in romance: vec[4] +=1
			elif word in positive: vec[5] +=1
			elif word in negative: vec[6] +=1
			elif word in emotional: vec[7] +=1
		if len(text) > 0:
			retvar.append([x/len(text) for x in vec])
		else:
			retvar.append(vec)
	return retvar



raw_data = open_csv()
column_dictionary = {}
column_dictionary['text'] = get_all_text(raw_data)
column_dictionary['education'] = get_column(raw_data, 5)
column_dictionary['sex'] = get_column(raw_data, 26)
column_dictionary['fscore'] = get_fscore(raw_data)
column_dictionary['gpref'] = gender_preferential(raw_data)
column_dictionary['factor'] = factor_analysis(raw_data)

# column_dictionary['age'] = get_column(raw_data, 0)
# column_dictionary['body'] = get_column(raw_data, 1)
# column_dictionary['diet'] = get_column(raw_data, 2)
# column_dictionary['drinks'] = get_column(raw_data, 3)
# column_dictionary['drugs'] = get_column(raw_data, 4)
# column_dictionary['income'] = get_column(raw_data, 18)
# column_dictionary['orientation'] = get_column(raw_data, 23)
# column_dictionary['pets'] = get_column(raw_data, 24)
# column_dictionary['religion'] = get_column(raw_data, 25)
# column_dictionary['smokes'] = get_column(raw_data, 28)

pickle.dump(column_dictionary, open("columns.pickle", 'wb'))


