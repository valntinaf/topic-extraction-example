import sys
if len(sys.argv) < 2:
    print("Usage help: ")
    print("python topic_extraction.py 'sentence_to_parse' ")
    sys.exit()

import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from collections import OrderedDict, Counter
import math
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.dependencygraph import DependencyGraph
import sys
import nltk


nltk.download('wordnet')
parser = CoreNLPDependencyParser(url='http://localhost:9000')


sentence = sys.argv[1]

dictionary = OrderedDict()
cnt = Counter()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def count(lem_word):
    cnt[lem_word] += 1

# Reading times_500.csv from 'data' folder
stemmer = SnowballStemmer(language='spanish')
data = pd.read_csv('data/times_500.csv', error_bad_lines=False)

result = []
for index, row in data.iterrows():
     # Adding preprocessedd word to array for counting
     result.append(preprocess(row['description']))

# Counting words in each row
for row in result:
    [count(x) for x in row]

print("Conteo de palabras para recorte de diccionario")
print(cnt)

# Getting TF-IDF value per word
tf_idfs = {}

parse, = parser.raw_parse(sentence)
conll = parse.to_conll(4)
dg = DependencyGraph(conll)
print(conll)

for word in sentence.split(' '):
    preprocessed_word = preprocess(word)
    if preprocessed_word:
        dfi = cnt[preprocessed_word[0]]
        if dfi:
            tf_idf = 1 * math.log( 500 / dfi)
        else:
            # 0.01 in order to avoid ZeroDivisionException
            tf_idf = 1 * math.log( 500 / 0.01)
        tf_idfs[word] = tf_idf

# Showing TF-IDF values
sorted_tf_idfs = {k: v for k, v in sorted(tf_idfs.items(), key=lambda item: item[1])}
print("Indices TF-IDFs")
print(sorted_tf_idfs)
