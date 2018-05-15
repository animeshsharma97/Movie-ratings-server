
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import load_model


stopwords = stopwords.words('english')
newStopWords = ['',' ','  ','   ','    ',' s']
stopwords.extend(newStopWords)
stop_words = set(stopwords)

def clean_doc(doc):
    tokens = word_tokenize(doc)
    tokens = [re.sub('[^a-zA-Z]',' ', word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

    
df = pd.read_csv('train.tsv',delimiter='\t')
df = df.iloc[:60000,:]

X = df['Phrase']
y = df['Sentiment']
y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
del df,X,y

len1 = len(X_train)

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

def clean_doc_load(text, vocab):
    tokens = text.split()
    tokens = [re.sub('[^a-zA-Z]',' ', word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

train_doc = []
for i in range(len1):
    text = X_train.iloc[i]
    doc = clean_doc_load(text , vocab)
    train_doc.append(doc)

index_train = []
for i in range(len(train_doc)):
    if len(train_doc[i]) == 0 :
        index_train.append(i)
           
train_doc = pd.DataFrame(train_doc)
train_doc.drop(index_train , inplace = True)

train_doc = np.array(train_doc.iloc[:,0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_doc)
X_train = tokenizer.texts_to_matrix(train_doc, mode='binary')
n_words = X_train.shape[1]

model_rnn = load_model('rnn.h5')










