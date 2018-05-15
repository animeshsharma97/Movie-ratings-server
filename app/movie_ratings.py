
# This file checks the performance of various models on our data and also checks the performance of that best model on Twitter data.

import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential , load_model
from keras.layers import Dense , Dropout , LSTM , Bidirectional
from keras.layers import Flatten , Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau

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
 
#adding words to vocabulary
def add_doc_to_vocab(text, vocab):
    tokens = clean_doc(text)
    vocab.update(tokens)
    
    
vocab = Counter()
df = pd.read_csv('train.tsv',delimiter='\t')
df = df.iloc[:60000,:]

X = df['Phrase']
y = df['Sentiment']
y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
del df,X,y

len1 = len(X_train)
for i in range(len1):
    text = X_train.iloc[i]
    add_doc_to_vocab(text , vocab)


print(len(vocab))
print(vocab.most_common(20))

min_occurance = 2
tokens = [k for k,c in vocab.items() if (c >= min_occurance & len(k) > 1)]
print(len(tokens))

def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
 
save_list(tokens, 'vocab.txt')

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

test_doc = []
len2 = len(X_test)
for i in range(len2):
    text = X_test.iloc[i]
    doc = clean_doc_load(text , vocab)
    test_doc.append(doc)

index_train = []
for i in range(len(train_doc)):
    if len(train_doc[i]) == 0 :
        index_train.append(i)
    
index_test = []
for i in range(len(test_doc)):
    if len(test_doc[i]) == 0 :
        index_test.append(i)
        
train_doc = pd.DataFrame(train_doc)
test_doc = pd.DataFrame(test_doc)
y_train , y_test = pd.DataFrame(y_train) , pd.DataFrame(y_test)

train_doc.drop(index_train , inplace = True)
test_doc.drop(index_test , inplace = True)
y_train.drop(index_train , inplace = True)
y_test.drop(index_test , inplace = True)

train_doc , test_doc = np.array(train_doc.iloc[:,0]) , np.array(test_doc.iloc[:,0])
y_train , y_test = y_train.values , y_test.values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_doc)

X_train = tokenizer.texts_to_matrix(train_doc, mode='binary')
X_test = tokenizer.texts_to_matrix(test_doc, mode='binary')
n_words = X_test.shape[1]

####################################################################################
# Naive Bayes Model

clf = MultinomialNB()
clf.fit(X_train,y_train)  #implement Multinomial NB algo

acc_nb = accuracy_score(y_test,clf.predict(X_test))
print('Test Accuracy of Naive Bayes : %f' % (acc_nb))  

####################################################################################
# LSTM Model

model = Sequential()
model.add(Bidirectional(LSTM(100 , activation='relu') ,input_shape=(None,n_words)))
model.add(Dropout(0.2))
model.add(Dense(units = 50,input_dim = 100, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    

model.fit(X_train.reshape((-1,1,n_words)), y_train, epochs = 20, batch_size = 100)

loss_rnn, acc_rnn = model.evaluate(X_test.reshape((-1,1,n_words)), y_test, verbose=0)

model.model.save('rnn.h5')  # saving model weights

model_rnn = load_model('rnn.h5')  # loading saved weights

new_doc = [input()]   # taking new inputs
new_doc = tokenizer.texts_to_matrix(new_doc)
pred = np.argmax(model_rnn.predict(new_doc.reshape((1,1,n_words))) )
print('\n{} stars'.format(pred+1))

####################################################################################

# predicting on twitter data
test_df = pd.read_csv('Tweets_test.csv')

tweet_doc = []
len_df = len(test_df['Tweet'])
for i in range(len_df):
    text = test_df['Tweet'].loc[i]
    doc = clean_doc_load(text , vocab)  # cleaning the tweets
    tweet_doc.append(doc)
    
tweet_doc = tokenizer.texts_to_matrix(tweet_doc)
pred = (model_rnn.predict(tweet_doc.reshape((-1,1,n_words))))

test_df['Rating'] = np.zeros((len(pred),1))
for i in range(len(pred)):
    test_df['Rating'].loc[i] = np.argmax(pred[i]) + 1  # saving predictions for each tweet
    










