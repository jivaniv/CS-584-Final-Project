#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import sklearn

import seaborn as sns

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[2]:


#first I will begin by pre-training an LSTM model on a labeled Twitter dataset

#loading and prepping data
dataset_columns = ['target','ids','date','flag','user','text']
df = pd.read_csv('/Users/zenoviyivaniv/Downloads/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1', names = dataset_columns)
data = df.drop(['ids','date','flag','user'],axis=1) #only leave labels and tweets
data.drop(data[(data['target']==2)].index,inplace=True) #drop neutral labels
data = data.sample(frac=1) #need to mix data because it's ordered by label
data = data[0:10000] #going to use 10000 data points for training/testing
data['target'] = data['target'].replace(4,1) #positive labels (4) need to be replaced with 1
data.head()

X = data['text']
y = data['target']

y = data[['target']].to_numpy() #need to make y data same format as X data after X data gets pre-processed
data.head()


# In[3]:


data.isnull().values.any() #check if there are any blank rows


# In[4]:


sns.countplot(data=data,x='target'); #check if data is skewed in any one direction


# In[5]:


#data pre-processing

cachedStopWords = stopwords.words("english") #to get rid of stopwords quickly
list1 = [] #initialize array that will hold text data

for i in range(len(data)):
    text = data.iloc[i,1] #for every tweet
    text = text.lower() #lowercase
    text = re.sub('@[^\s]+','', text) #remove mentions
    text = re.sub('http\S+','',text) #remove links
    text = re.sub('\s+',' ',text) #remove extra whitespaces
    text = re.sub('[^\w\s]', '', text) #remove special characters
    text = re.sub('[0-9]', '', text) #remove numbers
    text = re.sub('\b(\w+)(?:\W+\1\b)+', '\1', text) #remove duplicate words
    text = re.sub("(.)\\1{2,}", "\\1",text) #remove letters that repeat more than 2 times
    text = word_tokenize(text) #split tweets into individual words
    text = [word for word in text if not word in cachedStopWords] #remove stopwords
    list1.append(text) #final list of words for the tweet
    
list1


# In[6]:


#even though I did my own pre-processing, I used the tokenizer function to map the words to numbers
#chose 3000 as the number of words in my dictionary

tk = Tokenizer(num_words=3000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")

tk.fit_on_texts(list1)

X_seq = tk.texts_to_sequences(list1)
X_seq


# In[7]:


#check max and min number of words in a tweet for padding

first = len(X_seq[0])
for i in range(len(X_seq)-1):
    second = len(X_seq[i+1])
    if second > first:
        maximum = second
        first = second
    else:
        maximum = first
        
print(maximum)

first = len(X_seq[0])
for i in range(len(X_seq)-1):
    second = len(X_seq[i+1])
    if second < first:
        minimum = second
        first = second
    else:
        minimum = first
        
print(minimum)


# In[8]:


#I chose the length of the tweets based on the above and used the padding function to fill in the rest

length = 20
X_seq_padded = pad_sequences(X_seq, maxlen=length)
X_seq_padded


# In[9]:


#split the data into train and test

X_train,X_test,y_train,y_test = train_test_split(X_seq_padded,y,test_size = 0.20)


# In[10]:


X_train.shape


# In[11]:


y_train.shape


# In[12]:


#LSTM model using keras

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

model = Sequential()

model.add(Embedding(3000,16,input_length=X_train.shape[1])) #input layer

model.add(SpatialDropout1D(0.6)) 

model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5)) #LSTM layer

model.add(Dropout(0.4))

model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# In[13]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[14]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=8)


# In[15]:


scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])


# In[16]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[17]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[18]:


y_prediction = model.predict(X_test)

y_pred = []

for i in y_prediction:
    if i >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

cmatrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

cmatrixplot = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = [False, True])

cmatrixplot.plot()
plt.show()


# In[19]:


#Now I will use the former model to evaluate the Ukraine Twitter data
#I will do three different tests on Ukraine Twitter data from three different months: April 2022, September 2022, and March 2023

#April 2022
import pickle
with open('/Users/zenoviyivaniv/Downloads/0410_UkraineCombinedTweetsDeduped.pkl','rb') as f:
    uaDataApril = pickle.load(f)

uaDataApril.to_csv(r'/Users/zenoviyivaniv/Downloads/UkraineApril.csv',index=None)

uaDataApril = uaDataApril.drop(['location','tweetid','tweetcreatedts','retweetcount','hashtags'],axis=1)
uaDataApril = uaDataApril.sample(frac=1)
uaDataApril = uaDataApril[0:10000]
uaDataApril.head()



# In[20]:


uaDataApril.isnull().values.any()


# In[21]:


cachedStopWords = stopwords.words("english")
list2 = []

for i in range(len(uaDataApril)):
    text = uaDataApril.iloc[i,0]
    text = text.lower()
    text = re.sub('@[^\s]+','', text)
    text = re.sub('http\S+','',text)
    text = re.sub('\s+',' ',text)
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub('\b(\w+)(?:\W+\1\b)+', '\1', text)
    text = re.sub("(.)\\1{2,}", "\\1",text)
    text = word_tokenize(text)
    text = [word for word in text if not word in cachedStopWords]
    list2.append(text)


# In[22]:


tk = Tokenizer(num_words=3000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")

tk.fit_on_texts(list2)

X_seq2 = tk.texts_to_sequences(list2)


# In[23]:


first = len(X_seq2[0])
for i in range(len(X_seq2)-1):
    second = len(X_seq2[i+1])
    if second > first:
        maximum = second
        first = second
    else:
        maximum = first
        
print(maximum)

first = len(X_seq2[0])
for i in range(len(X_seq2)-1):
    second = len(X_seq2[i+1])
    if second < first:
        minimum = second
        first = second
    else:
        minimum = first
        
print(minimum)


# In[24]:


length = 40
X_seq2_padded = pad_sequences(X_seq2, maxlen=length)
X_seq2_padded


# In[25]:


prediction = model.predict(X_seq2_padded)
plabels_1 = []
plabels_0 = []
plabels = []
for i in prediction:
    if i >= 0.5:
        plabels_1.append(1)
        plabels.append(1)
    else:
        plabels_0.append(0)
        plabels.append(0)


# In[26]:


print(len(plabels_1))
print(len(plabels_0))


# In[27]:


column = uaDataApril['text']
for i in range(3):
    print(uaDataApril['text'].iloc[i])
    if plabels[i] == 1:
        sent = 'Pos'
    else:
        sent = 'Neg'
    print("Sentiment : ", sent)


# In[28]:


#September 2022

with open('/Users/zenoviyivaniv/Downloads/0910_UkraineCombinedTweetsDeduped.pkl','rb') as f:
    uaDataSept = pickle.load(f)

uaDataSept.to_csv(r'/Users/zenoviyivaniv/Downloads/UkraineSept.csv',index=None)

uaDataSept = uaDataSept.drop(['location','tweetid','tweetcreatedts','retweetcount','hashtags'],axis=1)
uaDataSept = uaDataSept.sample(frac=1)
uaDataSept = uaDataSept[0:10000]
uaDataSept.head()


# In[29]:


uaDataSept.isnull().values.any()


# In[30]:


cachedStopWords = stopwords.words("english")
list3 = []

for i in range(len(uaDataSept)):
    text = uaDataSept.iloc[i,0]
    text = text.lower()
    text = re.sub('@[^\s]+','', text)
    text = re.sub('http\S+','',text)
    text = re.sub('\s+',' ',text)
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub('\b(\w+)(?:\W+\1\b)+', '\1', text)
    text = re.sub("(.)\\1{2,}", "\\1",text)
    text = word_tokenize(text)
    text = [word for word in text if not word in cachedStopWords]
    list3.append(text)


# In[31]:


tk = Tokenizer(num_words=3000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")

tk.fit_on_texts(list3)

X_seq3 = tk.texts_to_sequences(list3)


# In[32]:


first = len(X_seq3[0])
for i in range(len(X_seq3)-1):
    second = len(X_seq3[i+1])
    if second > first:
        maximum = second
        first = second
    else:
        maximum = first
        
print(maximum)

first = len(X_seq3[0])
for i in range(len(X_seq3)-1):
    second = len(X_seq3[i+1])
    if second < first:
        minimum = second
        first = second
    else:
        minimum = first
        
print(minimum)


# In[33]:


length = 40
X_seq3_padded = pad_sequences(X_seq3, maxlen=length)


# In[34]:


prediction = model.predict(X_seq3_padded)
plabels_1 = []
plabels_0 = []
plabels = []
for i in prediction:
    if i >= 0.5:
        plabels_1.append(1)
        plabels.append(1)
    else:
        plabels_0.append(0)
        plabels.append(0)


# In[35]:


print(len(plabels_1))
print(len(plabels_0))


# In[36]:


column = uaDataSept['text']
for i in range(3):
    print(uaDataSept['text'].iloc[i])
    if plabels[i] == 1:
        sent = 'Pos'
    else:
        sent = 'Neg'
    print("Sentiment : ", sent)


# In[37]:


#March 2023

with open('/Users/zenoviyivaniv/Downloads/UkraineCombinedTweetsDeduped_MAR30.pkl','rb') as f:
    uaDataMarch = pickle.load(f)

uaDataMarch.to_csv(r'/Users/zenoviyivaniv/Downloads/UkraineMarch.csv',index=None)

uaDataMarch = uaDataMarch.drop(['location','tweetid','tweetcreatedts','retweetcount','hashtags'],axis=1)
uaDataMarch = uaDataMarch.sample(frac=1)
uaDataMarch = uaDataMarch[0:10000]
uaDataMarch.head()


# In[38]:


uaDataSept.isnull().values.any()


# In[39]:


cachedStopWords = stopwords.words("english")
list4 = []

for i in range(len(uaDataMarch)):
    text = uaDataMarch.iloc[i,0]
    text = text.lower()
    text = re.sub('@[^\s]+','', text)
    text = re.sub('http\S+','',text)
    text = re.sub('\s+',' ',text)
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub('\b(\w+)(?:\W+\1\b)+', '\1', text)
    text = re.sub("(.)\\1{2,}", "\\1",text)
    text = word_tokenize(text)
    text = [word for word in text if not word in cachedStopWords]
    list4.append(text)


# In[40]:


tk = Tokenizer(num_words=3000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")

tk.fit_on_texts(list4)

X_seq4 = tk.texts_to_sequences(list4)


# In[41]:


first = len(X_seq4[0])
for i in range(len(X_seq4)-1):
    second = len(X_seq4[i+1])
    if second > first:
        maximum = second
        first = second
    else:
        maximum = first
        
print(maximum)

first = len(X_seq4[0])
for i in range(len(X_seq4)-1):
    second = len(X_seq4[i+1])
    if second < first:
        minimum = second
        first = second
    else:
        minimum = first
        
print(minimum)


# In[44]:


length = 20
X_seq4_padded = pad_sequences(X_seq4, maxlen=length)


# In[45]:


prediction = model.predict(X_seq4_padded)
plabels_1 = []
plabels_0 = []
pred_labels = []
for i in prediction:
    if i >= 0.5:
        plabels_1.append(1)
        plabels.append(1)
    else:
        plabels_0.append(0)
        plabels.append(0)


# In[46]:


print(len(plabels_1))
print(len(plabels_0))


# In[49]:


column = uaDataMarch['text']
for i in range(3):
    print(uaDataMarch['text'].iloc[i])
    if plabels[i] == 1:
        sent = 'Pos'
    else:
        sent = 'Neg'
    print("Sentiment : ", sent)


# In[50]:


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


# In[51]:


dataNew = data 

dataNew['compound'] = dataNew['text'].apply(lambda text: sid.polarity_scores(text)['compound'])
dataNew['newTarget'] = dataNew['compound'].apply(lambda compound: 1 if compound >=0 else 0)

dataNew.head(20)


# In[52]:


cmatrix = sklearn.metrics.confusion_matrix(dataNew['target'], dataNew['newTarget'])

cmatrixplot = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = [False, True])

cmatrixplot.plot()
plt.show()


# In[53]:


print('Accuracy:', sklearn.metrics.accuracy_score(dataNew['target'], dataNew['newTarget']))


# In[54]:


dataNew2 = uaDataApril

dataNew2['compound'] = dataNew2['text'].apply(lambda text: sid.polarity_scores(text)['compound'])
dataNew2['newTarget'] = dataNew2['compound'].apply(lambda compound: 1 if compound >=0 else 0)

dataNew2.head()


# In[55]:


print('Positive sentiment:', sum(dataNew2['newTarget']))
print('Negative sentiment:', 10000-sum(dataNew2['newTarget']))


# In[56]:


dataNew3 = uaDataSept

dataNew3['compound'] = dataNew3['text'].apply(lambda text: sid.polarity_scores(text)['compound'])
dataNew3['newTarget'] = dataNew3['compound'].apply(lambda compound: 1 if compound >=0 else 0)

dataNew3.head()


# In[57]:


print('Positive sentiment:', sum(dataNew3['newTarget']))
print('Negative sentiment:', 10000-sum(dataNew3['newTarget']))


# In[58]:


dataNew4 = uaDataMarch

dataNew4['compound'] = dataNew4['text'].apply(lambda text: sid.polarity_scores(text)['compound'])
dataNew4['newTarget'] = dataNew4['compound'].apply(lambda compound: 1 if compound >=0 else 0)

dataNew4.head()


# In[59]:


print('Positive sentiment:', sum(dataNew4['newTarget']))
print('Negative sentiment:', 10000-sum(dataNew4['newTarget']))

