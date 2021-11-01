# Import  libraries
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re, string, unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
import keras
from keras.models import Sequential
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Read the files
# use python as the engine to avoid any encoding errors
train_dataset, test_dataset = pd.read_csv('Corona_NLP_train.csv',
                                          engine='python' ), \
                                          pd.read_csv('Corona_NLP_test.csv',
                                          engine='python')

# Display them
from IPython.display import display

display(train_dataset.head(), train_dataset.shape,
        test_dataset.head(), test_dataset.shape)

data = pd.concat([train_dataset, test_dataset], axis=0)
data.head()

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

data.Sentiment[data.Sentiment == 'Extremely Positive'] = 'Positive'
data.Sentiment[data.Sentiment == 'Extremely Negative'] = 'Negative'

nltk.download('stopwords')
stop_word = stopwords.words('english')

# our clean function

def clean(text):
  # remove urls
  text = re.sub(r'http\S+', " ", text)

  # remove mentions
  text = re.sub(r'@\w+', ' ', text)

  # remove hashtags
  text = re.sub(r'#\w+', ' ', text)

  # remove digits
  text = re.sub(r'\d+', ' ', text)

  # remove html tags
  text = re.sub(r'<.*?>', ' ', text)

  # remove stop words
  text = text.split()
  text = " ".join([word for word in text if not word in stop_word])

  return text

data['OriginalTweet'] = data['OriginalTweet'].apply(lambda x: clean(x))
useful_cols = ['OriginalTweet', 'Sentiment']

data = data[useful_cols]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data.Sentiment = encoder.fit_transform(data.Sentiment)

from sklearn.model_selection import train_test_split

X = data.OriginalTweet.copy()
y = data.Sentiment.copy()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    train_size=.9,
                                                    )

max_len = np.max(X_train.apply(lambda x :len(x)))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)
vocab_length = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

saved_model = tf.keras.models.load_model('CoronaNlpModel.hdf5')

import streamlit as st
st.title('Covid19 Tweets Classifier')
st.subheader('A Natural Language Processing Project')

st.text('Enter the text you would like to classify.')
text = st.text_input('Enter text') #text is stored in this variable

text_series = pd.Series(text).apply(lambda x: clean(x))
text_series

token_text = tokenizer.texts_to_sequences(text_series)
pad_text = pad_sequences(token_text, maxlen=286, padding='post')

predict_text = saved_model.predict(pad_text)
classes_text = np.argmax(predict_text,axis=1)


if classes_text[0] == 1:
  st.write('Neutral')
elif classes_text[0] == 2:
  st.write('Positive')
else:
  st.write('Negative')
