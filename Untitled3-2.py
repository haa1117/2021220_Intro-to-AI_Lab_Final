#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
import numpy as np
import re


# In[9]:


df = pd.read_csv('chatgpt1.csv')


# In[10]:


def preprocess_tweet(tweet):
    if isinstance(tweet, str):
        # Convert to lowercase
        tweet = tweet.lower()
        # Remove special characters and punctuation
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # Remove numbers
        tweet = re.sub(r'\d+', '', tweet)
        # Remove whitespace
        tweet = tweet.strip()
    return tweet


# In[14]:


df['processed_text'] = df['Text'].apply(preprocess_tweet)


# In[15]:


df['processed_text'] = df['processed_text'].fillna('')


# In[16]:


X_user = df[['Username', 'Tweet Id', 'LikeCount']]  # Modify this based on your desired features


# In[17]:


y_user = df['Username']  # Assuming username as the target variable


# In[18]:


X_user_train, X_user_test, y_user_train, y_user_test = train_test_split(X_user, y_user, test_size=0.2, random_state=42)
X_user_train = pd.get_dummies(X_user)


# In[19]:


knn_user = KNeighborsClassifier()
knn_user.fit(X_user_train, y_user_train)


# In[ ]:


y_user_pred = knn_user.predict(X_user_test)


# In[ ]:


user_classification_report = classification_report(y_user_test, y_user_pred)
print("User Classification Report:")
print(user_classification_report)


# In[ ]:


vectorizer = TfidfVectorizer(max_features=2000)  # Reduce the number of features
X_text = vectorizer.fit_transform(df['processed_text'])


# In[20]:


pca = TruncatedSVD(n_components=1000)
X_text_reduced = pca.fit_transform(X_text)


# In[21]:


X_text_train, X_text_test, y_user_train, y_user_test = train_test_split(X_text_reduced, y_user, test_size=0.2, random_state=42)


# In[22]:


knn_text = KNeighborsClassifier()
knn_text.fit(X_text_train, y_user_train)


# In[23]:


y_text_pred = knn_text.predict(X_text_test)


# In[24]:


text_classification_report = classification_report(y_user_test, y_text_pred)
print("Text Classification Report:")
print(text_classification_report)


# In[25]:


X_combined = hstack([X_user_test.values, X_text_test])


# In[26]:


knn_combined = KNeighborsClassifier()
knn_combined.fit(X_combined, y_user_test)


# In[27]:


y_combined_pred = knn_combined.predict(X_combined)


# In[28]:


combined_classification_report = classification_report(y_user_test, y_combined_pred)
print("Combined Classification Report:")
print(combined_classification_report)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(data[['Text', 'Hashtags']], data['User ID'], test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)
   # Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance
print(classification_report(y_test,y_pree))


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(data['Tweet text'], data['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
   # Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test_vec)

# Evaluate the performance
print(classification_report(y_test,y_pred)


# In[ ]:




