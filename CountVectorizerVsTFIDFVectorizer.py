# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


normalizedTweets = pd.read_csv('normalizedDatas.txt',sep="\n",quotechar=None, quoting=3,header=None)
originalTweets = pd.read_csv('tweetsOriginal.csv')
normalizedTweets.columns = ["Tweetler"]
normalizedTweets["Duygu"]= originalTweets["Duygu"]

datas = normalizedTweets["Tweetler"]


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2)) 
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidf = TfidfVectorizer( min_df=0.0, max_df=1.0,ngram_range=(1,2))
  
X1 = tfidf.fit_transform(datas).toarray() #bagimsiz degiskenler tfidf icin
X = cv.fit_transform(datas).toarray() # bagimsiz degiskenler countvectorizer


y = normalizedTweets.iloc[:,1].values # bagimli degiskenler countvectorizer
y1=normalizedTweets.iloc[:,1].values # bagimli degiskenler tfidf



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)  # traing verileriyle bir model olusturuldu
gnb1 = GaussianNB()
gnb1.fit(X1_train,y1_train)  # traing verileriyle bir model olusturuldu

y_pred = gnb.predict(X_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(CountVectorizer)
y1_pred = gnb1.predict(X1_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(TFIDF)

from sklearn.metrics import classification_report,  accuracy_score

print("CountVectorizer ile : ")  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))  

print("TFIDF ile : ")
print(classification_report(y1_test,y1_pred))  
print(accuracy_score(y1_test, y1_pred))


