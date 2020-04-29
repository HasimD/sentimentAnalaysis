# -*- coding: utf-8 -*-

import pandas as pd
import re
import numpy as np

normalizedTweets = pd.read_csv('normalizedDatas.txt',sep="\n",quotechar=None, quoting=3,header=None)
originalTweets = pd.read_csv('tweetsOriginal.csv')
normalizedTweets.columns = ["Tweetler"]
normalizedTweets["Duygu"]= originalTweets["Duygu"]

newFeatureTweets = normalizedTweets.copy(deep=True)

pivots=""
# ---------------------- BASLAMA NOKTASI:  YENI OZNITELIK(EMOTICONS) EKLEME ----------------------
for i in range(len(originalTweets)):
    pivots = re.findall('[:.]\)|[:.][dD]|[:.][pP]|[Xx][dD]|:-\)|=\)|;\)',originalTweets['Tweetler'][i]) #turkce kararkter harici herseyi sil
    num=0;
    if pivots:
        for j in pivots:
            newFeatureTweets['Tweetler'][i] = newFeatureTweets['Tweetler'][i] + " posEmos"
    
    pivots = re.findall('[:.]\(|[:.][sS]|:-\(|=\(|;\(',originalTweets['Tweetler'][i]) #turkce kararkter harici herseyi sil
    if pivots:
        for j in pivots:
            newFeatureTweets['Tweetler'][i] = newFeatureTweets['Tweetler'][i] + " negEmos"
                

# ---------------------- BITIS NOKTASI: YENI OZNITELIK(EMOTICONS) EKLEME ----------------------
            
            
            
#---------------------- BASLANGIC NOKTASI: TFIDF ile TFIDF+EMOTICONS,Naıve Bayes yontemlerini deneyelim ----------------------            
            
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidf = TfidfVectorizer( min_df=0.0, max_df=1.0,ngram_range=(1,2))
X = tfidf.fit_transform(normalizedTweets['Tweetler']).toarray() #bagimsiz degiskenler tfidf icin
y=normalizedTweets.iloc[:,1].values # bagimli degiskenler tfidf

tfidfNormalizedFeatures = tfidf.get_feature_names()# oznitelik boyutu gormek icin

tfidf = TfidfVectorizer( min_df=0.0, max_df=1.0,ngram_range=(1,2))
X1 = tfidf.fit_transform(newFeatureTweets['Tweetler']).toarray() #bagimsiz degiskenler tfidf+emoticons icin
y1=newFeatureTweets.iloc[:,1].values # bagimli degiskenler tfidf+emoticons


tfidfNewFeatures = tfidf.get_feature_names()# oznitelik boyutu gormek icin

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # tfidf feature
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0) # tfidf + emoticons feature

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)  # tfidf model olusturuldu

gnb1 = GaussianNB()
gnb1.fit(X1_train,y1_train)  # tfidf+emoticons model olusturuldu


y_pred = gnb.predict(X_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(TFIDF)
y1_pred = gnb1.predict(X1_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(TFIDF+emoticons)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#---------------------- SONUCLAR ----------------------
print("\nNAIVE BAYES : \nTFIDF ile : \n")
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

print("\nTFIDF+EMOTICONS ile : \n")
print(classification_report(y1_test,y1_pred))  
print(accuracy_score(y1_test, y1_pred))

print("\n-------------------------------------------------------------------------------------\n")

#---------------------- BITIS NOKTASI  ----------------------


#---------------------- BASLANGIC NOKTASI: AYNI SEYLERI COUNTVECTORIZER ILE DENEYELIM ----------------------
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2)) 
X = cv.fit_transform(normalizedTweets['Tweetler']).toarray() #bagimsiz degiskenler cv icin
y=normalizedTweets.iloc[:,1].values # bagimli degiskenler cv

cvNormalizedFeatures = cv.get_feature_names() # oznitelik boyutu gormek icin

cv = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2)) 
X1 = cv.fit_transform(newFeatureTweets['Tweetler']).toarray() #bagimsiz degiskenler tfidf+emoticons icin
y1=newFeatureTweets.iloc[:,1].values # bagimli degiskenler tfidf+emoticons

cvNewFeatures = cv.get_feature_names()# oznitelik boyutu gormek icin

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # cv feature
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0) # cv + emoticons feature

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)  # cv model olusturuldu

gnb1 = GaussianNB()
gnb1.fit(X1_train,y1_train)  # cv+emoticons model olusturuldu


y_pred = gnb.predict(X_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(cv)
y1_pred = gnb1.predict(X1_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(cv+emoticons)

from sklearn.metrics import classification_report, accuracy_score

#---------------------- SONUCLAR ----------------------
print("\nNAIVE BAYES : \nCountVectorizer ile : \n")
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

print("\nCountVectorizer+EMOTICONS ile : \n")
print(classification_report(y1_test,y1_pred))  
print(accuracy_score(y1_test, y1_pred))


#---------------------- BITIS NOKTASI  ----------------------

newFeatureTweets.to_csv("lastDatas.csv") # son hali diger classifierler ile denemek uzere dosyaya atildi


 