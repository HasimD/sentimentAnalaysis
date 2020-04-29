# -*- coding: utf-8 -*-


import pandas as pd



tweetsWithEmoticons = pd.read_csv('lastDatas.csv') # tfidf + emoticons
tweets = pd.read_csv('normalizedDatas.txt',sep="\n",quotechar=None, quoting=3,header=None) # sadece tfidf
tweets.columns = ["Tweetler"]
tweets["Duygu"]= tweetsWithEmoticons["Duygu"]


#---------------------- TFIDF ile TFIDF+EMOTICONS,LogisticRegression yontemlerini deneyelim ----------------------


from sklearn.feature_extraction.text import TfidfVectorizer  
tfidf = TfidfVectorizer( min_df=0.0, max_df=1.0,ngram_range=(1,2))


X = tfidf.fit_transform(tweets['Tweetler']).toarray() #bagimsiz degiskenler tfidf icin
y=tweets.iloc[:,1].values # bagimli degiskenler tfidf

X1 = tfidf.fit_transform(tweetsWithEmoticons['Tweetler']).toarray() #bagimsiz degiskenler tfidf+emoticons icin
y1=tweetsWithEmoticons.iloc[:,2].values # bagimli degiskenler tfidf+emoticons

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # tfidf feature
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0) # tfidf + emoticons feature


from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(TFIDF)

lr1= LogisticRegression()
lr1.fit(X1_train, y1_train)
y1_pred = lr1.predict(X1_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(TFIDF+Emoticons)

from sklearn.metrics import classification_report,  accuracy_score

#---------------------- SONUCLAR ----------------------
print("\nLOGISTICREGRESSION : \nTFIDF ile : \n")
print(classification_report(y_test,y_pred))
print("Accuracy Score : ")  
print(accuracy_score(y_test, y_pred))

print("\nTFIDF+EMOTICONS ile : \n")
print(classification_report(y1_test,y1_pred))  
print("Accuracy Score : ")
print(accuracy_score(y1_test, y1_pred))



print("\n-------------------------------------------------------------------------------------\n")



#---------------------- AYNI SEYLERI COUNTVECTORIZER ILE DENEYELIM ----------------------


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2)) #bag of words


X = cv.fit_transform(tweets['Tweetler']).toarray() #bagimsiz degiskenler cv icin
y=tweets.iloc[:,1].values # bagimli degiskenler cv

X1 = cv.fit_transform(tweetsWithEmoticons['Tweetler']).toarray() #bagimsiz degiskenler cv+emoticons icin
y1=tweetsWithEmoticons.iloc[:,2].values # bagimli degiskenler cv+emoticons

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # cv feature
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0) # cv + emoticons feature


from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(cv)

lr1= LogisticRegression()
lr1.fit(X1_train, y1_train)
y1_pred = lr1.predict(X1_test) # olusturulan modele test verileri verilerek bir tahmin yapildi(cv+emoticons)

from sklearn.metrics import classification_report,  accuracy_score

#---------------------- SONUCLAR ----------------------
print("\nLOGISTICREGRESSION : \nCountVectorizer ile : \n")
print(classification_report(y_test,y_pred))  
print("Accuracy Score : ")
print(accuracy_score(y_test, y_pred))

print("\nCountVectorizer+EMOTICONS ile : \n")
print(classification_report(y1_test,y1_pred))  
print("Accuracy Score : ")
print(accuracy_score(y1_test, y1_pred))
