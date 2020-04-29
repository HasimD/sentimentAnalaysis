
# -*- coding: utf-8 -*-

import pandas as pd
import glob



#--------------------ON ISLEME KISMI ----------------------------------------

files = glob.glob("datasetsforSA/pos/*.txt")
df = pd.concat([pd.read_csv(fp,encoding="windows-1254",sep="\n",quotechar=None, quoting=3,header=None)for fp in files], ignore_index=True)
df.columns = ["Tweetler"]
df["Duygu"] = 1                    #pozitif kelimeleri 1 ile kodladık

files = glob.glob("datasetsforSA/neg/*.txt")
df2 = pd.concat([pd.read_csv(fp,encoding="windows-1254",sep="\n",quotechar=None, quoting=3,header=None)for fp in files], ignore_index=True)
df2.columns = ["Tweetler"]
df2["Duygu"] = 0                    #negatif kelimeleri 0 ile kodladık

Tweets = pd.concat([df2, df], ignore_index=True)    # negatif ve pozitif tum cumleleri bir dataFrame'de topladik
Tweets = Tweets.sample(frac=1).reset_index(drop=True)   # tweetleri randomize ettik


import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

trStopWords = stopwords.words('turkish') # turkce stopword'ler
nrmDatas= Tweets.copy()
nrmDatas.to_csv("tweetsOriginal.csv") #sonradan emoji ozniteligi icin

    for i in range(len(Tweets)): # NOT : islem uzun surebilir
    pivot = re.sub('[^a-zA-ZÇŞĞÜÖİçşğüöı]',' ',Tweets['Tweetler'][i]) #turkce kararkter harici herseyi sil
    pivot = re.sub(r'\b\w{1,1}\b', '',pivot) #tek harfleri sil
    pivot = pivot.lower() # tum harfleri kucult
    pivot = pivot.split() # kelimelere ayir
    pivot = [(word) for word in pivot if not word in set(trStopWords)] # stopword'leri sil
    pivot = ' '.join(pivot) # tekrar birlestir
    nrmDatas['Tweetler'][i]=pivot



nrmDatas["Tweetler"].to_csv("normalizingDatas.txt")


#--------------------ON ISLEME KISMI ----------------------------------------

