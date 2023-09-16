import pandas as pd
import numpy as np


# Data Import **
data = pd.read_csv("data/IMDB-Dataset.csv")


# Bag Of Word | Kelime Dağarcığı ** 
"""
# Kütüphanesiz kendi ellerimizle yazdık

# Preprocessing **
observations = ""
from bs4 import BeautifulSoup
def removeHTML(text):
    soup =  BeautifulSoup(text, "html.parser")
    return soup.get_text()

for n in range(20):
    observations += removeHTML(data.iloc[n, 0])

observationsClean = ",".join(i.lower() for i in observations.split() if i.isalnum())

setOfWords = set(observationsClean.split(","))


from tqdm import tqdm # döngünün ne kadar süreceğini gösteren çubuk

dictList = []
for n in tqdm(range(len(data))):
    observation = removeHTML(data.iloc[n, 0])
    clean = ",".join(i.lower() for i in observations.split() if i.isalnum())
    dictOfWords = dict.fromkeys(setOfWords, 0)
    for word in clean.split(","):
        if word in dictOfWords:
            dictOfWords[word] += 1
    dictList.append(dictOfWords)
"""
# Kütüphane ile bow
# Bizim kendi yerimize cümle içindeki html tagleri veya noktalama işaretlerini vs temizler
from sklearn.feature_extraction.text import CountVectorizer
y = data.sentiment.replace({"positive":1, "negative":0})
x = data.review

bag = CountVectorizer()
X = bag.fit_transform(x)


# Data Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)


# Predict
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)


# Kendi yorumumu tahmin ettiricem
test1 = "I really did not enjoy watching this. Very disappointed." # real = 0
test2 = "What a wonderful movie. I enjoyed watching this with my kids." # real = 1

print(rfc.predict(bag.transform([test1])))
print(rfc.predict(bag.transform([test2])))