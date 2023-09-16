import pandas as pd
import numpy as np

# Data Import
data = pd.read_csv("data/comments_and_reviews.csv")

# We organize our data set.
from sklearn.feature_extraction.text import CountVectorizer
y = data.sentiment.replace({"positive":1, "negative":0})
x = data.review
bag = CountVectorizer()
X = bag.fit_transform(x)

# Data Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Train testing data with Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X_train, y_train)

# Predict
y_pred = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)  # -> Accuracy rate


# We are testing the algorithm
test1 = "Very disappointed." # expected output = [0]
test2 = "What a wonderful movie. I enjoyed watching this with my kids." # expected output = [1]

print(rfc.predict(bag.transform([test1])))
print(rfc.predict(bag.transform([test2])))