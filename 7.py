import pandas as pd

col = ['Age, Gender, Family History', 'Diet', 'Life Style', 'Cholestrol', 'Heart Disease']
df = pd.read_csv('datasets/7.csv', names = col)

print(df)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
#Encode target labels with value between 0 and n_classes-1.
#print(df.iloc[0])
for i in range(len(col)):
    df.iloc[:, i] = encoder.fit_transform(df.iloc[:, i])
#print(df.iloc[0])

X = df.iloc[:, 0:6]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)

print('xtrain\n')
print(xtrain)
print('xtest\n')
print(xtest)
print('ytrain\n')
print(ytrain)
print('ytest\n')
print(ytest)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(xtrain, ytrain)

predictions = clf.predict(xtest)

print('predictions')
print(predictions)

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics

labels = np.unique(ytest)
a = confusion_matrix(ytest, predictions, labels = labels)

print('confusion matrix')
cm = pd.DataFrame(a, index = labels, columns=labels)
print(cm)

print('recall')
print(metrics.recall_score(ytest, predictions))
print('precision')
print(metrics.precision_score(ytest, predictions))