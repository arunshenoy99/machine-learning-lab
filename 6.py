import pandas as pd

msg = pd.read_csv('datasets/6.csv', names = ['message', 'label'])
print('The dimensions of the dataset:', msg.shape)
print ("The dataset:")
print(msg)
print()

msg['labelnum'] = msg.label.map({ 'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names())

#The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)
print("ytest")
print(ytest)
print("predicted")
print(predicted)

from sklearn import metrics

print("Accuracy metrics")
print("Accuracy of the classifier is", metrics.accuracy_score(ytest, predicted))

print("Confusion matrix")

import numpy as np
labels = np.unique(ytest)
a = metrics.confusion_matrix(ytest, predicted, labels = labels)
cm = pd.DataFrame(a, index = labels, columns = labels)
print(cm)

print("Recall(TP/TP+FN)")
print(metrics.recall_score(ytest, predicted))
print("Precision(TP/TP+FN)")
print(metrics.precision_score(ytest, predicted))