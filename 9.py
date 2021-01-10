import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print(X[: 5])
print(y[: 5])

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y)

print('xtrain')
print(xtrain.shape)
print('xtest')
print(xtest.shape)
print('ytrain')
print(ytrain.shape)
print('ytest')
print(ytest.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(xtrain, ytrain)

pred = knn.predict(xtest)

from sklearn import metrics

print('accuracy')
print(metrics.accuracy_score(ytest, pred))
print(iris.target_names[2])

ytestn=[iris.target_names[i] for i in ytest]
predn=[iris.target_names[i] for i in pred]
print(" predicted Actual")
for i in range(len(pred)):
    print(i," ",predn[i]," ",ytestn[i])