from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris() # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
#build model on the training set.
knn.fit(X_train, y_train)
#97% correct predictions.
result = knn.score(X_test, y_test)
print(result)