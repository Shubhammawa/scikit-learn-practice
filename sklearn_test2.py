from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
iris = load_iris()
type(iris)
#print(iris.data)
#print(iris.feature_names)
#print(iris.target)
#print(iris.target_names)
#print(type(iris.data))
#print(type(iris.target))
#print(iris.data.shape)
#print(iris.target.shape)

# Feature matrix
X = iris.data

# Labels
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
knn  = KNeighborsClassifier(n_neighbors = 1)    # n_neighbors - Hyperparamter
#print(knn)

knn.fit(X_train,y_train)

#print(knn.predict([[3,5,4,2],[3,5,4,2]]))
#X_new = [[3,5,4,2],[5,4,3,2]]

p = knn.predict(X_test)
print(p)
print('Test set accuracy - knn')
print(metrics.accuracy_score(y_test,p))

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

p_logreg = logreg.predict(X_test)
print(p_logreg)
print('Test set accuracy - logistic regression')
print(metrics.accuracy_score(y_test,p_logreg))
