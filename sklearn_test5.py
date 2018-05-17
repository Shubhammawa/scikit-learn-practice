from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 3)

kf = KFold(25,n_folds=5, shuffle=False)

print('{} {:^61} {}'.format('Iteration','Training set observations','Testing set observations'))
for iteration, data in enumerate(kf, start=1):
	print('{:^9} {} {:^25}'.format(iteration, data[0],data[1]))

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())