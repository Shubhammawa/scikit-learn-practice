import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)

print(pima.head())

feature_cols = ['pregnant','insulin','bmi','age']
X = pima[feature_cols]
y = pima.label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegres	sion()
logreg.fit(X_train, y_train)

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

# Calculating null accuracy
# y_test.value_counts() - Gives number of zeroes and ones
# y_test.mean()	- Percentage of ones
# 1 - y_test.mean() - Percentage of zeroes
# Null_accuracy = max(y_test.mean(), 1 - y_test.mean())

# Calculating confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))

# Metrics computed from confusion matrix

# Senstivity
# TP/(TP + FN)

# Specificity 
# TN/(TN + FP)

# Fasle positive rate
# FP/(TN + FP)

# Precision
# TP/(TP + FP)