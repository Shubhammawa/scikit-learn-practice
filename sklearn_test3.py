import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
print(data.shape)
print(data.head())
print(data.tail())
#print(data.shape)

X = data[['TV', 'radio', 'newspaper']]
#X.head()
y = data.sales
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', kind='reg' )#,size=7,aspect=0.7)
plt.show()

feature_cols = ['TV', 'radio', 'newspaper']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

print(linreg.intercept_)
print(linreg.coef_)

list(zip(feature_cols, linreg.coef_))

#make predictions on the testing set
y_pred = linreg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# create a Python list of feature names
feature_cols = ['TV', 'radio']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# select a Series from the DataFrame
y = data.sales

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
