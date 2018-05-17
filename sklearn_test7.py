import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

feature_cols = ['TV', 'radio','newspaper']

X = data[feature_cols]

y = data.sales

lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
print(scores)

mse_scores = -scores
print mse_scores

rmse_scores = np.sqrt(mse_scores)
print rmse_scores

print rmse_scores.mean()

feature_cols_new = ['TV','radio']
X_new = data[feature_cols_new]
print(np.sqrt(-cross_val_score(lm, X_new, y, cv=10, scoring='mean_squared_error'))).mean()