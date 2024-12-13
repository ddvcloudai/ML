import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset= pd.read_csv(r'/Users/divyadeepverma/Desktop/DS/Sep25-MLR/MLR/House_data.csv')

#checking if any value is missing
print(dataset.isnull().any())

print(dataset.dtypes)

#dropping the id and date column
dataset = dataset.drop(['id','date'], axis = 1)
print(dataset)

#understanding the distribution with seaborn
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[]);

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)

import statsmodels.api as sm

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
