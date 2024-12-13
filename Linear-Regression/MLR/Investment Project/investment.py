import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'/Users/divyadeepverma/Desktop/DS/Sep25-MLR/MLR/Investment.csv')

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, 4]

X=pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) #added new column because constant is missing from equation

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5]] # now we are considering all row values and all columns from index 0 to index 5 i.e 6 columns
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# step1: check the summary from line 31. You might observe ANOVA sheet shows r2> adj.R2.
# step2: check p value for each variable. Reject Ho if p>0.05. Remove the variable having highest p value.
# step3: showing t test coz data is sample data
X_opt = X[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
#Repeat the steps
X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Here at the end, came to understand that index1 is the actual winner. Therefore, the company should invest in R&D
