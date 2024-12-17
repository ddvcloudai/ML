import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Sep28-SVR-KNN-POLY/28th/EMP_SAL.csv")
x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)

regressor=DecisionTreeRegressor(criterion='friedman_mse',splitter='random')
regressor.fit(x_train_scaler,y_train)

y_pred=regressor.predict([[6.5]])
print(y_pred)
