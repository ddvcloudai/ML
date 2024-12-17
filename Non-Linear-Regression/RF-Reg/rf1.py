import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Sep28-SVR-KNN-POLY/28th/EMP_SAL.csv")

# segregating the dataset
x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

#perform splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Apply standard scaler 
scaler=StandardScaler()
x_scaler_train=scaler.fit_transform(x_train)
x_scaler_test=scaler.fit(x_test)

#Applying Random Forest
regressor=RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x_scaler_train,y_train)
'''
In case of RF or DT regression, you can ignore even scaling the data unless it is essential and can
try using the below code instead:
regressor=RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x_train,y_train)
This will generate promising results
'''
#predicting the value
y_pred=regressor.predict([[6.5]])
print(y_pred)
