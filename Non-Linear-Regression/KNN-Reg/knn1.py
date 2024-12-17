import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Sep28-SVR-KNN-POLY/28th/EMP_SAL.csv")

x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

regressor_knn=KNeighborsRegressor(n_neighbors=6,weights="distance",p=1)
regressor_knn.fit(x,y)

y_pred_knn=regressor_knn.predict([[6.5]])
print(y_pred_knn)
