import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

dataset=pd.read_csv(r'/Users/divyadeepverma/Desktop/DS/Sep28-SVR-KNN-POLY/28th/EMP_SAL.csv')
x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

#svr
regressor=SVR(kernel='rbf',degree=3,gamma='auto')
regressor.fit(x,y)

#predicting svr
y_pred_svr=regressor.predict([[6.5]])
print(y_pred_svr)
# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Might show incorrect results, need to look into the better version for fine tuning the model
