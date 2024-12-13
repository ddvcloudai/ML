import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r'/Users/divyadeepverma/Desktop/DS/Sep22-SLR/22nd/SLR - Practicle/House_data.csv')
space=df['sqft_living']
price=df['price']

x = np.array(space).reshape(-1,1)
y = np.array(price)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#Visualizing the training Test Results
plt.scatter(x_train, y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('Visuals fro Trainning Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

#Visualizing the Test Results
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Visuals for test dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

#overfitting check
bias=regressor.score(x_train,y_train)
print(bias)

#underfitting check
variance=regressor.score(x_test,y_test)
print(variance)

'''This suggests that the model is achieving a reasonable balance between bias and variance. 
It indicates that the model is neither underfitting (high bias, low variance) nor 
overfitting (low bias, high variance) excessively.'''


