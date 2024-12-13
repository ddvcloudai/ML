
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'/Users/XYZ/Desktop/DS/Sep22-SLR/22nd/SIMPLE LINEAR REGRESSION/Salary_Data.csv')

#splitting the data to dv and iv
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting the data into training and testing with 80 % split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#now is the time to apply regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

'''in line 18, we are trying to find the best fit line which finds out an equation. In addition, as
we put values of x_test to the model equation we get predicted values of y. The overall outcome is the comparison
of y_pred to the y_test values to come to a conclusion. This we can see based on graphs below'''

#visualization of training data points
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title("SALARY VS EXPERIENCE")
plt.xlabel('years of experience')
plt.ylabel('salary')
#plt.show()

#visualization of test data points
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title("SALARY VS EXPERIENCE")
plt.xlabel('years of experience')
plt.ylabel('salary')
#plt.show()

#generating slope
m=regressor.coef_
c=regressor.intercept_
#print(m,c)

#overfitting check
bias=regressor.score(x_train,y_train)
print(bias)

#underfitting check
variance=regressor.score(x_test,y_test)
print(variance)

# this is considered good model as the values of bias and variances are low
