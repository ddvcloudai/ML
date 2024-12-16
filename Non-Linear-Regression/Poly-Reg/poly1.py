import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Sep28-SVR-KNN-POLY/28th/EMP_SAL.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#Poly Regression
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#since the dataset is small, we are not supposed to perform train test split

#Building plots for poly
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
#print(plt.show())

#checking salary for x=6.5 (experience)
a=lin_reg.predict([[10]])
b=lin_reg_2.predict(poly_reg.fit_transform([[10]]))
print(a,b)

#try changing the degree in order to squeeze the salary and find out the best convergence
#for optimal value of salary 
