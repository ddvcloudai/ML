import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

data=pd.read_csv("/Users/divyadeepverma/Desktop/DS/Sep27-L1-L2/27th/lasso, ridge, elastic net/TASK-22_LASSO,RIDGE/car-mpg.csv")
#print(data.head(5))

#dropping car names
data=data.drop(['car_name'],axis=1)
#Replace the origin column
data['origin']=data['origin'].replace({1:'america',2:'europe',3:'asia'})
data=pd.get_dummies(data, columns=['origin'])
data=data.replace('?',np.nan)
data = data.apply(
    lambda x: x.fillna(x.median()) if x.dtype in [np.int64, np.float64] else x.fillna(x.mode()[0]),
    axis=0
)
#print(data.head(5))
x=data.drop(['mpg'], axis=1)
y=data[['mpg']]
#print(x.head(5))
#print(type(x))

#Stanrdising columns
scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)
y_scaler=scaler.fit_transform(y)
x_scaler_df=pd.DataFrame(x_scaler,columns=x.columns)
y_scaler_df=pd.DataFrame(y_scaler,columns=y.columns)
#print(y_scaler_df.tail(5))

#train test split after standardization
x_train,x_test,y_train,y_test=train_test_split(x_scaler_df,y_scaler_df, test_size=0.2, random_state=0)
#print(x_train)

#model fitting
model=LinearRegression()
model.fit(x_train,y_train)

#finding coeficient
coeficients=model.coef_.flatten() #converts 2D to 1D array
intercept=model.intercept_
coeficients_df=pd.DataFrame(coeficients,columns=["coeficient"], index=x_train.columns)
#print("coeficient for each feature:")
#print(coeficients_df)
#print("\nIntercept:")
#print(intercept)

#Applying Ridge:
ridge_model=Ridge(alpha=0.3)
ridge_model.fit(x_train,y_train)
print('Ridge model coef:{}'.format(ridge_model.coef_))

#Apply lasso
lasso_model=Lasso(alpha=0.1)
lasso_model.fit(x_train,y_train)
print('Lasso model coef:{}'.format(lasso_model.coef_))

#model comparison:
#1.linear regression
print(model.score(x_train,y_train))
print(x_test,y_test)
#2.Ridge regression
print(ridge_model.score(x_train,y_train))
print(ridge_model.score(x_test,y_test))
#3.Lasso regression
print(lasso_model.score(x_train,y_train))
print(lasso_model.score(x_test,y_test))
