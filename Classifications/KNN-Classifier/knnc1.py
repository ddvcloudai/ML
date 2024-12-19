import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Oct6/6th/Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)

y_pred=knn_classifier.predict(X_test)

cm=confusion_matrix(y_pred, y_test)
print(cm)

ac=accuracy_score(y_pred,y_test)
print(ac)

bias=knn_classifier.score(X_train,y_train)
print(bias)

variance=knn_classifier.score(X_test,y_test)
print(variance)

#Now we feed some value and find if the customer is likely to purchase car or not

#new data
new_data=np.array([[30,50000]])

#scaling new data
new_data_scaled=sc.transform(new_data)

#predicting the class for new data set
prediction=knn_classifier.predict(new_data_scaled)

#Output of prediction
print(f"Prediction for new data point is: {prediction[0]}")




