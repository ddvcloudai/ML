import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.svm import SVC

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Oct6/6th/Social_Network_Ads.csv")
print(dataset.info())

x=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=SVC()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print(y_pred)

cm=confusion_matrix(y_test,y_pred)
print(cm)

ac=accuracy_score(y_test,y_pred)
print(ac)

bias=classifier.score(X_train,y_train)
print(bias)

variance=classifier.score(X_test,y_test)
print(variance)

cr=classification_report(y_test,y_pred)
print(cr)

