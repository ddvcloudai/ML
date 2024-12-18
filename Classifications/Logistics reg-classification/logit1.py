import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Oct3-Logit/3rd,4th/2.LOGISTIC REGRESSION CODE/logit classification.csv")

x=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,4]
'''
#checking type of dataset (good to know)
print(isinstance(dataset,pd.DataFrame))
print (isinstance(x,pd.DataFrame))
print(isinstance(y,np.ndarray))
print(isinstance(y,pd.Series))
print(y.head(5))
'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#apply logistic regression
classifier=LogisticRegression(penalty="l2",solver="sag")
classifier.fit(x_train,y_train)

#predicting test set results
y_pred=classifier.predict(x_test)
#print(y_pred)

#creating dataframe to compare results (optional)

compare_df=pd.DataFrame({
    "Predicted y" : y_pred,
    "Actual y" : y_test,
    "Match" : np.where(y_pred==y_test, 'Yes','No')
})
#Finding total numbers of mismatch (optional)
mismatch=(compare_df["Match"]=='No').sum()
total_records=len(compare_df)
#print(f"Total Records: {total_records}")
#print(f"Mismatches: {mismatch}")

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#creating accuracy score (to verify model accuracy)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

#bias variance check
bias = classifier.score(x_train, y_train)
print(bias)

variance = classifier.score(x_test, y_test)
print(variance)




