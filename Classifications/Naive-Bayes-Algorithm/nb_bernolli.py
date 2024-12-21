import numpy as np, pandas as pd, matplotlib.pyplot as plt
dataset=pd.read_csv(r"/Users/divyadeepverma/Desktop/DS/Oct9-10/9th, 10th/Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train, y_train)
print(bias)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)
#Now we feed some value and find if the customer is likely to purchase car or not

#new data
new_data=np.array([[30,50000]])

#scaling new data
new_data_scaled=sc.transform(new_data)

#predicting the class for new data set
prediction=classifier.predict(new_data_scaled)

#Output of prediction
print(f"Prediction for new data point is: {prediction[0]}")
