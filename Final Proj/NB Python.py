#code taken from https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
#https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python
#code taken from 
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv(r'C:\\Users\\Cole\\Downloads\\datasetreducedhikari.csv')

X = df.drop('Label',axis=1).values
y = df['Label'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)


model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))



