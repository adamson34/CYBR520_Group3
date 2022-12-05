#used https://www.section.io/engineering-education/detecting-malicious-url-using-machine-learning/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

df = pd.read_csv(r'datasetreducedhikari.csv')
print(df.shape)

X = df.drop('Label',axis=1).values
y = df['Label'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)

neighbors = np.arange(1,81)

train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))
y_pred = knn.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test,y_pred))

# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(networkDataset, networkDataset.Label, test_size=0.3,random_state=42)

