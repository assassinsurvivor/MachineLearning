#Dataset:https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("c:/users/ashub/downloads/breast-cancer-wisconsin.data.txt")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)#useless feature would increase the time as well as decrease the score

X=df.drop(["j"],1)# axis=1 represents column is dropped and 0 represents a row is dropped.X stores features
y=df["j"]#label
classifier=KNeighborsClassifier(n_neighbors=5,n_jobs=-1) #5 is default in sklearn
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21,stratify=y)#random_state is used to maintain the score
classifier.fit(x_train,y_train)
tester=np.array([8,9,9,8,7,10,9,7,1])
print(classifier.predict(tester.reshape(1,-1)))
print(classifier.score(x_test,y_test))


