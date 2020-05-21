
import numpy as np
import cv2
from numpy import load
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_csv("DIP_data.csv")

X = data.drop(["Label"],axis = 1)
y = list(data["Label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# load the model from disk
filename = 'SVM_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print("Score: ",result)

y_pred = loaded_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
