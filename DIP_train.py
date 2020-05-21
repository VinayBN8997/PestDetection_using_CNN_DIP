import sys
import os
import numpy as np
import json
import copy
import cv2
from numpy import load
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from DIP_utilities import get_DWF_image, get_segments, get_lbp_values


data = load("PestData_Train.npz")
X, y = data['arr_0'],data['arr_1']

print(X[0].shape)

df = pd.DataFrame({"Label":list(y.reshape(-1))},columns =["Label"])

df["Vector"] = np.nan
df = df.astype('object')

for i in tqdm(range(len(df))):
    img = X[i]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_img = get_DWF_image(img)
    segments = get_segments(result_img)
    lbp_values = get_lbp_values(segments)
    df["Vector"][i] = lbp_values

print(df.head(20))

data = pd.DataFrame(df.Vector.tolist(), index= df.index)
data["Label"] = df["Label"]

data.to_csv("DIP_data.csv",index = False)

X = data.drop(["Label"],axis = 1)
y = list(data["Label"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

filename = 'SVM_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))
