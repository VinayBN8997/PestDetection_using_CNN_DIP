# load json and create model
from tensorflow.python.keras.models import model_from_json
from sklearn.metrics import classification_report
from numpy import load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = load("PestData_Train.npz")
X, y = data['arr_0'],data['arr_1']

print(X.shape)
print(y.shape)

X = np.array(X)
y = np.array(y)
# Normalizing X values between 0 to 1.
X = X.astype('float32')
X /= np.max(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("CNN_weights.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Test data")
score = model.evaluate(X_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

y_preds = model.predict(X_test, verbose=1)
print(y_preds.shape)

print(y_preds[:10])

preds = model.predict_classes(X_test, verbose=1)
print(preds.shape)

print("Actual healthy: ",len(y_test[y_test == 1]))
print("Predicted healthy: ",len(preds[preds == 1]))

print(classification_report(y_test, preds, target_names=['Non-healthy', 'Healthy']))
