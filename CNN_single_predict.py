from keras.preprocessing.image import img_to_array
import cv2
from tensorflow.python.keras.models import model_from_json
import numpy as np


def make_predict_CNN(image, size):
    X = cv2.resize(image, size)
    X = img_to_array(X)
    X = np.array(X).reshape(-1, size[0], size[1], 3)
    X = X.astype('float32')
    X /= 255

    json_file = open('CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("CNN_weights.h5")

    res = model.predict_classes(X)[0][0]

    return res

size = (180, 120)

image = cv2.imread("./images/Train_11.jpg", 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
res = make_predict_CNN(image, size)

if res == 0:
    print("Not healthy")
else:
    print("Healthy")
