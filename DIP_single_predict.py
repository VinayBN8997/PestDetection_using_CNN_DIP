from keras.preprocessing.image import img_to_array
import cv2
import pickle
import numpy as np
from DIP_utilities import get_DWF_image, get_segments, get_lbp_values

def make_predict_DIP_SVM(image, size):
    X = cv2.resize(image, size)
    X = img_to_array(X)
    X = np.array(X).reshape(-1, size[0], size[1], 3)
    img = cv2.cvtColor(X[0], cv2.COLOR_BGR2GRAY)
    result_img = get_DWF_image(img)
    segments = get_segments(result_img)
    lbp_values = get_lbp_values(segments)
    filename = 'SVM_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    res = loaded_model.predict([lbp_values])[0]

    return res


size = (180, 120)

image = cv2.imread("./images/Train_6.jpg", 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
res = make_predict_DIP_SVM(image, size)

if res == 0:
    print("Not healthy")
else:
    print("Healthy")
