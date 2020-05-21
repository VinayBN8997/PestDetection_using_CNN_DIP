import numpy as np
import pandas as pd
import cv2
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from tqdm import tqdm_notebook as tqdm
from numpy import load

size = (180,120)

def save_tarin_images_as_list(train_files, images_path, y_dict, size):
    data_list = list()
    label_list = list()
    for file in tqdm(train_files):
        path = images_path + file
        pixels = load_img(path, target_size= size)
        pixels = img_to_array(pixels)
        data_list.append(pixels)
        label_list.append(y_dict[file.split(".")[0]])
    return [data_list,label_list]

images_path = "./images/"
files = os.listdir(images_path)

train_files = [i for i in files if "Train" in i]

train_df = pd.read_csv("train.csv")

print("Total: ", len(train_df))
print("Healthy: ",len(train_df[train_df["healthy"] == 1]))
print("Diseased: ",len(train_df[train_df["healthy"] == 0]))

y_dict = train_df.set_index('image_id')['healthy'].to_dict()

data = save_tarin_images_as_list(train_files, images_path, y_dict, size)

X = data[0]
y = data[1]

filename = 'PestData_Train.npz'
savez_compressed(filename, X, np.array([[i] for i in y]))
print('Saved dataset: ', filename)
