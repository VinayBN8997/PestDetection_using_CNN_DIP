from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint
from numpy import load


model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(size[0], size[1], 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

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

filepath="CNN_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, y_train, epochs=16, batch_size=16, callbacks=callbacks_list, validation_data=(X_test, y_test))

# serialize model to JSON
model_json = model.to_json()
with open("CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNN_model.h5")
print("Saved model to disk")
