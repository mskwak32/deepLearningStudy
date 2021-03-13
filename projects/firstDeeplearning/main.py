import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

DATA_PATH = "../../dataset/ThoraricSurgery.csv"

np.random.seed(3)
tf.random.set_seed(3)

Data_set = np.loadtxt(DATA_PATH, delimiter=",")
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=10)
