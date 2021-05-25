import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import datetime


print(os.getcwd())

DATA = pd.read_csv("train.csv")

Y = DATA.iloc[:,1:3].to_numpy()
X = DATA.iloc[:,3:].to_numpy()

model = keras.Sequential()
model.add(layers.Dense(100, input_shape=(100,)))
model.add(layers.Activation('relu'))
model.add(layers.Dense(100))
model.add(layers.Activation('relu'))
model.add(layers.Dense(100))
model.add(layers.Activation('relu'))
model.add(layers.Dense(100))
model.add(layers.Activation('relu'))
model.add(layers.Dense(100))
model.add(layers.Activation('relu'))
model.add(layers.Dense(2))

loss_fn = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
model.compile(loss=loss_fn, optimizer='adam')

model.summary()

example_batch = X[:10]
example_result = model.predict(example_batch)
example_result

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('.', end='')
    if epoch % 1000 == 0: print('')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


EPOCHS = 100

history = model.fit(X, Y,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  callbacks=[PrintDot(), tensorboard_callback])

model.predict(X[:10])
Y[:10]

np.abs(model.predict(X[:10]) - Y[:10])

model.save(os.getcwd())