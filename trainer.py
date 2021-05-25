# import numpy as np
# import tensorflow as tf
from tensorflow import keras
import pandas as pd
import datetime
import os


print(os.getcwd())
DATA = pd.read_csv("train.csv")
Y = DATA.iloc[:,1:3].to_numpy()
X = DATA.iloc[:,3:].to_numpy()

model = keras.Sequential()
for k in range(6):
  model.add(keras.layers.Dense(1000))
  model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(2))
model.compile(loss='mse', optimizer='adam')
model.build((None, 100))
model.summary()

# example_batch = X[:3]
# example_result = model.predict(example_batch)
# example_result

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# earlystopping = callbacks.EarlyStopping(monitor='val_loss')
modelcheckpoint = keras.callbacks.ModelCheckpoint(filepath = "best.h5",
    monitor='val_loss',
    verbose = 1,
    save_best_only=True)

history = model.fit(X, Y,
  epochs=1000,
  validation_split = 0.2,
  callbacks=[tensorboard, modelcheckpoint])

# model.predict(X[:3])
# Y[:3]
# np.abs(model.predict(X[:3]) - Y[:3])

# model.save(os.getcwd() + '\\last')
# loaded = keras.models.load_model("last")
loaded = keras.models.load_model("best.h5")

loaded.predict(X[:10]); Y[:10]
loaded.evaluate(X[:300], Y[:300])