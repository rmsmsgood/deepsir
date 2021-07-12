# import numpy as np
# import tensorflow as tf
from tensorflow import keras
import pandas as pd
import datetime
import os, sys, shutil

# os.chdir("C:/Users/rmsms/OneDrive/lab/deepsir")

# try:
#   endtime = int(sys.argv[1])
# except IndexError:
#   endtime = 100
endtime = 100
endtime -= 1

DATA = pd.read_csv("training_I " + sys.argv[1] + ".csv")
Y = DATA.iloc[:1500,2:4].to_numpy()
X = DATA.iloc[:1500,4:(4+endtime)].to_numpy(); X.shape

model = keras.Sequential()
for k in range(6):
  model.add(keras.layers.Dense(1000))
  model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(2))
model.compile(loss='mse', optimizer='adam')
model.build((None, endtime))
# model.summary()

print("- " * 10)
print("recieved endtime:" + str(endtime))
print(os.getcwd())

try:
  os.makedirs("result/" + str(endtime))
except:
  print("result/" + str(endtime) + "already exsits!")

report = open("report.csv", 'a')

# if os.path.isfile("best.h5"):
#   print("cuation: best.h5 already exsits!")
#   os.rename("best.h5", "backup_best.h5")
# os.system("pause")

# example_batch = X[:3]
# example_result = model.predict(example_batch)
# example_result

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# earlystopping = callbacks.EarlyStopping(monitor='val_loss')
modelcheckpoint = keras.callbacks.ModelCheckpoint(filepath = "result/" + str(endtime) + "/best.h5",
# modelcheckpoint = keras.callbacks.ModelCheckpoint(filepath = "best.h5",
    monitor='val_loss',
    verbose = 1,
    save_best_only=True)

history = model.fit(X, Y,
  epochs=1000,
  validation_split = 0.2,
  # callbacks=[modelcheckpoint])
  callbacks=[tensorboard, modelcheckpoint])
  # model_to_save = "T" + str(endtime) + " val_loss" + str(round(min(history.history['val_loss']),4))[1:] + ".h5"
  # shutil.copy("best.h5 ", model_to_save)
report.write(str(endtime) + "," + str(round(min(history.history['val_loss']),4)) + "\n")
report.close()

# model.predict(X[:3])
# Y[:3]
# np.abs(model.predict(X[:3]) - Y[:3])

# model.save(os.getcwd() + '\\last')
# loaded = keras.models.load_model("last")
loaded = keras.models.load_model("best.h5")

loaded.predict(X[:10]); Y[:10]
loaded.evaluate(X[:300], Y[:300])