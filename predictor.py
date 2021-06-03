from tensorflow import keras
import pandas as pd

DATA = pd.read_csv("training_I.csv")
Y = DATA.iloc[:,2:4].to_numpy()
X = DATA.iloc[:,4:].to_numpy()

loaded = keras.models.load_model("best.h5")

loaded.predict(X[:5]); Y[:5]

loaded.evaluate(X[:300], Y[:300])
sum(sum((loaded.predict(X[:300]) - Y[:300])**2))/600