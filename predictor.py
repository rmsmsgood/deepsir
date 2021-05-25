from tensorflow import keras
import pandas as pd

DATA = pd.read_csv("train.csv")
Y = DATA.iloc[:,1:3].to_numpy()
X = DATA.iloc[:,3:].to_numpy()

loaded = keras.models.load_model("best.h5")

loaded.predict(X[:10]); Y[:10]
loaded.evaluate(X[:300], Y[:300])