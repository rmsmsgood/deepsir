from tensorflow import keras
import pandas as pd

endtime = 10

DATA = pd.read_csv("training_I.csv")
Y = DATA.iloc[:40000,2:4].to_numpy()
X = DATA.iloc[:40000,4:(4+endtime)].to_numpy(); X.shape
# Y = DATA.iloc[:,2:4].to_numpy()
# X = DATA.iloc[:,4:].to_numpy()

loaded = keras.models.load_model("best.h5")

loaded.predict(X[:5]); Y[:5]

loaded.evaluate(X[:300], Y[:300])
sum(sum((loaded.predict(X[:300]) - Y[:300])**2))/600

# ---

DATA = pd.read_csv("대구광역시_기초지자체별 일일 코로나19 확진자 수_20210309.csv", encoding='utf-8')
real = DATA.iloc[:endtime,2].to_numpy().reshape(1,endtime)
loaded.predict(real) # seed number 68