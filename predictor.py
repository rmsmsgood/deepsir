from tensorflow import keras
import pandas as pd

endtime = 100

DATA = pd.read_csv("training_I 5.csv")
Y = DATA.iloc[:40000,1:3].to_numpy()
X = DATA.iloc[:40000,3:(3+endtime)].to_numpy(); X.shape
# Y = DATA.iloc[:,2:4].to_numpy()
# X = DATA.iloc[:,4:].to_numpy()

loaded = keras.models.load_model("best.h5")

loaded.predict(X[1500:1505]); Y[1500:1505]

loaded.evaluate(X[1500:1800], Y[1500:1800])
sum(sum((loaded.predict(X[1500:1800]) - Y[1500:1800])**2))/600

# ---

DATA = pd.read_csv("대구광역시_기초지자체별 일일 코로나19 확진자 수_20210309.csv", encoding='utf-8')
real = DATA.iloc[:endtime,2].to_numpy().reshape(1,endtime)
loaded.predict(real) # seed number 68