import os

for n in [10, 50, 100]:
    os.system("python trainer.py " + str(n))

for n in [30, 20, 40, 70, 60, 80, 90]:
    os.system("python trainer.py " + str(n))

for n in range(10,100,5):
    if n % 10 != 0:
        os.system("python trainer.py " + str(n))