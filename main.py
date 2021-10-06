# %%
from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
data = pd.read_csv('data/1.csv')
# %%
data_np = data[data.columns[1:]].to_numpy()
data_x = data_np[:, 100:-100]
data_y = np.concatenate(
    [np.ones((len(train_x)//2, )), np.zeros((len(train_x)//2, ))], axis=0)
# %%
for n in [100, 200, 500, 1000]:
    acc = 0
    for fold in range(4):
        train_x = np.concatenate([data_x[(fold+i) % 4::5]
                                  for i in range(3)], axis=0)
        train_y = np.concatenate([data_y[(fold+i) % 4::5]
                                  for i in range(3)], axis=0)
        val_x = data_x[(fold+3) % 4::5]
        val_y = data_y[(fold+3) % 4::5]
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(train_x, train_y)
        acc += rf.score(val_x, val_y)/4
        #prediction = rf.predict(val_x)
        #acc += (prediction == val_y).astype(np.float).mean()/4
    print(f'{acc} {n}')
# %%
rf = RandomForestClassifier(n_estimators=100)
train_x = np.concatenate([data_x[(i) % 4::5]
                          for i in range(4)], axis=0)
train_y = np.concatenate([data_y[(i) % 4::5]
                          for i in range(4)], axis=0)
rf.fit(train_x, train_y)
test_x = data_x[4::5]
test_y = data_y[4::5]
acc = rf.score(test_x, test_y)
# %%
