import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost
from sklearn.model_selection import train_test_split


df = pd.read_excel('two_hundred_points.xlsx', index_col=[0])

X_feature = df.iloc[:, 3:]
Y_label = pd.DataFrame(np.zeros((len(df), 2)), columns=['y_left', 'y_right'])

for idx, val in df.iterrows():
    y_left = val['y' + str(val['left'])]
    y_right = val['y' + str(val['right'])]
    Y_label.iloc[idx] = {'y_left': y_left, 'y_right': y_right}

x_train, x_test, y_train_left, y_test_left, y_train_right, y_test_right = train_test_split(
    X_feature, Y_label['y_left'], Y_label['y_right'], test_size=0, shuffle=True, random_state=42
)

model_L = xgboost.XGBRegressor()
model_L.fit(x_train, y_train_left)

model_R = xgboost.XGBRegressor()
model_R.fit(x_train, y_train_right)

left_pred = model_L.predict(x_test)
left_error = (y_test_left - left_pred)/y_test_left * 100

right_pred = model_R.predict(x_test)
right_error = (y_test_right - right_pred)/y_test_right * 100

idx = 0
for _, val in x_test.iterrows():
    x_array = val[:200].to_numpy()
    y_array = 1.5 - val[200:].to_numpy()

    left_height = 1.5 - left_pred[idx]
    right_height = 1.5 - right_pred[idx]
    idx += 1

    plt.scatter(x_array, y_array)
    plt.plot([0, 0.5], [left_height, left_height], c='r')
    plt.plot([0.5, 1], [right_height, right_height], c='r')
    plt.axis('equal')
    plt.savefig(f'height_predict/{idx}.png', dpi=300)
    plt.cla()