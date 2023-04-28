import logging
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

from utils.xgb_regressor_from_cargen import GetContourHeightRatio

df = pd.read_excel('two_hundred_points_v0.3.xlsx')
df_result = pd.DataFrame(np.zeros((len(df), 2)),
                         columns=['left_pred', 'right_pred'],
                         index=df.index)

img_path = './segmented_images'
model_path = './xgb_model'
model_left_path = os.path.join(model_path, 'outline_left_v0.3.model')
model_right_path = os.path.join(model_path, 'outline_right_v0.3.model')

left_model = xgb.XGBRegressor()
left_model.load_model(model_left_path)

right_model = xgb.XGBRegressor()
right_model.load_model(model_right_path)

for index, values in tqdm(df.iterrows()):
    print(index, values['file'])
    foreground = cv2.imread(os.path.join(img_path, values['file']), cv2.IMREAD_UNCHANGED)
    segmented_car_np = np.array(foreground)

    # Foreground(자동차) 이미지에서 상하, 좌우에 위치한 여백을 지운다.
    # 결과물인 segmented_car_array는 직사각형 이미지에 자동차가 여백없이 채워진 RGBA 이미지다.
    y, x = np.where(segmented_car_np[:, :, 3] > 0)
    car_in_box_np = segmented_car_np[y.min(): y.max(), x.min(): x.max(), :]

    # contour_left_ratio: 차량 높이 / 왼쪽 윤곽선 높이, contour_right_ratio: 차량 높이 / 오른쪽 윤곽선 높이
    contour_height_regressor = GetContourHeightRatio(car_in_box_np)
    contour_height_ratio = contour_height_regressor.predict(left_model, right_model, just_predict=True)

    df_result.loc[index, 'left_pred'] = contour_height_ratio[0]
    df_result.loc[index, 'right_pred'] = contour_height_ratio[1]

df_result.to_excel('test_pred.xlsx')
