import logging
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import xgboost
from hundred_point_two_index import ContourPointsLabeler

logger = logging.getLogger()


# assert (new_foreground.height < new_background.height)
# pip install openpyxl natsort
# upload from filename of excel


class ContourPointsPredicter(ContourPointsLabeler):
    def __init__(self, excel_path, left_model_path, right_model_path):
        super().__init__(excel_path)
        self.left_model_path = left_model_path
        self.right_model_path = right_model_path

    def predict_idx(self, index, x_array, y_array):
        model_input = np.append(x_array, y_array).reshape((1, 400))

        model_L = xgboost.XGBRegressor()
        model_L.load_model(self.left_model_path)

        model_R = xgboost.XGBRegressor()
        model_R.load_model(self.right_model_path)

        left_pred = model_L.predict(model_input)
        right_pred = model_R.predict(model_input)

        self.df.loc[index, ['left']] = int(round(left_pred[0]))
        self.df.loc[index, ['right']] = int(round(right_pred[0]))

    def process_images(self, n, img_path: str, output_path: str):
        for index, self.file_name in tqdm(self.df['file'].iloc[n:].items()):
            print(f"index: {index}, file_loaded: {self.df.loc[index, 'file']}")
            self.xy_coordinate = []

            # Generate points of a semented boundary
            contour = self.generate_contour(img_path)

            # get a hundred indexes from contour
            hundred_idx = np.rint(np.linspace(0, len(contour) - 1, 200)).astype('int16')

            #
            x_array = (contour[hundred_idx, 0, 0].flatten() - contour[:, 0, 0].min()) / (
                    contour[:, 0, 0].max() - contour[:, 0, 0].min() + 1)
            y_array = (contour[hundred_idx, 0, 1].flatten() - contour[:, 0, 1].min()) / (
                    contour[:, 0, 1].max() - contour[:, 0, 1].min() + 1) * self.aspect_ratio

            # predict indices of two points
            self.predict_idx(index, x_array, y_array)

            # write x feature (model input)
            self.df.loc[index, self.columns[4:404]] = np.append(x_array, y_array).reshape((1, 400))
            self.df.loc[index, ['contour_n']] = len(contour)
            self.df.to_csv(output_path)


if __name__ == '__main__':

    labeler = ContourPointsPredicter(excel_path='two_hundred_points.xlsx',
                                     left_model_path="./xgb_model/outline_left_v0.4.model",
                                     right_model_path="./xgb_model/outline_right_v0.4.model",
                                     )
    labeler.process_images(1248, './segmented_images/', 'two_hundred_points.csv')
