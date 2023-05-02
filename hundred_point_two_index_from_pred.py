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

    def predict_idx(self, index, contour, hundred_idx, x_array, y_array):
        model_input = np.append(x_array, y_array).reshape((1, 400))

        model_L = xgboost.XGBRegressor()
        model_L.load_model(self.left_model_path)

        model_R = xgboost.XGBRegressor()
        model_R.load_model(self.right_model_path)

        contour_left_ratio = model_L.predict(model_input) / self.aspect_ratio
        left_height = contour_left_ratio * (
                    contour[:, 1].max() - contour[:, 1].min() + 1) + contour[:, 1].min()
        contour_right_ratio = model_R.predict(model_input) / self.aspect_ratio
        right_height = contour_right_ratio * (
                    contour[:, 1].max() - contour[:, 1].min() + 1) + contour[:, 1].min()
        self.df.loc[index, 'left'] = left_height
        self.df.loc[index, 'right'] = right_height

        #
        # ex_length_L = ex_length_R = 100000
        # for idx, val in enumerate(contour[hundred_idx][101:]):
        #     length_left = np.abs(left_height - val[1])
        #     if ex_length_L > length_left:
        #         ex_length_L = length_left
        #         idx_left = idx + 101
        # for idx, val in enumerate(contour[hundred_idx][:101]):
        #     length_right = np.abs(right_height - val[1])
        #     if ex_length_R > length_right:
        #         ex_length_R = length_right
        #         idx_right = idx
        #
        # self.df.loc[index, 'left'] = idx_left
        # self.df.loc[index, 'right'] = idx_right

    def process_images(self, n, img_path: str, output_path: str):
        for index, self.file_name in tqdm(self.df['file'].iloc[n:].items()):
            print(f"index: {index}, file_loaded: {self.df.loc[index, 'file']}")
            self.xy_coordinate = []

            # Generate points of a semented boundary
            contour = np.squeeze(self.generate_contour(img_path), axis=1)

            # get a hundred indexes from contour
            hundred_idx = np.rint(np.linspace(0, len(contour) - 1, 200)).astype('int16')

            #
            x_array = (contour[hundred_idx, 0].flatten() - contour[:, 0].min()) / (
                    contour[:, 0].max() - contour[:, 0].min() + 1)
            y_array = (contour[hundred_idx, 1].flatten() - contour[:, 1].min()) / (
                    contour[:, 1].max() - contour[:, 1].min() + 1) * self.aspect_ratio

            # predict indices of two points
            self.predict_idx(index, contour, hundred_idx, x_array, y_array)

            # write x feature (model input)
            self.df.loc[index, self.columns[4:404]] = np.append(x_array, y_array).reshape((1, 400))
            self.df.loc[index, ['contour_n']] = len(contour)
        self.df.to_excel(output_path)


if __name__ == '__main__':

    labeler = ContourPointsPredicter(excel_path='two_hundred_points.xlsx',
                                     left_model_path="./xgb_model/outline_left_v0.3.model",
                                     right_model_path="./xgb_model/outline_right_v0.3.model",
                                     )
    labeler.process_images(1565, './segmented_images/', 'two_hundred_points_v0.3_pred.xlsx')
