import logging
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.xgb_regressor_from_cargen import GetContourHeightRatio, select_appropriate_contour

logger = logging.getLogger()


# assert (new_foreground.height < new_background.height)
# pip install openpyxl natsort
# upload from filename of excel


class ContourHeightLabeler:
    def __init__(self, excel_path):
        self.image_np = None
        self.df = pd.read_excel(excel_path, index_col=0)
        self.columns = self.df.columns
        self.xy_coordinate = []

    def mouse_click_left_side(self, event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.image_np, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("space", self.image_np)
            self.xy_coordinate.extend([x, y])

    def mouse_click_right_side(self, event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.image_np, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("space", self.image_np)
            self.xy_coordinate.extend([x, y])

    def get_label_idx(self, index, contour, hundred_idx):
        self.image_np = self.image_np[:, :, :3].copy()
        # 전체 화면으로 'image' 창 생성
        # cv2.namedWindow('space')
        # cv2.setWindowProperty('space', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for i in hundred_idx:
            point = contour[i, 0, :]
            cv2.circle(self.image_np, point, 2, (255, 0, 255), -1)

        idx_left = hundred_idx[self.df.loc[index, 'left']]
        idx_right = hundred_idx[self.df.loc[index, 'right']]
        cv2.circle(self.image_np, contour[idx_left, 0, :], 3, (0, 0, 255), -1)
        cv2.circle(self.image_np, contour[idx_right, 0, :], 3, (0, 0, 255), -1)

        # cv2.setMouseCallback("space", self.mouse_click_left_side)
        # cv2.setMouseCallback("space", self.mouse_click_right_side)
        # cv2.imshow("space", self.image_np)

        # save the labeled point
        filepath = os.path.join("output", self.file_name)
        cv2.imwrite(filepath, self.image_np)

        # waik for key press
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #
        ex_length_L = ex_length_R = 100000
        if not self.xy_coordinate.__eq__([]):
            for idx, val in enumerate(contour[hundred_idx]):
                length_left = np.sqrt(
                    (self.xy_coordinate[1] - val[0, 1]) ** 2 + (self.xy_coordinate[0] - val[0, 0]) ** 2)
                length_right = np.sqrt(
                    (self.xy_coordinate[3] - val[0, 1]) ** 2 + (self.xy_coordinate[2] - val[0, 0]) ** 2)
                if ex_length_L > length_left:
                    ex_length_L = length_left
                    idx_left = idx
                if ex_length_R > length_right:
                    ex_length_R = length_right
                    idx_right = idx
                self.df.loc[index, 'file'] = self.file_name
                self.df.loc[index, 'left'] = idx_left
                self.df.loc[index, 'right'] = idx_right

    def generate_contour(self, img_path):
        foreground = cv2.imread(os.path.join(img_path, self.file_name), cv2.IMREAD_UNCHANGED)

        # Remove the top and bottom, left and right margins of a photo (여백 제거)
        y, x = np.where(foreground[:, :, 3] > 0)
        foreground_np = foreground[y.min(): y.max(), x.min(): x.max(), :]
        self.aspect_ratio = foreground_np.shape[0] / foreground_np.shape[1]

        contour_generator = GetContourHeightRatio(foreground_np)
        car_on_rectangle_np = contour_generator.resize_and_paste_to_rectangle()
        contours_sorted = contour_generator.get_contours(car_on_rectangle_np)
        contour = select_appropriate_contour(contours_sorted)

        self.image_np = car_on_rectangle_np

        return contour

    def process_images(self, n, img_path: str, output_path: str):

        for index, self.file_name in tqdm(self.df['file'].iloc[n:].items()):
            print(f"index: {index}, file_loaded: {self.df.loc[index, 'file']}")
            self.xy_coordinate = []

            # Generate points of a semented boundary
            contour = self.generate_contour(img_path)

            # get a hundred indexes from contour
            hundred_idx = np.rint(np.linspace(0, len(contour) - 1, 200)).astype('int16')
            self.get_label_idx(index, contour, hundred_idx)

            #
            x_array = (contour[hundred_idx, 0, 0].flatten() - contour[:, 0, 0].min()) / (
                    contour[:, 0, 0].max() - contour[:, 0, 0].min() + 1)
            y_array = (contour[hundred_idx, 0, 1].flatten() - contour[:, 0, 1].min()) / (
                    contour[:, 0, 1].max() - contour[:, 0, 1].min() + 1) * self.aspect_ratio
            self.df.loc[index, self.columns[4:404]] = np.append(x_array, y_array).reshape((1, 400))
            self.df.loc[index, ['contour_n']] = len(contour)
        # self.df.to_excel(output_path)


if __name__ == '__main__':
    labeler = ContourHeightLabeler('two_hundred_points.xlsx')
    labeler.process_images(0, './segmented_images/', 'two_hundred_points2.xlsx')
