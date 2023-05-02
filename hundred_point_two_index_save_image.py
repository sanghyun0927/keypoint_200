import logging
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from hundred_point_two_index import ContourPointsLabeler

logger = logging.getLogger()


# assert (new_foreground.height < new_background.height)
# pip install openpyxl natsort
# upload from filename of excel


class PointsImageGenerator(ContourPointsLabeler):
    def __init__(self, excel_path, out_dir):
        super().__init__(excel_path)
        self.out_dir = out_dir

    def get_label_idx(self, index, contour, hundred_idx):
        self.image_np = self.image_np[:, :, :3].copy()
        # 전체 화면으로 'image' 창 생성
        # cv2.namedWindow('space')
        # cv2.setWindowProperty('space', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for i in hundred_idx:
            point = contour[i, :]
            cv2.circle(self.image_np, point, 2, (255, 0, 255), -1)

        # idx_left = hundred_idx[self.df.loc[index, 'left']]
        # idx_right = hundred_idx[self.df.loc[index, 'right']]
        # left_height = contour[idx_left, 1]
        # right_height = contour[idx_right, 1]

        left_height = int(round(self.df.loc[index, 'left']))
        right_height = int(round(self.df.loc[index, 'right']))
        horizon_height = int(round(0.75 * (contour[:, 1].max() - contour[:, 1].min() + 1) + contour[:, 1].min()))
        mask_height = max(min(left_height, right_height), horizon_height)

        # cv2.circle(self.image_np, contour[idx_left, :], 5, (0, 0, 255), -1)
        # cv2.circle(self.image_np, contour[idx_right, :], 5, (0, 0, 255), -1)
        cv2.line(self.image_np, (0, mask_height), (1000, mask_height), (200, 200, 200), 4)
        cv2.line(self.image_np, (0, left_height), (500, left_height), (0, 0, 255), 1)
        cv2.line(self.image_np, (501, right_height), (1000, right_height), (255, 0, 0), 1)

        # cv2.setMouseCallback("space", self.mouse_click_left_side)
        # cv2.setMouseCallback("space", self.mouse_click_right_side)
        # cv2.imshow("space", self.image_np)

        # save the labeled point
        filepath = os.path.join(self.out_dir, self.file_name)
        cv2.imwrite(filepath, self.image_np)

        # waik for key press
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_images(self, n, img_path: str):

        for index, self.file_name in tqdm(self.df['file'].iloc[n:].items()):
            print(f"index: {index}, file_loaded: {self.df.loc[index, 'file']}")
            self.xy_coordinate = []

            # Generate points of a semented boundary
            contour = np.squeeze(self.generate_contour(img_path), axis=1)

            # get a hundred indexes from contour
            hundred_idx = np.rint(np.linspace(0, len(contour) - 1, 200)).astype('int16')
            self.get_label_idx(index, contour, hundred_idx)


if __name__ == '__main__':
    labeler = PointsImageGenerator('two_hundred_points_v0.3_pred.xlsx', './bobaedream_v0.3_pred')
    labeler.process_images(1248, './segmented_images/')
