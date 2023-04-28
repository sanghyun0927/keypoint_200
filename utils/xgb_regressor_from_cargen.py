import logging

import numpy as np
import cv2
import xgboost as xgb


def select_appropriate_contour(contours_sorted):
    """
    Select the appropriate contour from a sorted list of contours.
    Args:
        contours_sorted (list): A list of sorted contours.
    Returns:
        numpy.ndarray: The selected contour.
    """
    # Try to select the appropriate contour
    if len(contours_sorted) > 1:
        first_contour, second_contour = contours_sorted[:2]
        # if the contour contain "0" value as x or y coordinates
        # you should choose other contour, because the contour with "0" value is boundary of whole image
        if (first_contour[0, 0, 0] == 0) or (first_contour[0, 0, 1] == 0):
            return second_contour
        else:
            return first_contour
    else:
        # len(contours_sorted) == 1
        return contours_sorted[0]


def generate_1000_rectangle():
    """
    transparent: 투명한
    """
    transparent_rect_np = np.zeros((1000, 1000, 4), dtype='uint8')

    return transparent_rect_np


class GetContourHeightRatio:
    def __init__(self, segmented_car_np: np.array):
        """

        """
        self.image_np = None
        self.aspect_ratio = None
        self.segmented_car_np = segmented_car_np

    def predict(
            self,
            model_left: xgb.XGBRegressor,
            model_right: xgb.XGBRegressor,
            just_predict: bool = False
    ):
        """
        Args:
            model_left (xgb.XGBRegressor):
            model_right (xgb.XGBRegressor):
            just_predict:
        Returns:
            left_pred (float):
            right_pred (float):
        """
        car_on_rectangle_np = self.resize_and_paste_to_rectangle()
        contours_sorted = self.get_contours(car_on_rectangle_np)
        contour = select_appropriate_contour(contours_sorted)

        x_array, y_array = self.min_max_normalize(contour)
        features = np.append(x_array, y_array).reshape((1, 400))
        if just_predict:
            contour_left_ratio = model_left.predict(features)
            contour_right_ratio = model_right.predict(features)
        else:
            contour_left_ratio = model_left.predict(features) / self.aspect_ratio
            contour_right_ratio = model_right.predict(features) / self.aspect_ratio

        return contour_left_ratio, contour_right_ratio

    def resize_and_paste_to_rectangle(self):
        """
        1. resize car (object_fit, width=800px)
        2. generate transparent(투명) rentangle image (width=1000px, height=1000px)
        3. locate resized car at the center of the rectangle (center=(500px, 500px))
        """
        resized_car_np = self.resize_to_width_800()
        rectangle_np = generate_1000_rectangle()

        resized_car_height = resized_car_np.shape[0]
        box_left_idx = 100
        box_right_idx = 900
        box_top_idx = 500 - resized_car_height // 2
        box_bottom_idx = box_top_idx + resized_car_height
        rectangle_np[box_top_idx:box_bottom_idx, box_left_idx:box_right_idx, :] = resized_car_np

        return rectangle_np

    def resize_to_width_800(self):
        """
        resize car array to 800px width image (NO margin at left, right, top, below)
        """
        # calculate aspect_ratio for min_max_normalize
        self.aspect_ratio = self.segmented_car_np.shape[0] / self.segmented_car_np.shape[1]

        # resize car array
        segmented_car_width = self.segmented_car_np.shape[1]
        size_factor = 800 / segmented_car_width
        resized_car_np = cv2.resize(
            self.segmented_car_np, (0, 0), fx=size_factor, fy=size_factor,
            interpolation=cv2.INTER_LANCZOS4
        )

        return resized_car_np

    def get_contours(self, image_np):
        """
        Select the appropriate contour from the alpha mask of an image.
        Args:
            image_np (np.array): An array of image with an alpha channel.
        Returns:
            numpy.ndarray: The selected contour.
        """
        # Convert alpha channel to numpy array
        self.image_np = image_np

        # Threshold the alpha mask to binary
        _, bw = cv2.threshold(self.image_np[:, :, 3], 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find contours in the binary alpha mask
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Sort contours by length in descending order
        contours_sorted = sorted(contours, key=len, reverse=True)

        return contours_sorted

    def min_max_normalize(self, contour):
        """
        1. Extract 200 points from contour
        2. Conduct custom MinMax normalization
        Args:
            contour (np.array): An array of x, y coordinates of car bourdary.
        Returns:
            x_array (np.array): 200 normalized x coordinates
            y_array (np.array): 200 normalized y coordinates
        """
        # get 200 indexes from contour
        hundred_idx = np.rint(np.linspace(0, len(contour) - 1, 200)).astype('int16')

        # MinMax normalization, aspect_ratio used for preserve original ratio (height/width)
        contour = np.squeeze(contour, axis=1)
        x_array = (contour[hundred_idx, 0].flatten() - contour[:, 0].min()) / (
                contour[:, 0].max() - contour[:, 0].min() + 1)
        y_array = (contour[hundred_idx, 1].flatten() - contour[:, 1].min()) / (
                contour[:, 1].max() - contour[:, 1].min() + 1) * self.aspect_ratio

        return x_array, y_array
