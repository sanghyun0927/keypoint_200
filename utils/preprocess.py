import logging
from typing import Union, Tuple

import numpy as np
import cv2
from PIL import Image
import xgboost as xgb
from rembg.session_base import BaseSession


def rescale_and_paste(
        old_foreground: Image.Image,
        old_background: Image.Image,
        only_rescale: bool,
        fore_x_ratio=0.5,
        fore_y_ratio=0.75,
        fore_scale=0.7,
) -> Union[Tuple[Image.Image, Image.Image], Image.Image]:
    """
    Rescale and paste the foreground image onto the background image.

    Args:
        old_foreground (Image.Image): The original foreground image.
        old_background (Image.Image): The original background image.
        only_rescale (bool): Whether to return only the rescaled foreground image.
        fore_x_ratio (float): TODO
        fore_y_ratio (float): TODO
        fore_scale (float): TODO
    Returns:
        Union[Tuple[Image.Image, Image.Image], Image.Image]: Either a tuple containing the
        resulting image and the new foreground image, or only the new foreground image.
    """

    # Define scale and resized image size
    if old_foreground.width > old_background.width:
        new_fore_width = old_background.width * fore_scale
        new_fore_height = old_foreground.height * new_fore_width / old_foreground.width

        new_width = np.rint(new_fore_width).astype("uint16")
        new_height = np.rint(new_fore_height).astype("uint16")

        # Resizing
        new_foreground = old_foreground.resize((new_width, new_height))
        new_background = old_background
    else:
        new_back_width = old_foreground.width / fore_scale
        new_back_height = old_background.height * new_back_width / old_background.width

        new_width = np.rint(new_back_width).astype("uint16")
        new_height = np.rint(new_back_height).astype("uint16")

        # Resizing
        new_foreground = old_foreground
        new_background = old_background.resize((new_width, new_height))

    # Foreground(차량)의 높이가 background(배경)의 높이를 초과하는 경우 에러를 발생시킴
    if new_foreground.height > new_background.height:
        scale_factor = new_foreground.height / new_foreground.height
        size = (int(new_background.width * scale_factor), int(new_background.height * scale_factor))
        new_background.resize(size)

        logger = logging.getLogger()
        logger.info("입력 이미지의 세로 길이가 제한 공간을 초과 하였습니다. "
                    "배경 이미지의 크기를 입력 이미지의 세로 길이에 맞게 조정합니다.")

    # Define the location of the foreground
    y_center = int(new_background.height * fore_y_ratio)
    x_center = int(new_background.width * fore_x_ratio)
    y = y_center - new_foreground.height // 2
    x = x_center - new_foreground.width // 2

    # 새 alpha 채널 직사각형 이미지 생성
    new_image = Image.new(mode="RGBA", size=new_background.size)

    # 차량 이미지 하단이 배경 이미지 아래로 위치하여 위치 조정
    if y + new_foreground.height > new_background.height:
        logger = logging.getLogger()
        logger.info("차량 이미지 하단이 배경 이미지 아래로 위치하여 위치 조정")
        y = new_background.height - new_foreground.height

    # Paste foreground on transparent background
    new_image.paste(new_foreground, (x, y))

    # rescale만 하는 경우, 크기가 조정된 결과물만 반환한다.
    if only_rescale:
        return new_image
    else:
        # Paste foreground onto background
        result = Image.alpha_composite(new_background, new_image)
        return result, new_image


def stroke_mask(
        img_array: np.array, mask_thickness: int
) -> (Image.Image, Image.Image, Image.Image):
    """
    세그멘테이션된 차량 이미지 외곽에 매우 굵은 윤곽선을 생성하여 Stable diffusion inpaint 모델의 마스크로 사용함

    Args:
        img_array (np.array): Input image as numpy array of shape (height, width, 4).
        mask_thickness (int): Padding thickness for the mask.

    Returns:
        padded_img (PIL.Image.Image): Input image with added padding.
        result (PIL.Image.Image): Mask image with thick stroke contour, without background.
        alpha (PIL.Image.Image): Thresholded alpha channel of the input image.
    """

    # Add padding to the image
    padding = mask_thickness
    alpha = img_array[:, :, 3]
    rgb_img = img_array[:, :, 0:3]
    padded_img = cv2.copyMakeBorder(
        rgb_img,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )
    alpha = cv2.copyMakeBorder(
        alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )

    # Merge the padded image and alpha channel
    padded_img = cv2.merge((padded_img, alpha))

    # Apply threshold to the alpha channel
    _, alpha_without_shadow = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    # Calculate distance transform
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(
        alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3
    )  # dist l1 : L1 , dist l2 : l2

    # Modify distance matrix based on mask size
    masked = change_matrix(dist, mask_thickness)
    masked_alpha = (masked * 255).astype(np.uint8)

    # Create stroke image
    h, w, _ = padded_img.shape
    colors = (255, 255, 255)
    stroke_b = np.full((h, w), colors[2], np.uint8)
    stroke_g = np.full((h, w), colors[1], np.uint8)
    stroke_r = np.full((h, w), colors[0], np.uint8)
    mask = cv2.merge((stroke_r, stroke_g, stroke_b, masked_alpha))

    # Modify distance matrix based on contour size
    mask = Image.fromarray(mask)
    padded_img = Image.fromarray(padded_img)

    # Combine stroke and padded_img using alpha composite
    result = Image.alpha_composite(mask, padded_img)
    alpha = Image.fromarray(alpha_without_shadow)

    return padded_img, result, alpha


def change_matrix(input_mat: np.ndarray, stroke_size: int) -> np.ndarray:
    """
    Modify input matrix based on the stroke size.

    Args:
        input_mat (np.array): Input matrix to be modified.
        stroke_size (int): Width of the stroke in pixels.

    Returns:
        mat (np.array): Modified matrix.
    """

    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)
    return mat


