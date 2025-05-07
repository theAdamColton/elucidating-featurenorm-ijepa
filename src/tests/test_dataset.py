import unittest

import torch
import numpy as np
from PIL import Image

from src.dataset import RandomImageResizer


def make_random_pil_image(h, w):
    image_arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    image = Image.fromarray(image_arr)
    return image


class DatasetTests(unittest.TestCase):
    def test_random_image_resizer_can_make_full_size(self):
        """
        Tests that the random resizer will sometimes resize images
        at max_side_length. This is important because during validation,
        the images are cropped to max_side_length.
        """
        h, w = 256, 256
        image = make_random_pil_image(h, w)
        rz = RandomImageResizer(
            min_side_length=248,
            max_side_length=256,
            multiple_of=8,
            max_num_pixels=h * w,
        )

        trials = 20
        trial = 0

        while True:
            self.assertLess(trial, trials, "ran out of trials")
            pixel_values = rz({"pixel_values": image})["pixel_values"]
            nh, nw, c = pixel_values.shape
            if nh == h and nw == w:
                break
