from pathlib import Path
import unittest
import random

import einx
import torch
import numpy as np
from PIL import Image

from main_conf import MainConfig
from src.dataset import (
    MASK_SAMPLE_ID,
    ContextTargetDatasetConfig,
    PILImageResizer,
    RandomImageResizer,
    TorchImageResizer,
    get_context_target_dataloader,
)


def make_random_pil_image(h, w):
    image_arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    image = Image.fromarray(image_arr)
    return image


class DatasetTests(unittest.TestCase):
    def test_torch_image_resizer_matches_pil_image_resizer(self):
        """
        Tests that the torch image resizer has the same behavior
        as the pil image resizer
        """

        rng = random.Random()
        sample_image_folder_path = Path("sample_images")
        for image_path in sample_image_folder_path.iterdir():
            image_pil = Image.open(str(image_path))
            size = rng.randint(64, 384)

            # image_pil = make_random_pil_image(h, w)
            image_pt = torch.from_numpy(np.array(image_pil))
            image_pt = einx.rearrange("h w c -> c h w", image_pt)

            resized_pil = PILImageResizer(size)({"pixel_values": image_pil})[
                "pixel_values"
            ]
            resized_pt = TorchImageResizer(size)(image_pt)

            resized_pil = einx.rearrange("h w c -> c h w", resized_pil)

            self.assertTrue(
                torch.allclose(
                    resized_pil / 255, resized_pt / 255, atol=1e-1, rtol=1e-2
                )
            )

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

    def test_dataset_no_patch_repeats(self):
        config = ContextTargetDatasetConfig(packer_batch_size=8)
        main_config = MainConfig()

        dataloader = get_context_target_dataloader(
            config=config,
            dataset_pattern=main_config.train_dataset_pattern,
            shuffle_size_samples=100,
            num_workers=2,
        )

        batch, *_ = next(iter(dataloader))

        b = batch.size(0)

        sample_ids = batch["sample_ids"]
        position_ids = batch["position_ids"][..., -2:]

        for i in range(b):
            element_sample_ids = sample_ids[i]
            element_position_ids = position_ids[i]
            unique_ids = torch.unique(element_sample_ids).tolist()
            if MASK_SAMPLE_ID in unique_ids:
                unique_ids.remove(MASK_SAMPLE_ID)

            for id in unique_ids:
                mask = element_sample_ids == id

                sample_positions = element_position_ids[mask]

                seen_positions = set()

                for position in sample_positions:
                    position = (int(position[0]), int(position[1]))
                    self.assertNotIn(
                        position, seen_positions, f"{id} has duplicated patches!"
                    )
                    seen_positions.add(position)
