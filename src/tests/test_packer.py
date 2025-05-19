import torch
import random
import unittest

import tensorset as ts

from src.packer import PairPacker


class TestPacker(unittest.TestCase):
    def test_packer(self):
        rng = random.Random()
        torch_rng = torch.Generator()

        num_batches = 20
        batch_size = 8
        pack_size_x = 64
        pack_size_y = 64

        column_names = ["rgb", "depth", "features"]
        trailing_shapes = [(3,), (8,), (4, 3)]
        pad_value_dict = {name: 0 for name in column_names}
        MASK_SAMPLE_ID = -100
        pad_value_dict["sample_ids"] = MASK_SAMPLE_ID

        batches = []

        packer = PairPacker(
            pack_size_x=pack_size_x,
            pack_size_y=pack_size_y,
            batch_size=batch_size,
            pad_value_dict=pad_value_dict,
        )

        id = 0

        id_to_data = dict()
        id_to_metadata = dict()

        while len(batches) < num_batches:
            x_sequence_length = rng.randint(1, pack_size_x)
            x_sample_ids = torch.full((x_sequence_length,), id)
            x_named_columns = {
                name: torch.randn(x_sequence_length, *trailing_shape)
                for name, trailing_shape in zip(column_names, trailing_shapes)
            }
            x_named_columns["sample_ids"] = x_sample_ids

            x = ts.TensorSet(**x_named_columns)

            y_sequence_length = rng.randint(1, pack_size_y)
            y_sample_ids = torch.full((y_sequence_length,), id)
            y_named_columns = {
                name: torch.randn(y_sequence_length, *trailing_shape)
                for name, trailing_shape in zip(column_names, trailing_shapes)
            }
            y_named_columns["sample_ids"] = y_sample_ids

            y = ts.TensorSet(**y_named_columns)

            metadata = dict(label=rng.randint(0, 2**20))

            id_to_data[id] = (x, y)
            id_to_metadata[id] = metadata

            for batch in packer.append(x, y, id, metadata):
                batches.append(batch)

        import bpdb

        bpdb.set_trace()
