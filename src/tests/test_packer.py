import torch
import random
import unittest

import tensorset as ts

from src.packer import PairPacker


class TestPacker(unittest.TestCase):
    def test_packer(self):
        """
        generate a bunch of random sequence data and metadata,
        append it to the packer, and make sure that the packer
        returns it still associated with the ids that it was
        appended with
        """
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

        packer = PairPacker(
            pack_size_x=pack_size_x,
            pack_size_y=pack_size_y,
            batch_size=batch_size,
            pad_value_dict=pad_value_dict,
        )

        id = 0

        id_to_data = dict()
        id_to_metadata = dict()

        packed_batches = []

        while len(packed_batches) < num_batches:
            # generate some randomly lengthed data
            x_sequence_length = rng.randint(1, pack_size_x)
            x_sample_ids = torch.full((x_sequence_length,), id)
            x_named_columns = {
                name: torch.randn(
                    x_sequence_length, *trailing_shape, generator=torch_rng
                )
                for name, trailing_shape in zip(column_names, trailing_shapes)
            }
            x_named_columns["sample_ids"] = x_sample_ids

            x = ts.TensorSet(**x_named_columns)

            y_sequence_length = rng.randint(1, pack_size_y)
            y_sample_ids = torch.full((y_sequence_length,), id)
            y_named_columns = {
                name: torch.randn(
                    y_sequence_length, *trailing_shape, generator=torch_rng
                )
                for name, trailing_shape in zip(column_names, trailing_shapes)
            }
            y_named_columns["sample_ids"] = y_sample_ids

            y = ts.TensorSet(**y_named_columns)

            metadata = dict(label=rng.randint(0, 2**20))

            # save the ground truth data for later comparison
            id_to_data[id] = (x, y)
            id_to_metadata[id] = metadata

            # add the batch to the packer, and potentially get back a packed batch
            for batch in packer.append(x, y, id, metadata):
                packed_batches.append(batch)

            id += 1

        for tensorset_batch, metadata_batch in packed_batches:
            sample_ids = tensorset_batch["sample_ids"]

            b, s = sample_ids.shape

            self.assertEqual(batch_size, b)
            self.assertEqual(pack_size_x + pack_size_y, s)

            for i in range(batch_size):
                unique_ids = torch.unique(sample_ids[i]).tolist()

                if MASK_SAMPLE_ID in unique_ids:
                    unique_ids.remove(MASK_SAMPLE_ID)

                self.assertGreater(
                    len(unique_ids), 0, "should be at least one sample per batch item"
                )

                for id in unique_ids:
                    self.assertIn(id, metadata_batch[i])

                    self.assertDictEqual(id_to_metadata[id], metadata_batch[i][id])

                    x_sample_ids, y_sample_ids = (
                        sample_ids[i, :pack_size_x],
                        sample_ids[i, pack_size_x:],
                    )
                    x_mask = x_sample_ids == id
                    y_mask = y_sample_ids == id

                    for column_name, column_data in tensorset_batch.iloc[
                        i
                    ].named_columns.items():
                        x_column_data, y_column_data = (
                            column_data[:pack_size_x],
                            column_data[pack_size_x:],
                        )

                        x_sample_data = x_column_data[x_mask]
                        y_sample_data = y_column_data[y_mask]

                        self.assertTrue(
                            torch.allclose(
                                x_sample_data, id_to_data[id][0][column_name]
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                y_sample_data, id_to_data[id][1][column_name]
                            )
                        )
