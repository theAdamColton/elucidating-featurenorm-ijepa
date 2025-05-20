from typing import Any
import torch
import tensorset as ts


def pad_tensorsequence_to_length(
    sequence: list[ts.TensorSet],
    sequence_length: int,
    pad_value_dict=dict(),
) -> ts.TensorSet:
    sequence = ts.cat(sequence, 0)

    # if the sequence length is too short, pads
    pad_amt = sequence_length - sequence.size(0)
    needs_pad = pad_amt > 0
    if needs_pad:
        sequence = sequence.pad(pad_amt, 0, value_dict=pad_value_dict)

    return sequence


class PairPacker:
    """
    A Packer that can pack sequences pairs into batches.
    Produces pairs of batches. Between pairs of batches,
    sequence pairs will be placed in the same batch element
    and have the same sequence id.
    """

    def __init__(
        self,
        pack_size_x: int,
        pack_size_y: int,
        batch_size: int,
        pad_value_dict: dict[str, Any] = dict(),
    ):
        self.pack_size_x = pack_size_x
        self.pack_size_y = pack_size_y
        self.batch_size = batch_size

        self.pad_value_dict = pad_value_dict

        self._init_or_reset_buffers()

    def _init_or_reset_buffers(self):
        self.batch_buffer_size_x = [0 for _ in range(self.batch_size)]
        self.batch_buffer_size_y = [0 for _ in range(self.batch_size)]

        if not hasattr(self, "batch_buffer"):
            # Can't init the batch buffer before we append the first x,y pair
            # because we don't know what the size is.
            self.batch_buffer: dict[str, torch.Tensor] | None = None
        else:
            for column_name, tensor in self.batch_buffer.items():
                pad_value = self.pad_value_dict.get(column_name)
                if pad_value is None:
                    raise ValueError(
                        f"{column_name} does not have a pad_value in the pad_value_dict!"
                    )

                tensor.fill_(pad_value)

        self.metadata_buffer: list[dict[int, dict]] = [
            dict() for _ in range(self.batch_size)
        ]

    def _add_to_batch_buffer(self, x: ts.TensorSet, y: ts.TensorSet, i: int):
        """
        Puts x and y into the batch_buffer at the ith position
        Also updates the batch buffer sizes
        """
        x_column_names = set(x.named_columns.keys())
        y_column_names = set(y.named_columns.keys())

        if len(x_column_names.symmetric_difference(y_column_names)) > 0:
            raise ValueError(
                "x TensorSet must have the same named columns as y TensorSet!"
            )

        if self.batch_buffer is None:
            # Initialize the batch buffer
            batch_buffer = dict()

            for column_name in x_column_names:
                x_tensor = x.named_columns[column_name]
                y_tensor = y.named_columns[column_name]

                if x_tensor.dtype != y_tensor.dtype:
                    raise ValueError("dtypes across named x,y columns should match")
                if x_tensor.device != y_tensor.device:
                    raise ValueError("devices across named x,y columns should match")
                if x_tensor.shape[1:] != y_tensor.shape[1:]:
                    raise ValueError(
                        "trailing shapes across named x,y columns should match"
                    )

                device, dtype = x_tensor.device, x_tensor.dtype
                # The shape, sliced after the sequence dimension
                trailing_shape = x_tensor.shape[1:]

                buffer_shape = (
                    self.batch_size,
                    self.pack_size_x + self.pack_size_y,
                    *trailing_shape,
                )
                buffer_value = self.pad_value_dict.get(column_name)
                if buffer_value is None:
                    raise ValueError(f"Can't initialize the buffer of {column_name} when it doesn't have a value in \
                                     the pad_value_dict!")
                buffer = torch.full(
                    buffer_shape, buffer_value, device=device, dtype=dtype
                )
                batch_buffer[column_name] = buffer

            self.batch_buffer = batch_buffer

        # Add x and y to the buffer
        buffer_size_x = self.batch_buffer_size_x[i]
        buffer_size_y = self.batch_buffer_size_y[i]

        x_sequence_length: int = x.size(0)
        y_sequence_length: int = y.size(0)

        for name in x_column_names:
            buffer = self.batch_buffer[name][i]

            x_tensor = x.named_columns[name]
            x_start_idx = buffer_size_x
            buffer[x_start_idx : x_start_idx + x_sequence_length] = x_tensor

            y_tensor = y.named_columns[name]
            y_start_idx = self.pack_size_x + buffer_size_y
            buffer[y_start_idx : y_start_idx + y_sequence_length] = y_tensor

        self.batch_buffer_size_x[i] += x_sequence_length
        self.batch_buffer_size_y[i] += y_sequence_length

    def append(
        self, x: ts.TensorSet, y: ts.TensorSet, id: int, metadata: dict = dict()
    ):
        if len(x.columns) > 0 or len(y.columns) > 0:
            raise ValueError("Can't append TensorSet with unnamed columns!")

        sequence_length_x: int = x.size(0)
        sequence_length_y: int = y.size(0)

        if sequence_length_x > self.pack_size_x:
            raise ValueError(
                f"{sequence_length_x} is greater than pack size {self.pack_size_x}"
            )
        if sequence_length_y > self.pack_size_y:
            raise ValueError(
                f"{sequence_length_y} is greater than pack size {self.pack_size_y}"
            )

        can_both_fit = False
        for i in range(self.batch_size):
            buffer_size_x = self.batch_buffer_size_x[i]
            buffer_size_y = self.batch_buffer_size_y[i]

            can_fit_x = buffer_size_x + sequence_length_x <= self.pack_size_x
            can_fit_y = buffer_size_y + sequence_length_y <= self.pack_size_y

            can_both_fit = can_fit_x and can_fit_y
            if can_both_fit:
                self.metadata_buffer[i][id] = metadata
                self._add_to_batch_buffer(x, y, i)
                break

        if not can_both_fit:
            packed_batch = {k: v.clone() for k, v in self.batch_buffer.items()}
            packed_batch = ts.TensorSet(**packed_batch)
            yield (packed_batch, self.metadata_buffer)

            self._init_or_reset_buffers()

            self.append(x, y, id, metadata)
