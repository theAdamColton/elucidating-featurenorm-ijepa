import torch
import einx
import torchvision
import matplotlib.pyplot as plt
import tensorset as ts
from src.dataset import MASK_SAMPLE_ID
from src.utils import get_viz_output_path


def prepare_context_target_batch(batch, device, dtype):
    packed_batch, *_ = batch

    if not isinstance(packed_batch, ts.TensorSet):
        raise ValueError()

    position_ids = packed_batch.named_columns.pop("position_ids")
    sample_ids = packed_batch.named_columns.pop("sample_ids")
    # Token ids contains along the channel dim (sample_ids, register id, height idx, width idx)
    token_ids = torch.cat((sample_ids.unsqueeze(-1), position_ids), -1)

    patches = packed_batch.named_columns.pop("patches")

    patches = patches.to(device=device, dtype=dtype, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)

    # Scale from [0,255] to [-1,1]
    patches = (patches / 255) * 2 - 1

    return patches, token_ids


def plot_sample_losses(
    dataloader,
    context_sequence_length,
    model,
    device,
    dtype,
    patch_size,
    num_image_channels,
    autocast_fn,
):
    """
    Plot the distribution of losses for each sample in a batch.
    """
    batch = next(iter(dataloader))

    patches, token_ids = prepare_context_target_batch(batch, device, dtype)

    x_patches = patches[:, :context_sequence_length]
    y_patches = patches[:, context_sequence_length:]

    x_token_ids = token_ids[:, :context_sequence_length]
    y_token_ids = token_ids[:, context_sequence_length:]

    b, x_patches_length, d = x_patches.shape

    sample_ids = torch.cat((x_token_ids, y_token_ids), 1)[..., 0]

    batch_unique_sample_ids = []
    for i in range(b):
        unique_sample_ids = torch.unique(sample_ids[i]).tolist()
        if MASK_SAMPLE_ID in unique_sample_ids:
            unique_sample_ids.remove(MASK_SAMPLE_ID)
        batch_unique_sample_ids.append(unique_sample_ids)

    sample_losses = [
        torch.zeros(len(unique_sample_ids))
        for unique_sample_ids in batch_unique_sample_ids
    ]
    sample_losses_variance = [
        torch.zeros(len(unique_sample_ids))
        for unique_sample_ids in batch_unique_sample_ids
    ]

    # Compute the loss several times, each time using a different context and target
    iters = 1

    for i in range(iters):
        all_patches = torch.cat((x_patches, y_patches), 1)
        all_token_ids = torch.cat((x_token_ids, y_token_ids), 1)

        # Shuffle the patches, to create new random sets of context
        # and target tokens
        # Note, that this isn't windowed masking like in the
        # official context-target dataset

        b, s, d = all_patches.shape
        indices = torch.rand(b, s).argsort(dim=-1)
        all_patches = einx.get_at("b [s] d, b n -> b n d", all_patches, indices)
        all_token_ids = einx.get_at("b [s] nd, b n -> b n nd", all_token_ids, indices)

        x_patches, y_patches = (
            all_patches[:, :x_patches_length],
            all_patches[:, x_patches_length:],
        )
        x_token_ids, y_token_ids = (
            all_token_ids[:, :x_patches_length],
            all_token_ids[:, x_patches_length:],
        )

        patches = torch.cat((x_patches, y_patches), 1)
        token_ids = torch.cat((x_token_ids, y_token_ids), 1)

        with torch.inference_mode():
            with autocast_fn():
                result_dict = model(
                    patches,
                    token_ids,
                    context_sequence_length=context_sequence_length,
                    return_tokenwise_loss=True,
                    return_predictor_target_token_ids=True,
                )

        # tokenwise loss is the batch repeated loss
        # from the predictor
        tokenwise_loss = result_dict["tokenwise_loss"].cpu().float()
        # target sample_ids is batch repeated sequence ids fed to the predictor
        target_sample_ids = result_dict["predictor_target_token_ids"][..., 0].cpu()

        tokenwise_loss = einx.mean("rb ys [d]", tokenwise_loss)

        # Compute the mean loss for each unique sample across the batch
        for i in range(model.config.predictor_batch_repeat):
            for j in range(b):
                batch_index = i * b + j
                sample_ids = target_sample_ids[batch_index]
                for k, sample_id in enumerate(batch_unique_sample_ids[j]):
                    mask = sample_ids == sample_id
                    if not mask.any():
                        continue

                    # Measure the mean sample loss, and the variance of the sample loss
                    sample_loss = tokenwise_loss[batch_index, mask].mean()
                    sample_loss_variance = tokenwise_loss[batch_index, mask].var()

                    # Take the mean of sample losses across iterations
                    # TODO! This doesnt handle the special case
                    # where sometimes a sample might not be included
                    # in the loss for a batch because it is randomly dropped
                    sample_losses[j][k] += sample_loss / iters
                    sample_losses_variance[j][k] += sample_loss_variance / iters

    # Save an image for each sample
    output_path = get_viz_output_path()

    all_patches = all_patches.cpu().float()
    all_token_ids = all_token_ids.cpu()
    for i in range(b):
        for j, sample_id in enumerate(batch_unique_sample_ids[i]):
            mask = all_token_ids[i, :, 0] == sample_id
            if not mask.any():
                continue
            sample_patches = all_patches[i, mask]
            sample_ids = all_token_ids[i, mask]

            sample_patches = (sample_patches + 1) / 2

            sample_position_ids = sample_ids[:, -2:]
            nph, npw = (sample_position_ids.amax(0) + 1).tolist()
            unpacked_patches = torch.zeros(nph, npw, d)

            for (hid, wid), patch in zip(sample_position_ids, sample_patches):
                unpacked_patches[hid, wid] = patch

            image = einx.rearrange(
                "nph npw (ph pw c) -> c (nph ph) (npw pw)",
                unpacked_patches,
                ph=patch_size,
                pw=patch_size,
                c=num_image_channels,
            )
            image = (image.clip(0, 1) * 255).to(torch.uint8)

            sample_loss = sample_losses[i][j].item()
            sample_loss = round(sample_loss, 5)

            sample_loss_variance = sample_losses_variance[i][j].item()
            sample_loss_variance = round(sample_loss_variance, 5)

            image_save_path = (
                output_path
                / f"sample-{i:04}-{sample_id:08} loss {sample_loss} variance {sample_loss_variance}.png"
            )

            torchvision.io.write_png(image, str(image_save_path))

            print("wrote to ", image_save_path)

    # Plot the distribution of the losses
    all_losses = []
    for batch in sample_losses:
        all_losses.extend(batch.tolist())

    all_losses_variance = []
    for batch in sample_losses_variance:
        all_losses_variance.extend(batch.tolist())

    plt.hist(all_losses, bins=20, density=True, color="skyblue", edgecolor="black")
    plt.xlabel("sample loss")
    plt.ylabel("frequency")
    plot_save_path = output_path / "histogram.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        all_losses,
        bins=20,
        cumulative=True,
        density=True,
        color="skyblue",
    )
    plt.xlabel("sample loss")
    plt.ylabel("frequency")
    plot_save_path = output_path / "cum_histogram.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        all_losses_variance,
        bins=20,
        density=True,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("sample loss variance")
    plt.ylabel("frequency")
    plot_save_path = output_path / "histogram_variance.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        all_losses_variance,
        bins=20,
        cumulative=True,
        density=True,
        color="skyblue",
    )
    plt.xlabel("sample loss variance")
    plt.ylabel("frequency")
    plot_save_path = output_path / "cum_histogram_variance.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()
