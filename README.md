# Elucidating the Role of Feature Normalization in IJEPA

[\[Paper\]](Elucidating_the_Role_of_Feature_Normalization_in_IJEPA.pdf)

<img src="https://github.com/user-attachments/assets/a0c713bb-2982-4863-a24a-81347181f801" width="450" />
<img src="https://github.com/user-attachments/assets/060b492d-1818-4473-8555-c6d54ec9c4c9" width="300" />
<img src="https://github.com/user-attachments/assets/87add32f-c54c-4f5a-a6bd-c3ce7b5184f8" width="300" />
<img src="https://github.com/user-attachments/assets/904fb2b9-17bf-4781-a22b-5265fb0cd42b" width="400" />


# How to run our code and reproduce our results


We use `uv` for dependency management.


Download the training datasets and NYU-Depth tar files:
`uv run download_dataset.py`

This requires roughly 100GB of storage space.


Run the default training configuration which trains a ~300m parameter ViT-Small with a patch size of 16 and a batch size of 320. This consumes ~22GB of VRAM and takes 116 hours (assuming validation logging is turned off):
`uv run main.py --config conf/small.yaml`

Or resume a training run:
`uv run main.py --config /path/to/checkpoint/config.yaml --conf.resume_checkpoint_path /path/to/checkpoint/checkpointfile.pt`

Or evaluate the IN1k validation performance of a pretrained model:
`uv run main.py --config /path/to/checkpoint/config.yaml --conf.resume_checkpoint_path /path/to/checkpoint/checkpointfile.pt --conf.mode validate`

Or visualize features of a pretrained model:
`uv run main.py --config /path/to/checkpoint/config.yaml --conf.resume_checkpoint_path /path/to/checkpoint/checkpointfile.pt --conf.mode visualize-embeddings`

Or plot the losses of a pretrained model:
`uv run main.py --config /path/to/checkpoint/config.yaml --conf.resume_checkpoint_path /path/to/checkpoint/checkpointfile.pt --conf.mode plot-sample-losses`

Run tests:
`uv run python -m unittest`



# Gotchas

The code refers to `token_ids` this is a LongTensor that contains 4 integers for each token: register id, sample id, height id, width id.
Register ID refers to the index of the register, if this patch is a register and does not contain image data, or a MASK_TOKEN_ID.
Sample ID refers to the unique index of the sample that this patch/register comes from.
Height ID refers to the index of this patch into the patched image, or MASK_TOKEN_ID if this token is a register.
Width ID refers to the index of this patch into the patched image, or MASK_TOKEN_ID if this token is a register.

We need to keep track of these IDs because unlike most ViT models, our model processes one or more samples per batch element. Our model processes batches that contains patches from many images of varied resolution.


Unlike many transformer model's, pytorch's eval mode will effect our model's forward. Calling `eval()` will
cause the DiffMOEMLP layers to use dynamic allocation causing the number of allocated experts to be determined
by the capacity predictor. Make sure to call `model.eval()` before doing any evaluation. For training, use train mode.

Our LiDAR score is computed from a random subset of the training data. This subset is random, so if you resume a run you may observe a change in the LiDAR score.

# Performance Tips

* This code only supports single-gpu training.

* You can optionally install PILLOW-SIMD for a boost in dataloading speed. 

* You should probably disable LiDAR score logging if you have limited system RAM.

# Hidden features

* You can enable TOME for the encoder and predictor. We only tested this breifly and observed a distinct performance
decline.

* You can use absolute factorized learnable position embeddings instead of ROPE2D. In a short test we found this decreases performance very slightly

* The predictor can be trained without token dropping and without batch repeat. We found this drastically decreases downstream performance. 

* You can add register tokens to the encoder and to the predictor. The encoder's register tokens can be passed unchanged to the predictor, or be wiped. We found that adding 8 register tokens dramatically reduced downstream performance and leave it as an open problem as to why register tokens decrease performance by so much.

* You can choose a feature normalization mode other than LN and DynTanh. We have batchnorm, disabled, and running batchnorm. 



