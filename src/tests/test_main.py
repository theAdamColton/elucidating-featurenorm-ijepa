import unittest


from src.dataset import ContextTargetDatasetConfig
from src.model import EncoderConfig, IJEPAConfig, PredictorConfig
from main import MainConfig, main
from src.transformer_blocks import (
    AttentionConfig,
    TransformerBlockConfig,
)


class TestMain(unittest.TestCase):
    def get_mini_conf(self):
        attn_config = AttentionConfig(embed_dim=16, head_dim=16)
        block_conf = TransformerBlockConfig(attention_config=attn_config, embed_dim=16)
        encoder_conf = EncoderConfig(num_transformer_blocks=1, block_config=block_conf)
        predictor_conf = PredictorConfig(
            input_size=16,
            num_transformer_blocks=1,
            block_config=block_conf,
        )
        model_conf = IJEPAConfig(
            encoder=encoder_conf,
            predictor=predictor_conf,
        )
        context_target_dataset = ContextTargetDatasetConfig(packer_batch_size=2)
        conf = MainConfig(
            should_compile=False,
            context_target_dataset=context_target_dataset,
            device="cpu",
            dtype="float32",
            model=model_conf,
            validation_monocular_depth_feature_depth=-1,
            batch_size=8,
            test_mode=True,
            validation_probe_batch_size=32,
            num_workers=0,
            lidar_num_unique_samples=20,
            lidar_num_augmentations=10,
            visualize_features_depth=-1,
        )

        return conf

    def test_train_mode(self):
        conf = self.get_mini_conf()
        conf.mode = "train"
        main(conf)

    def test_train_mode_diffmoe(self):
        conf = self.get_mini_conf()
        conf.mode = "train"
        conf.model.encoder.block_config.mlp_mode = "diffmoe"
        main(conf)

    def test_train_mode_tome(self):
        conf = self.get_mini_conf()
        conf.mode = "train"
        conf.model.encoder.tome_merge_rate = 0.05
        main(conf)

    def test_validate_mode(self):
        conf = self.get_mini_conf()
        conf.mode = "validate"
        main(conf)

    def test_validate_monocular_depth(self):
        conf = self.get_mini_conf()
        conf.mode = "validate-monocular-depth"
        main(conf)

    def test_make_viz_mode(self):
        conf = self.get_mini_conf()
        conf.mode = "make-viz"
        main(conf)

    def test_visualize_embeddings_mode(self):
        conf = self.get_mini_conf()
        conf.mode = "visualize-embeddings"
        main(conf)

    def test_plot_sample_losses_mode(self):
        conf = self.get_mini_conf()
        conf.mode = "plot-sample-losses"
        main(conf)
