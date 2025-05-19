import unittest


from src.model import EncoderConfig, IJEPADepthSmartConfig, PredictorConfig
from main import MainConfig, main
from src.transformer_blocks import (
    AttentionConfig,
    TransformerBlockConfig,
    DiffMoeMLPConfig,
)


class TestMain(unittest.TestCase):
    def get_mini_conf(self):
        mlp_config = DiffMoeMLPConfig(embed_dim=16, mlp_ratio=1, num_experts=2)
        attn_config = AttentionConfig(embed_dim=16, head_dim=16)
        block_conf = TransformerBlockConfig(
            mlp_config=mlp_config, attention_config=attn_config
        )
        encoder_conf = EncoderConfig(num_transformer_blocks=1, block_config=block_conf)
        predictor_conf = PredictorConfig(
            input_size=16, num_transformer_blocks=1, block_config=block_conf
        )
        model_conf = IJEPADepthSmartConfig(
            encoder=encoder_conf, predictor=predictor_conf
        )
        conf = MainConfig(
            should_compile=False,
            device="cpu",
            dtype="float32",
            model=model_conf,
            batch_size=8,
            packer_batch_size=4,
            test_mode=True,
            validation_probe_batch_size=32,
            num_workers=0,
        )
        return conf

    def test_train_mode(self):
        conf = self.get_mini_conf()
        conf.mode = "train"
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
