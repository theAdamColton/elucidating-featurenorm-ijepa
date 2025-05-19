import unittest


from src.model import IJEPADepthSmartConfig
from main import MainConfig, main


class TestMain(unittest.TestCase):
    def get_mini_conf(self):
        model_conf = IJEPADepthSmartConfig()
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
