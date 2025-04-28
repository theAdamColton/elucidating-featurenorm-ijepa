# IJEPA-DepthSmart

Vanilla IJEPA consists of a n layer encoder, and a m layer predictor. The predictor is tasked with predicting
masked patches encdoded from the ema-encoder. This corresponds to IJEPA learning to model deep features that are
useful for powerful semantic tasks such as imagenet classification. However, there is a tradeoff between feature
semanticity and feature granularity. Some downstream tasks such as segmentation or edge modelling require
more granular features than outputted by the final layer of the IJEPA encoder.

IJEPA-DepthSmart attempts to train an encoder to encode image tokens to a sliding scale of semantic granularity.

