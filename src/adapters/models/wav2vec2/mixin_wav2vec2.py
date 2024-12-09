from typing import Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin, ModelBaseAdaptersMixin
from ...utils import patch_forward


class Wav2Vec2EncoderLayerMixin:
    def init_adapters(self, model_config, adapters_config):
        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")
        patch_forward(self)


# class Wav2Vec2TransformerAdaptersMixin:
#     """Adds adapters to the Transformer module of Wav2Vec2."""
#
#     def init_adapters(self, model_config, adapters_config):
#         patch_forward(self)
#
#     def forward(self, *args, **kwargs):
#         if hasattr(self, "pre_forward_fn"):
#             kwargs["x"] = self.pre_forward_fn(self, kwargs["x"])
#         return super().forward(*args, **kwargs)


class Wav2Vec2ModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the Wav2Vec2 module."""

    support_prompt_tuning = False
    support_lora_delta_w_svd = False

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config, add_prefix_tuning_pool=False)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layers):
            yield i, layer


class Wav2Vec2ForCTCMixin(Wav2Vec2ModelAdaptersMixin):
    """Adds adapters to the Wav2Vec2ForCTC module."""

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
