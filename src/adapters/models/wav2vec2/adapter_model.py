import torch.nn as nn

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    WAV_2_VEC_2_START_DOCSTRING,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...heads.model_mixin import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import ModelWithHeadsAdaptersMixin
from ...wrappers import init


@add_start_docstrings(
    """Wav2Vec2 Model transformer with the option to add flexible adapters within the attention layers.""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2AdapterModel(ModelWithHeadsAdaptersMixin, Wav2Vec2PreTrainedModel):

    # TODO Do I have to understand why this is necesarry?
    _convert_to_flex_head = False

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        init(self.wav2vec2)

        self.init_weights()

    def iter_layers(self):
        yield from self.wav2vec2.iter_layers()

    @property
    def adapters_config(self):
        return self.wav2vec2.adapters_config

    def forward(self, *args, **kwargs):
        return Wav2Vec2Model.forward(self, *args, **kwargs)
