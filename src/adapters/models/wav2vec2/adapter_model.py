import torch.nn as nn

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    WAV_2_VEC_2_START_DOCSTRING,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...wrappers import init


@add_start_docstrings(
    """Wav2Vec2 Model transformer with the option to add flexible adapters within the attention layers.""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2AdapterModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        init(self.wav2vec2)

        self._init_head_modules()

        self.init_weights()
