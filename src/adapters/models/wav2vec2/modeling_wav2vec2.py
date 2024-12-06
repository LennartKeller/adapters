from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer

from ...composition import adjust_tensors_for_parallel, adjust_tensors_for_parallel_, match_attn_matrices_for_parallel
from .mixin_wav2vec2 import Wav2Vec2EncoderLayerMixin


class Wav2Vec2EncoderLayerWithAdapters(Wav2Vec2EncoderLayerMixin, Wav2Vec2EncoderLayer):

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        adjust_tensors_for_parallel_(hidden_states, attention_mask)
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        # hidden_states = attn_residual + hidden_states

        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.attention_adapters(hidden_states, attn_residual, self.layer_norm)

        # hidden_states = hidden_states + self.feed_forward(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.output_adapters(hidden_states, attn_residual, self.final_layer_norm)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
