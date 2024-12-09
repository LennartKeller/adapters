import torch
from torch import nn

from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import _HIDDEN_STATES_START_POSITION

from .base import PredictionHead


class CTCHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        hidden_size=None,
        final_dropout=None,
        blank=None,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
    ):
        super(CTCHead, self).__init__(head_name)
        self.config = {
            "head_type": "ctc",
            "vocab_size": vocab_size or model.config.vocab_size,
            "hidden_size": hidden_size or model.config.hidden_size,
            "final_dropout": final_dropout or model.config.final_dropout,
            "blank": blank or model.config.pad_token_id,
            "ctc_loss_reduction": ctc_loss_reduction or model.config.ctc_loss_reduction,
            "ctc_zero_infinity": ctc_zero_infinity or model.config.ctc_zero_infinity,
        }
        self.build(model)

    def build(self, model):
        model_config = model.config
        hidden_size = self.config.get("hidden_size", model_config.hidden_size)
        vocab_size = self.config.get("vocab_size", model_config.vocab_size)
        lm_head = nn.Linear(hidden_size, vocab_size)
        self.add_module("lm_head", lm_head)
        self.train(model.training)

    def forward(self, outputs, attention_mask, return_dict=True, **kwargs):

        # Copied from 'src/transformers/models/wav2vec2/modeling_wav2vec2.py'
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        labels = kwargs.pop("labels", None)
        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            if attention_mask is None:
                raise ValueError(
                    "The custom 'CTCHead' provided by `adapters` does not support inferring the attention_mask on the fly. "
                    "You have to pass one upfront while computing a loss."
                )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
