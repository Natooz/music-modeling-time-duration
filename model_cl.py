"""
BERT for contrastive learning.
Adapted (simplified) from https://github.com/princeton-nlp/SimCSE
"""


from torch import Tensor, cat, zeros_like, arange
import torch.nn as nn
import torch.distributed as dist

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertPooler
from transformers.modeling_outputs import SequenceClassifierOutput
from constants import TEMPERATURE_CON, POOLER_TYPE_CON


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, config, pooler_type: str):
        super().__init__()
        self.type_ = pooler_type
        assert pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                               "avg_first_last"], "unrecognized pooling type %s" % pooler_type
        self.mlp = BertPooler(config) if pooler_type == "cls" else None

    def forward(self, attention_mask, outputs) -> Tensor:
        last_hidden = outputs.last_hidden_state  # (N,T,E)
        hidden_states = outputs.hidden_states

        if self.type_ == 'cls':
            return self.mlp(last_hidden)  # (N,E)
        elif self.type_ == 'cls_before_pooler':
            return last_hidden[:, 0]
        elif self.type_ == "avg":
            if attention_mask is not None:
                return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            else:
                return last_hidden.sum(1)
        elif self.type_ == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.type_ == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooler_type: str = POOLER_TYPE_CON, temperature: float = TEMPERATURE_CON):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.pooler = Pooler(config, pooler_type)
        self.sim = Similarity(temp=temperature)
        self.post_init()

    def forward(self,
                input_ids=None,
                input_ids2=None,
                attention_mask=None,
                attention_mask2=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                return_dict=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get output hidden states
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler.type_ in ["avg_top2", "avg_first_last"] else False,
            return_dict=True,
        )
        z1 = self.pooler(attention_mask, outputs)  # (N*S,T,E) --> (N*S,E), S=2

        # Inference: Return representation
        if not self.training and labels is None:
            if not return_dict:
                return (outputs[0], z1) + outputs[2:]

            outputs.pooler_output = z1
            return outputs

        # Second pass with different dropout mask
        if input_ids2 is None:
            input_ids2 = input_ids.clone()
            attention_mask2 = attention_mask.clone()
        if attention_mask2 is None:
            attention_mask2 = (input_ids2 != self.config.pad_token_id).int()
        outputs_plus = self.bert(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler.type_ in ["avg_top2", "avg_first_last"] else False,
            return_dict=True,
        )
        z2 = self.pooler(attention_mask, outputs_plus)  # (N*S,T,E) --> (N*S,E), S=2
        if outputs.hidden_states is not None:
            outputs.hidden_states = (cat([h1, h2]) for h1, h2 in zip(outputs.hidden_states, outputs_plus.hidden_states))
        if outputs.attentions is not None:
            outputs.attentions = cat([outputs.attentions, outputs_plus.attentions])

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            z1_list = [zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = cat(z1_list, 0)
            z2 = cat(z2_list, 0)

        # Compute similarities between z1 and z2
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (N,N), rank

        # Compute loss
        if labels is None:
            labels = arange(input_ids.size(0)).long().to(cos_sim.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)  # labels are (N)

        # Evaluation: return loss and representations (not sim)
        logits = cos_sim if self.training else z1

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
