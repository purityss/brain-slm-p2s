"""Subset of flash_attn.bert_padding required by Kimi-Audio."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def index_first_axis(values, indices):
    return values[indices]


def unpad_input(hidden_states, attention_mask):
    batch_size, sequence_length = attention_mask.shape
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    flattened = hidden_states.reshape(batch_size * sequence_length, *hidden_states.shape[2:])
    values = flattened[indices]
    lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    cumulative = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    return values, indices, cumulative, int(lengths.max().item())


def pad_input(hidden_states, indices, batch_size, sequence_length):
    shape = (batch_size * sequence_length, *hidden_states.shape[1:])
    output = hidden_states.new_zeros(shape)
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch_size, sequence_length, *hidden_states.shape[1:])
