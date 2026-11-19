"""Inference-only FlashAttention API implemented with PyTorch SDPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__version__ = "sdpa-fallback"


def _finish(output, return_attn_probs):
    return (output, None) if return_attn_probs else output


def _sdpa(q, k, v, *, dropout_p=0.0, softmax_scale=None, causal=False):
    # FlashAttention layout: [batch, sequence, heads, head_dim].
    # FlashAttention natively supports grouped-query attention. PyTorch SDPA
    # requires matching head counts unless GQA is requested explicitly; expand
    # K/V here so this fallback also works on builds/backends without GQA.
    query_heads = int(q.shape[2])
    key_value_heads = int(k.shape[2])
    if query_heads != key_value_heads:
        if query_heads % key_value_heads:
            raise ValueError(
                f"Query heads ({query_heads}) must be divisible by "
                f"key/value heads ({key_value_heads})"
            )
        repeats = query_heads // key_value_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    # FlashAttention >=2.1 uses a bottom-right aligned causal mask when the
    # query is shorter than the cached K/V sequence (incremental decoding).
    # PyTorch SDPA's ``is_causal=True`` mask is top-left aligned, which makes a
    # one-token query attend only to the oldest key. Build the FlashAttention
    # mask explicitly so cached generation remains behaviorally equivalent.
    attention_mask = None
    use_native_causal = bool(causal)
    query_length = int(q.shape[1])
    key_length = int(k.shape[1])
    if causal and query_length != key_length:
        if query_length > key_length:
            raise ValueError(
                f"Causal query length ({query_length}) exceeds key length ({key_length})"
            )
        query_positions = torch.arange(query_length, device=q.device) + (
            key_length - query_length
        )
        key_positions = torch.arange(key_length, device=q.device)
        attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        use_native_causal = False

    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attention_mask,
        dropout_p=float(dropout_p),
        is_causal=use_native_causal,
        scale=softmax_scale,
    )
    return output.transpose(1, 2)


def flash_attn_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
    return_attn_probs=False, **_kwargs,
):
    output = _sdpa(
        q, k, v, dropout_p=dropout_p,
        softmax_scale=softmax_scale, causal=causal,
    )
    return _finish(output, return_attn_probs)


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
    softmax_scale=None, causal=False, return_attn_probs=False, **_kwargs,
):
    del max_seqlen_q, max_seqlen_k
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    outputs = []
    for index in range(len(q_offsets) - 1):
        one_q = q[q_offsets[index]:q_offsets[index + 1]].unsqueeze(0)
        one_k = k[k_offsets[index]:k_offsets[index + 1]].unsqueeze(0)
        one_v = v[k_offsets[index]:k_offsets[index + 1]].unsqueeze(0)
        outputs.append(
            _sdpa(
                one_q, one_k, one_v, dropout_p=dropout_p,
                softmax_scale=softmax_scale, causal=causal,
            ).squeeze(0)
        )
    output = torch.cat(outputs, dim=0)
    return _finish(output, return_attn_probs)


def flash_attn_qkvpacked_func(qkv, **kwargs):
    return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], **kwargs)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, **kwargs):
    return flash_attn_varlen_func(
        qkv[:, 0], qkv[:, 1], qkv[:, 2],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs,
    )
