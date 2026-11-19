"""Inference-only PyTorch SDPA fallback for Baichuan's varlen audio attention."""
import torch
import torch.nn.functional as F

__version__ = "sdpa-fallback"


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, causal=False, **_kwargs):
    outputs = []
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    for index in range(len(q_offsets) - 1):
        one_q = q[q_offsets[index]:q_offsets[index + 1]].transpose(0, 1).unsqueeze(0)
        one_k = k[k_offsets[index]:k_offsets[index + 1]].transpose(0, 1).unsqueeze(0)
        one_v = v[k_offsets[index]:k_offsets[index + 1]].transpose(0, 1).unsqueeze(0)
        one_out = F.scaled_dot_product_attention(one_q, one_k, one_v, is_causal=causal)
        outputs.append(one_out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)

