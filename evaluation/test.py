import torch
import torch.nn as nn
import math

import triton

import numpy as np
import bit_decode_cuda as bit_decode_cuda
from bit_decode import kvcache_pack_int, fwd_kvcache_int
from bit_decode import DynamicCache


def attention_ref(
    q,
    k,
    v,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype

    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    
    attention = torch.softmax(scores, dim=-1).to(v.dtype)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


# Quantization parameters
quant_mode = "k-channel"
num_bits = 4
pack_nums = 16 / num_bits
group_size = 32
residual_block_size = 128

device = "cuda"
dtype = torch.float16

layer_idx = 0
batch_size = 1
nheads = 32
nheads_k = 32
d = 128

seqlen_q = 1
seqlen_k = 1024
sm_scale = 1.0 / math.sqrt(d)


####### Round 1 : Prefill #######
torch.manual_seed(42)

q = torch.rand(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
k_state = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
v_state = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)

residual_len = seqlen_k % residual_block_size
residual     = residual_len > 0
seqlen_k_pack = seqlen_k - residual_len

print(f"residual_len: {residual_len}, residual: {residual}, seqlen_k_pack: {seqlen_k_pack}")

cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k_pack, seqlen_k_pack, 
                           dtype=torch.int32, device=device)

# Initialize quantization tensors
k_pack   = torch.zeros((batch_size, int(seqlen_k_pack // pack_nums), nheads_k, d),  dtype=torch.uint16, device=device)
k_params = torch.zeros((batch_size, int(seqlen_k_pack // group_size), nheads_k, d), dtype=torch.float32, device=device)

v_pack   = torch.zeros((batch_size, seqlen_k_pack, nheads_k, int(d // pack_nums)),  dtype=torch.uint16, device=device)
v_params = torch.zeros((batch_size, int(d // group_size), nheads_k, seqlen_k_pack), dtype=torch.float32, device=device)

# KV Cache Dynamic Cache
past_key_value = DynamicCache()

if residual:
    k_state_residual = k_state[:, -residual_len:, :, :]
    v_state_residual = v_state[:, -residual_len:, :, :]
    k_state_past = k_state[:, :-residual_len, :, :]
    v_state_past = v_state[:, :-residual_len, :, :]
    past_key_value.update_residual(k_state_residual, v_state_residual, layer_idx)
else:
    k_state_past = k_state
    v_state_past = v_state

kvcache_pack_int(
    k_state_past, k_pack, k_params,
    v_state_past, v_pack, v_params,
    None, # opt_block_table
    cu_seqlens_k,              
    seqlen_k_pack,
    quant_mode,
    group_size,
    num_bits
)
past_key_value.update_pack(k_pack, k_params, v_pack, v_params, layer_idx)

# self
k_pack_new = torch.empty((batch_size, int(residual_block_size // pack_nums), nheads_k, k_pack.size(-1)),  dtype=torch.uint16, device=device)
k_params_new = torch.empty((batch_size, int(residual_block_size // group_size), nheads_k, k_params.size(-1)), dtype=torch.float32, device=device)
v_pack_new = torch.empty((batch_size, residual_block_size, nheads_k, v_pack.size(-1)), dtype=torch.uint16, device=device)
v_params_new = torch.empty((batch_size, v_params.size(1), nheads_k, residual_block_size), dtype=torch.float32, device=device)

####### Round 2-3 : Decode #######
for round_idx in range(32):
    k_new = torch.randn(batch_size, 1, nheads_k, d, device=device, dtype=dtype)
    v_new = torch.randn(batch_size, 1, nheads_k, d, device=device, dtype=dtype)

    # Get kv cache_pack
    k_pack, k_params, v_pack, v_params = past_key_value.update_pack(None, None, None, None, layer_idx)

    seqlen_pack = v_pack.shape[1]
    seqlens_k = torch.full((batch_size,), seqlen_pack, dtype=torch.int32, device=device)

    # Get kv cache_residual and append new kv
    k_residual = torch.zeros((batch_size, residual_block_size, nheads_k, d), device=device, dtype=dtype)
    v_residual = torch.zeros((batch_size, residual_block_size, nheads_k, d), device=device, dtype=dtype)
    k_residual_cache, v_residual_cache = past_key_value.update_residual(k_new, v_new, layer_idx)

    cur_residual_len = k_residual_cache.shape[1]
    print(f"cur_residual_len: {cur_residual_len}")

    k_residual[:, :cur_residual_len, :, :] = k_residual_cache
    v_residual[:, :cur_residual_len, :, :] = v_residual_cache

    out_bitdecode, k_pack_new, k_params_new, v_pack_new, v_params_new = fwd_kvcache_int(
        q,
        k_pack, k_params, 
        v_pack, v_params,
        k_residual, v_residual, seqlens_k, #seqlens_k
        k_pack_new, k_params_new, v_pack_new, v_params_new,
        None, # opt_block_table
        sm_scale,
        quant_mode, 
        group_size,
        residual_block_size,
        cur_residual_len, # new_lens
        num_bits
    )

    if cur_residual_len == residual_block_size:
        past_key_value.update_pack(k_pack_new, k_params_new, v_pack_new, v_params_new, layer_idx)
        past_key_value.clear_residual(layer_idx)

    k_state = torch.cat([k_state, k_new], dim=1)
    v_state = torch.cat([v_state, v_new], dim=1)

    out_ref = attention_ref(q, k_state, v_state)[0]
    print(f"Round {round_idx+2}: bitdecode vs pytorch: {(out_bitdecode - out_ref).abs().mean().item()}")