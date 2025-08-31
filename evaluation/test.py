import torch
import torch.nn as nn
import math
import triton
from einops import rearrange, repeat
import numpy as np

from flash_attn import flash_attn_with_kvcache
from bit_decode import kvcache_pack_int, fwd_kvcache_int


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


# Define constants
batch_size = 1
nheads = 32
nheads_k = 32
d = 128

# Sequence length
seqlen_q = 1
seqlen_kv = 4096

# Quantization parameters
quant_mode = "k-channel"
num_bits = 4
pack_nums = 16 / num_bits
group_size = 128


# Set seed and parameters
device = "cuda"
dtype = torch.float16
torch.random.manual_seed(0)

# Initialize tensors
q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
k_cache = torch.randn(batch_size, seqlen_kv, nheads_k, d, device=device, dtype=dtype)
v_cache = torch.randn(batch_size, seqlen_kv, nheads_k, d, device=device, dtype=dtype)

k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=nheads // nheads_k)
v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=nheads // nheads_k)

# Reference attention computation
out_ref, _ = attention_ref(q, k_cache_rep, v_cache_rep)

##################### BitDecoding Packing Kernel ##################### 

# Initialize quantization tensors
if quant_mode == "k-channel":
    k_pack   = torch.zeros((batch_size, int(seqlen_kv // pack_nums), nheads_k, d),  dtype=torch.uint16, device=device)
    k_params = torch.zeros((batch_size, int(seqlen_kv // group_size), nheads_k, d), dtype=torch.float32, device=device)
else:
    k_pack   = torch.zeros((batch_size, seqlen_kv, nheads_k, int(d // pack_nums)),  dtype=torch.uint16, device=device)
    k_params = torch.zeros((batch_size, int(d // group_size), nheads_k, seqlen_kv), dtype=torch.float32, device=device)

v_pack   = torch.zeros((batch_size, seqlen_kv, nheads_k, int(d // pack_nums)),  dtype=torch.uint16, device=device)
v_params = torch.zeros((batch_size, int(d // group_size), nheads_k, seqlen_kv), dtype=torch.float32, device=device)

cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_kv, seqlen_kv, dtype=torch.int32, device=device)

kvcache_pack_int(
    k_cache, k_pack, k_params,
    v_cache, v_pack, v_params,
    None, # opt_block_table
    cu_seqlens_k,              
    seqlen_kv,
    quant_mode,
    group_size,
    num_bits
)

sm_scale = 1.0 / math.sqrt(d)
out_bitdecode = fwd_kvcache_int(
                    q,
                    k_pack, k_params, 
                    v_pack, v_params,
                    None, # opt_block_table
                    sm_scale,
                    quant_mode, 
                    group_size,
                    num_bits
                )

print(f"seqlen_kv:{seqlen_kv} BitDecode vs Pytorch: {(out_bitdecode - out_ref).abs().mean().item()}")

print(f"out_ref: \n{out_ref[0,0,0,:8]}")
print(f"out_bitdecode: \n{out_bitdecode[0,0,0,:8]}")