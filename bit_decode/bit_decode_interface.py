# Copyright (c) 2025, Dayou Du.

from typing import Optional, Union

import torch
import torch.nn as nn

import bit_decode_cuda as bit_decode_cuda

def kvcache_pack_int(k_cache: torch.Tensor, k_pack: torch.Tensor, k_params: torch.Tensor,
                     v_cache: torch.Tensor, v_pack: torch.Tensor, v_params: torch.Tensor,
                     opt_block_table: Optional[torch.Tensor] = None,
                     cu_seqlens_k: torch.Tensor = None,
                     seqlen_k: int = 0,
                     quant_mode: str = "k-channel",
                     group_size: int = 128,
                     num_bits: int = 4):
    
    batch_size, seqlen_k, nheads_k, d = k_cache.shape

    K_unpad = k_cache.reshape(batch_size * seqlen_k, nheads_k, d)
    V_unpad = v_cache.reshape(batch_size * seqlen_k, nheads_k, d)

    if num_bits == 4:
        bit_decode_cuda.kvcache_pack_i4(K_unpad, k_pack, k_params,
                                        V_unpad, v_pack, v_params,
                                        opt_block_table,
                                        cu_seqlens_k,
                                        seqlen_k,
                                        quant_mode,
                                        group_size
                                        )
    else:
        bit_decode_cuda.kvcache_pack_i2(K_unpad, k_pack, k_params,
                                        V_unpad, v_pack, v_params,
                                        opt_block_table,
                                        cu_seqlens_k,
                                        seqlen_k,
                                        quant_mode,
                                        group_size
                                        )

def fwd_kvcache_int(q: torch.Tensor, 
                    k_pack: torch.Tensor, k_params: torch.Tensor, 
                    v_pack: torch.Tensor, v_params: torch.Tensor,
                    opt_block_table: Optional[torch.Tensor] = None,
                    softmax_scale: float = 1.0,
                    quant_mode: str = "k-channel",
                    group_size: int = 128,
                    num_bits: int = 4):
    
    if num_bits == 4:
        out_bit = bit_decode_cuda.fwd_kvcache_i4(
            q,
            k_pack, k_params, 
            v_pack, v_params,
            opt_block_table,
            softmax_scale,
            quant_mode, 
            group_size,
            False,          # is_causal
            -1,             # window_size_left
            -1,             # window_size_right
            0.0,            # softcap
            True,           # is_rotary_interleaved
            0               # num_splits
        )
    else:
        out_bit = bit_decode_cuda.fwd_kvcache_i2(
            q,
            k_pack, k_params, 
            v_pack, v_params,
            opt_block_table,
            softmax_scale,
            quant_mode, 
            group_size,
            False,          # Added
            -1,             # Added
            -1,             # Added
            0.0,            # Added
            True,           # Added
            0               # Added
        )


    return out_bit
