# Copyright (c) 2025, Dayou Du.

from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import bit_decode_cuda as bit_decode_cuda

def kvcache_pack_int(k_cache: torch.Tensor, k_pack: torch.Tensor, k_params: torch.Tensor,
                     v_cache: torch.Tensor, v_pack: torch.Tensor, v_params: torch.Tensor,
                     opt_block_table: Optional[torch.Tensor] = None,
                     cu_seqlens_k: torch.Tensor = None,
                     seqlen_k: int = 0,
                     quant_mode: str = "k-tensor",
                     group_size: int = 128,
                     num_bits: int = 4):
    
    batch_size, seqlen_k, nheads_k, d = k_cache.shape

    K_unpad = k_cache.reshape(batch_size * seqlen_k, nheads_k, d)
    V_unpad = v_cache.reshape(batch_size * seqlen_k, nheads_k, d)

    if num_bits == 4:
        bit_decode_cuda.kvcache_pack_int4(K_unpad, k_pack, k_params,
                                          V_unpad, v_pack, v_params,
                                          opt_block_table,
                                          cu_seqlens_k,
                                          seqlen_k,
                                          quant_mode,
                                          group_size
                                         )
    # else:
    #     bit_decode_cuda.kvcache_pack_int2(K_unpad, k_pack, k_params,
    #                                        V_unpad, v_pack, v_params,
    #                                        opt_block_table,
    #                                        cu_seqlens_k,
    #                                        seqlen_k,
    #                                        quant_mode,
    #                                        group_size
    #                                        )

def fwd_kvcache_int(q: torch.Tensor, 
                    k_pack: torch.Tensor, k_params: torch.Tensor, 
                    v_pack: torch.Tensor, v_params: torch.Tensor,
                    opt_k_new: Optional[torch.Tensor] = None,
                    opt_v_new: Optional[torch.Tensor] = None,
                    opt_seqlens_k: Optional[torch.Tensor] = None,
                    k_pack_new: torch.Tensor = None, k_params_new: torch.Tensor = None,
                    v_pack_new: torch.Tensor = None, v_params_new: torch.Tensor = None,
                    opt_block_table: Optional[torch.Tensor] = None,
                    softmax_scale: float = 1.0,
                    quant_mode: str = "k-tensor",
                    group_size: int = 128,
                    residual_block_size: int = 128,
                    new_lens: int = 0,
                    num_bits: int = 4):
    
    if num_bits == 4:
        out_bit, k_pack_new, k_params_new, v_pack_new, v_params_new = bit_decode_cuda.fwd_kvcache_int4(
            q,
            k_pack, k_params, 
            v_pack, v_params,
            opt_k_new, opt_v_new, opt_seqlens_k,
            k_pack_new, k_params_new, v_pack_new, v_params_new,
            opt_block_table,
            softmax_scale,
            quant_mode, 
            group_size,
            residual_block_size,
            new_lens,
            False,          # Added
            -1,             # Added
            -1,             # Added
            0.0,            # Added
            True,           # Added
            0               # Added
        )
    # else:
    #     out_bit, k_pack_new, k_params_new, v_pack_new, v_params_new = bit_decode_cuda.fwd_kvcache_int2(
    #         q,
    #         k_pack, k_params, 
    #         v_pack, v_params,
    #         opt_k_new, opt_v_new, opt_seqlens_k,
    #         k_pack_new, k_params_new, v_pack_new, v_params_new,
    #         opt_block_table,
    #         softmax_scale,
    #         quant_mode, 
    #         group_size,
    #         residual_block_size,
    #         new_lens,
    #         False,          # Added
    #         -1,             # Added
    #         -1,             # Added
    #         0.0,            # Added
    #         True,           # Added
    #         0               # Added
    #     )


    return out_bit, k_pack_new, k_params_new, v_pack_new, v_params_new
