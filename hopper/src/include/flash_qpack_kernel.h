/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "qpack.h"
#include "dequantize.h"
// #include "include/softmax.h"
// #include "include/mask.h"
// #include "include/dropout.h"
// #include "include/rotary.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_qpack_1rowblock(const Params &params, const int bidb, const int bidh, const int blockN_idx) {

    using Element       = typename Kernel_traits::Element;
    using ElementKVPack = typename Kernel_traits::ElementKVPack;
    using SharedStorage = typename Kernel_traits::SharedStorage;
    using index_t       = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockN           = Kernel_traits::kBlockN;
    constexpr int kBlockP           = Kernel_traits::kBlockP;
    constexpr int kBlockK_params    = Kernel_traits::kBlockK_params;
    constexpr int kHeadDim          = Kernel_traits::kHeadDim;
    constexpr int kHeadDim_pack     = Kernel_traits::kHeadDim_pack;
    constexpr int kHeadDim_k        = Kernel_traits::kHeadDim_k;
    constexpr int kHeadDim_k_params = Kernel_traits::kHeadDim_k_params;
    constexpr int kHeadDim_v_params = Kernel_traits::kHeadDim_v_params;
    constexpr int kNWarps           = Kernel_traits::kNWarps;
    constexpr int tile_paramsk_j    = Kernel_traits::tile_paramsk_j;
    constexpr int tile_paramsk_k    = Kernel_traits::tile_paramsk_k;
    constexpr int tile_paramsk_g    = Kernel_traits::tile_paramsk_g;
    constexpr int tile_paramsv_k    = Kernel_traits::tile_paramsv_k;
    constexpr int num_bits          = Kernel_traits::num_bits;
    constexpr int group_size        = Kernel_traits::group_size;
    constexpr int num_params        = Kernel_traits::num_params;

    const BlockInfo binfo(params, bidb);

    const int bidb_cache              = bidb;
    const int *block_table            = nullptr;
    const int block_table_idx         = 0;
    const int block_table_offset      = 0;
    const int block_table_offset_pack = 0;

    const index_t row_offset_k      = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + blockN_idx * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_k_pack = binfo.k_offset(params.K_pack_batch_stride, params.K_pack_row_stride, bidb_cache)
          + blockN_idx * kBlockP * params.K_pack_row_stride + (bidh / params.h_h_k_ratio) * params.K_pack_head_stride;
    const index_t row_offset_k_params = binfo.k_offset(params.k_params_batch_stride, params.k_params_row_stride, bidb)
          + blockN_idx * kBlockK_params * params.k_params_row_stride + (bidh / params.h_h_k_ratio) * params.k_params_head_stride;

    const index_t row_offset_v        = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + blockN_idx * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_v_pack   = binfo.k_offset(params.v_pack_batch_stride, params.v_pack_row_stride, bidb_cache)
          + blockN_idx * kBlockN * params.v_pack_row_stride + (bidh / params.h_h_k_ratio) * params.v_pack_head_stride;
    const index_t row_offset_v_params = binfo.k_offset(params.v_params_batch_stride, params.v_params_row_stride, bidb)
          + blockN_idx * kBlockN * params.v_params_row_stride + (bidh / params.h_h_k_ratio) * params.v_params_head_stride;

    // Tensor, global memory
    Tensor gK           = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_stride(params.k_row_stride, _1{}));
    Tensor gK_pack      = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKVPack*>(params.K_pack_ptr) + row_offset_k_pack),
                           Shape<Int<kBlockP>, Int<kHeadDim_k>>{},
                           make_stride(params.K_pack_row_stride, _1{}));
    Tensor gK_params    = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.k_params_ptr) + row_offset_k_params),
                           Shape<Int<kBlockK_params>, Int<kHeadDim_k_params>>{},
                           make_stride(params.k_params_row_stride, params.k_params_dim_stride));

    Tensor gV           = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_stride(params.v_row_stride, _1{}));
    Tensor gV_pack      = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKVPack*>(params.v_pack_ptr) + row_offset_v_pack),
                           Shape<Int<kBlockN>, Int<kHeadDim_pack>>{},
                           make_stride(params.v_pack_row_stride, _1{}));
    Tensor gV_params    = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.v_params_ptr) + row_offset_v_params),
                           Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{},
                           make_stride(params.v_params_row_stride, params.v_params_dim_stride));

    Tensor sK           = make_tensor(make_smem_ptr(shared_storage.smem_K.data()), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV           = make_tensor(make_smem_ptr(shared_storage.smem_V.data()), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    
    Tensor sK_pack            = make_tensor(make_smem_ptr(shared_storage.smem_Kpack.data()), typename Kernel_traits::SmemLayoutKPack{});
    Tensor sK_pack_transposed = make_tensor(sK_pack.data(), typename Kernel_traits::SmemLayoutKPacktransposed{});
    Tensor sV_pack            = make_tensor(make_smem_ptr(shared_storage.smem_Vpack.data()), typename Kernel_traits::SmemLayoutVPack{});
    Tensor sVt_pack           = make_tensor(sV_pack.data(), typename Kernel_traits::SmemLayoutVPacktransposed{});
    Tensor sVtNoSwizzle_pack  = make_tensor(sV_pack.data().get(), typename Kernel_traits::SmemLayoutVPacktransposedNoSwizzle{});

    Tensor sReduce_tmp        = make_tensor(make_smem_ptr(shared_storage.smem_reduce_tmp.data()), typename Kernel_traits::SmemLayoutReduce_tmp{});

    //
    // copy: global - shared
    //
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tKgK            = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK            = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV            = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV            = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::GmemTileCopyK_Pack gmem_tiled_copy_k_pack;
    auto gmem_thr_copy_k_pack = gmem_tiled_copy_k_pack.get_thread_slice(tidx);
    Tensor tKsK_pack_s2g      = gmem_thr_copy_k_pack.partition_S(sK_pack);
    Tensor tKgK_pack_s2g      = gmem_thr_copy_k_pack.partition_D(gK_pack);
    Tensor tKgK_pack_g2s      = gmem_thr_copy_k_pack.partition_S(gK_pack);
    Tensor tKsK_pack_g2s      = gmem_thr_copy_k_pack.partition_D(sK_pack);

    typename Kernel_traits::GmemTileCopyV_Pack gmem_tiled_copy_v_pack;
    auto gmem_thr_copy_v_pack = gmem_tiled_copy_v_pack.get_thread_slice(tidx);
    Tensor tVsV_pack_s2g      = gmem_thr_copy_v_pack.partition_S(sV_pack);
    Tensor tVgV_pack_s2g      = gmem_thr_copy_v_pack.partition_D(gV_pack);
    Tensor tVgV_pack_g2s      = gmem_thr_copy_v_pack.partition_S(gV_pack);
    Tensor tVsV_pack_g2s      = gmem_thr_copy_v_pack.partition_D(sV_pack);

    //
    // Tensor: Register per thread
    //
    
    typename Kernel_traits::TiledMma tiled_mma;
    typename Kernel_traits::TiledMmaK_i4 tiled_mma_i4;
    auto thr_mma          = tiled_mma.get_thread_slice(tidx);
    auto thr_mma_i4       = tiled_mma_i4.get_thread_slice(tidx);
    Tensor tSrK           = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tSrK_dequant   = thr_mma.partition_fragment_B(sK);
    Tensor tSrK_pack_tmp  = thr_mma_i4.partition_fragment_B(sK_pack_transposed);                      // (MMA,MMA_N,MMA_K)
    Tensor tSrK_pack      = make_fragment_like<ElementKVPack>(tSrK_pack_tmp);

    Tensor tSrV           = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tSrV_dequant   = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tSrV_pack_tmp  = thr_mma_i4.partition_fragment_B(sVtNoSwizzle_pack);
    Tensor tSrV_pack      = make_fragment_like<ElementKVPack>(tSrV_pack_tmp);

    //
    // copy: shared - register
    //

    auto smem_tiled_copy_K       = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K         = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK                  = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK_view             = smem_thr_copy_K.retile_D(tSrK);
    
    auto smem_tiled_copy_kv_pack = make_tiled_copy_B(typename Kernel_traits::R2SCopyAtomPack{}, tiled_mma_i4);
    auto smem_thr_copy_kv_pack   = smem_tiled_copy_kv_pack.get_thread_slice(tidx);
    Tensor tSrK_pack_r2s_view    = smem_thr_copy_kv_pack.retile_S(tSrK_pack);
    Tensor tSsK_pack_r2s         = smem_thr_copy_kv_pack.partition_D(sK_pack);
    Tensor tSrV_pack_r2s_view    = smem_thr_copy_kv_pack.retile_S(tSrV_pack);
    Tensor tSsV_pack_r2s         = smem_thr_copy_kv_pack.partition_D(sVt_pack);
    Tensor tSsK_pack_s2r         = smem_thr_copy_kv_pack.partition_S(sK_pack);
    Tensor tSrK_pack_s2r_view    = smem_thr_copy_kv_pack.retile_D(tSrK_pack);

    auto smem_tiled_copy_V       = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_tiled_copy_V_pack  = make_tiled_copy_B(typename Kernel_traits::R2SCopyAtomPack{}, tiled_mma_i4);
    auto smem_thr_copy_V         = smem_tiled_copy_V.get_thread_slice(tidx);
    auto smem_thr_copy_V_pack    = smem_tiled_copy_V_pack.get_thread_slice(tidx);
    Tensor tSsV                  = smem_thr_copy_V.partition_S(sVt);
    Tensor tSrV_view             = smem_thr_copy_V.retile_D(tSrV);
    Tensor tSsV_pack_s2r         = smem_thr_copy_V_pack.partition_S(sVt_pack);
    Tensor tSrV_pack_s2r_view    = smem_thr_copy_V_pack.retile_D(tSrV_pack);

    // Advance gK
    cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    
    cute::cp_async_fence();
    flash::cp_async_wait<0>();                               
    __syncthreads();

    // Advance gV
    cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    cute::cp_async_fence();

    cute::copy(smem_tiled_copy_K, tSsK, tSrK_view);

    // quantize kv
    using TensorParamsKC = decltype(make_tensor<half_t>(make_shape(Int<4 * num_params>{}, Int<tile_paramsk_k>{})));
    using TensorParamsVG = decltype(make_tensor<half_t>(make_shape(Int<num_bits * num_params>{}, Int<tile_paramsv_k>{}))); // TODO: need to change, hardcode num_bits
    using TensorParamsG  = decltype(make_tensor<half_t>(make_shape(Int<tile_paramsk_g>{})));
    
    TensorParamsKC tScales_k_c, tZeros_k_c;
    TensorParamsVG tScales_v_c, tZeros_v_c;
    TensorParamsG tScales_k_g, tZeros_k_g;

    if (Kernel_traits::quant_mode == 1) {
        quant::qpack_Kchannel_Vtensor<num_bits>(tSrK, tSrK_pack, tScales_k_c, tZeros_k_c, sReduce_tmp, num_params);
    } else {
        quant::quant_Ktensor(tSrK, tSrK_pack, tScales_k_g, tZeros_k_g, num_params);
    }
    
    auto tScales_k_h2_c = cute::recast<__half2>(tScales_k_c);
    auto tZeros_k_h2_c  = cute::recast<__half2>(tZeros_k_c);
    auto tScales_k_h2_g = cute::recast<__half2>(tScales_k_g);
    auto tZeros_k_h2_g  = cute::recast<__half2>(tZeros_k_g);

    auto tScales_v_h2   = cute::recast<__half2>(tScales_v_c);
    auto tZeros_v_h2    = cute::recast<__half2>(tZeros_v_c);

    flash::cp_async_wait<0>();                               
    __syncthreads();
    cute::copy(smem_tiled_copy_V, tSsV, tSrV_view);

    quant::qpack_Kchannel_Vtensor<num_bits>(tSrV, tSrV_pack, tScales_v_c, tZeros_v_c, sReduce_tmp, num_params);

    const int num_params_2 = num_bits == 2 ? num_params / 2 : num_params;
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tScales_v_h2); ++i) {
        CUTE_UNROLL
        for (int j = 0; j < size<0>(tScales_v_h2); ++j) {
            gV_params(128 * (i / 8) + 0  + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2) = tScales_v_h2(j, i);
            gV_params(128 * (i / 8) + 64 + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2) = tZeros_v_h2(j, i);
        }
    }

    if (Kernel_traits::quant_mode == 1) {
        CUTE_UNROLL
        for (int i = 0; i < size<1>(tScales_k_h2_c); ++i) {
            CUTE_UNROLL
            for (int j = 0; j < size<0>(tScales_k_h2_c); ++j) {
                gK_params(j % num_params, 0  + 8 * i + 4 * (j / num_params) + tidx % 4) = tScales_k_h2_c(j, i);
                gK_params(j % num_params, 64 + 8 * i + 4 * (j / num_params) + tidx % 4) = tZeros_k_h2_c(j, i);
            }
        }
    } else {
        CUTE_UNROLL
        for (int j = 0; j < size<0>(tScales_k_h2_g); ++j) {
            gK_params(0  + 32 * (j / num_params) + tidx / 4, j % num_params) = tScales_k_h2_g(j);
            gK_params(64 + 32 * (j / num_params) + tidx / 4, j % num_params) = tZeros_k_h2_g(j);
        }
    }
    
    // copy from register to shared memory
    cute::copy(smem_tiled_copy_kv_pack, tSrK_pack_r2s_view, tSsK_pack_r2s);
    __syncthreads();
    if (kHeadDim == 128 && num_bits == 2) {
        if (tidx < 64) {
            cute::copy(smem_tiled_copy_kv_pack, tSrV_pack_r2s_view, tSsV_pack_r2s);
        }
    } else {
        cute::copy(smem_tiled_copy_kv_pack, tSrV_pack_r2s_view, tSsV_pack_r2s);
    }

    // copy from shared to global
    __syncthreads();
    cute::copy(gmem_tiled_copy_k_pack, tKsK_pack_s2g, tKgK_pack_s2g);
    __syncthreads();
    cute::copy(gmem_tiled_copy_v_pack, tVsV_pack_s2g, tVgV_pack_s2g);
    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////
    // verify the quantize
    // clear(tSrK_pack);
    // clear(tSsK_pack_r2s);
    // clear(tSrV_pack);
    // clear(tSsV_pack_r2s);

    // __syncthreads();
    // cute::copy(gmem_tiled_copy_k_pack, tKgK_pack_g2s, tKsK_pack_g2s);
    // cute::copy(gmem_tiled_copy_v_pack, tVgV_pack_g2s, tVsV_pack_g2s);

    // __syncthreads();
    // cute::copy(smem_tiled_copy_kv_pack, tSsK_pack_s2r, tSrK_pack_s2r_view);
    // cute::copy(smem_tiled_copy_V_pack, tSsV_pack_s2r, tSrV_pack_s2r_view); 

    // __syncthreads();
    // clear(tScales_k_h2_c);
    // clear(tZeros_k_h2_c);
    // clear(tScales_k_h2_g);
    // clear(tZeros_k_h2_g);
    
    // clear(tScales_v_h2);
    // clear(tZeros_v_h2);

    // CUTE_UNROLL
    // for (int i = 0; i < size<1>(tScales_v_h2); ++i) {
    //     CUTE_UNROLL
    //     for (int j = 0; j < size<0>(tScales_v_h2); ++j) {
    //         tScales_v_h2(j, i) = gV_params(128 * (i / 8) + 0  + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2);
    //         tZeros_v_h2(j, i)  = gV_params(128 * (i / 8) + 64 + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2);
    //     }
    // }

    CUTE_UNROLL
    for (int i = 0; i < size<2>(tSrV_pack); ++i) {  
       quant::dequant_Kchannel_Vtensor<num_bits>(tSrV_pack(_,_,i), tSrV_dequant(_,_,i), tScales_v_c(_,i), tZeros_v_c(_,i), num_params);
    }

    if (Kernel_traits::quant_mode == 1) {
        // CUTE_UNROLL
        // for (int i = 0; i < size<1>(tScales_k_h2_c); ++i) {
        //     CUTE_UNROLL
        //     for (int j = 0; j < size<0>(tScales_k_h2_c); ++j) {
        //         tScales_k_h2_c(j, i) = gK_params(j % num_params, 0  + 8 * i + 4 * (j / num_params) + tidx % 4);
        //         tZeros_k_h2_c(j, i)  = gK_params(j % num_params, 64 + 8 * i + 4 * (j / num_params) + tidx % 4);
        //     }
        // }

        CUTE_UNROLL
        for (int i = 0; i < size<2>(tSrK_pack); ++i) {
            quant::dequant_Kchannel_Vtensor<num_bits>(tSrK_pack(_,_,i), tSrK_dequant(_,_,i), tScales_k_c(_,i), tZeros_k_c(_,i), num_params);
        }
    } else {
        // CUTE_UNROLL
        // for (int j = 0; j < size<0>(tScales_k_h2_g); ++j) {
        //     tScales_k_h2_g(j) = gK_params(0  + 32*j + tidx/4, 0);
        //     tZeros_k_h2_g(j)  = gK_params(64 + 32*j + tidx/4, 0);
        // }

        // auto tScales_k_h1_g = cute::recast<__half>(tScales_k_h2_g);
        // auto tZeros_k_h1_g = cute::recast<__half>(tZeros_k_h2_g);

        // CUTE_UNROLL
        // for (int i = 0; i < size<2>(tSrK_pack); ++i) {
        //     quant::dequantize_Ktensor(tSrK_pack, tSrK_dequant, tScales_k_h2_g, tZeros_k_h2_g, 4, group_size, i);
        // }
    }

    //////////////////////////////////////////////////////////////////////////////
    #if DEBUG2
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        PRINT("tSrK", tSrK.layout());                                   PRINTTENSOR("tSrK", tSrK);
        PRINT("tSrK_dequant", tSrK_dequant.layout());                   PRINTTENSOR("tSrK_dequant", tSrK_dequant);
        PRINT("gK_pack", gK_pack.layout());                             PRINTTENSOR("gK_pack", gK_pack);
        // auto gK_params_f = cute::recast<cutlass::half_t>(gK_params);
        // PRINT("gK_params", gK_params.layout());                         PRINTTENSOR("gK_params", gK_params_f);
        printf("#####################################################################################\n");
    }
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_qpack(const Params &params) {
    // The block index for the number of blocks.
    const int blockN_idx = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    flash::compute_qpack_1rowblock<Kernel_traits>(params, bidb, bidh, blockN_idx);
}


} // namespace flash