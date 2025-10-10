/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "include/block_info.h"
#include "include/kernel_traits.h"
#include "include/utils.h"
#include "include/softmax.h"
#include "include/mask.h"
#include "include/dropout.h"
#include "include/rotary.h"
#include "include/dequantize.h"
#include "include/qpack.h"

namespace flash {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {
    // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Paged_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element       = typename Kernel_traits::Element;
    using ElementSF     = typename Kernel_traits::ElementSF;
    using ElementOut    = typename Kernel_traits::ElementOut;
    using ElementKVPack = typename Kernel_traits::ElementKVPack;
    using ElementAccum  = typename Kernel_traits::ElementAccum;
    using index_t       = typename Kernel_traits::index_t;
    using SharedStorage = typename Kernel_traits::SharedStorage;

    using BlkScaledConfig = typename Kernel_traits::BlkScaledConfig;
    using LayoutSF = typename Kernel_traits::LayoutSF;

    // Shared memory.
    extern __shared__ char smem_[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM               = Kernel_traits::kBlockM;
    constexpr int kBlockN               = Kernel_traits::kBlockN;

    constexpr int kHeadDim              = Kernel_traits::kHeadDim;
    constexpr int kNWarps               = Kernel_traits::kNWarps;

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, ElementOut, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((binfo.actual_seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = n_split_idx * n_blocks_per_split;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);

    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache              = bidb;
    const int *block_table            = !Paged_KV ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx         = !Paged_KV ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset      = !Paged_KV ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const int block_table_offset_pack = !Paged_KV ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size_pack;

    const index_t row_offset_k_pack   = !Paged_KV
        ? binfo.k_offset(params.k_pack_batch_stride, params.k_pack_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_pack_row_stride + (bidh / params.h_h_k_ratio) * params.k_pack_head_stride
        : block_table[block_table_idx] * params.k_pack_batch_stride + block_table_offset_pack * params.k_pack_row_stride + (bidh / params.h_h_k_ratio) * params.k_pack_head_stride;
    
    const index_t row_offset_v        = !Paged_KV
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride      + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : block_table[block_table_idx] * params.v_batch_stride      + block_table_offset * params.v_row_stride      + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_v_pack   = !Paged_KV
        ? binfo.k_offset(params.v_pack_batch_stride, params.v_pack_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_pack_row_stride + (bidh / params.h_h_k_ratio) * params.v_pack_head_stride
        : block_table[block_table_idx] * params.v_pack_batch_stride + block_table_offset * params.v_pack_row_stride + (bidh / params.h_h_k_ratio) * params.v_pack_head_stride;
    const index_t row_offset_v_params = binfo.k_offset(params.v_params_batch_stride, params.v_params_row_stride, bidb)
          + (n_block_max - 1) * kBlockN * params.v_params_row_stride + (bidh / params.h_h_k_ratio) * params.v_params_head_stride;    

    //
    // Tensor, global memory
    //

    // Q
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mSFQ = make_tensor(make_gmem_ptr(reinterpret_cast<ElementSF*>(params.sfq_ptr)), 
                              make_shape(binfo.actual_seqlen_q, params.h, params.d),
                              make_stride(params.sfq_row_stride, params.sfq_head_stride, _1{}));
    Tensor gSFQ = local_tile(mSFQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    // K
    Tensor gK   = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_pack_ptr)),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_stride(params.k_pack_row_stride, _1{}));
    Tensor gSFK = make_tensor(make_gmem_ptr(reinterpret_cast<ElementSF*>(params.sfk_ptr)),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_stride(params.sfk_row_stride, _1{}));

    // // V
    // Tensor gV_pack   = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKVPack*>(params.v_pack_ptr) + row_offset_v_pack),
    //                        Shape<Int<kBlockN>, Int<kHeadDim_pack>>{},
    //                        make_stride(params.v_pack_row_stride, _1{}));
    // Tensor gV_params = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.v_params_ptr) + row_offset_v_params),
    //                        Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{},
    //                        make_stride(_1{}, params.v_params_dim_stride));

    // #if DEBUG
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("num_n_splits = %d, n_blocks_per_split = %d\n", num_n_splits, n_blocks_per_split);
    //     printf("params.new_lens = %d, binfo.actual_seqlen_k = %d\n", params.new_lens, binfo.actual_seqlen_k);
    //     printf("params.seqlen_k = %d, binfo.seqlen_k_cache = %d\n", params.seqlen_k, binfo.seqlen_k_cache);
    //     printf("n_block_min = %d, n_block_max = %d\n", n_block_min, n_block_max);
    // }
    // #endif

    //
    // Tensor, shared memory
    //

    // Q
    Tensor sQ                   = make_tensor(make_smem_ptr(shared_storage.smem_Q         .data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sSFQ                 = make_tensor(make_smem_ptr(shared_storage.smem_SFQ        .data()), typename Kernel_traits::SmemLayoutSFQ{});

    // K
    Tensor sK                     = make_tensor(make_smem_ptr(shared_storage.smem_K        .data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sSFK                   = make_tensor(make_smem_ptr(shared_storage.smem_SFK      .data()), typename Kernel_traits::SmemLayoutSFK{});

    // // Acc
    // Tensor sAcc                 = make_tensor(make_smem_ptr(shared_storage.smem_acc     .data()), typename Kernel_traits::SmemLayoutAcc{});
    // Tensor sReduce_tmp          = make_tensor(make_smem_ptr(shared_storage.smem_acc     .data()), typename Kernel_traits::SmemLayoutReduce_tmp{});

    // // V
    // Tensor sVtNoSwizzle_dequant  = make_tensor(make_smem_ptr(shared_storage.smem_Q                                .data()), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    // Tensor sV_pack               = make_tensor(make_smem_ptr(shared_storage.smem_Vpack                            .data()), typename Kernel_traits::SmemLayoutVPack{});
    // Tensor sVt_pack              = make_tensor(sV_pack                                                             .data(), typename Kernel_traits::SmemLayoutVPacktransposed{});
    // Tensor sVtNoSwizzle_pack     = make_tensor(sV_pack                                                       .data().get(), typename Kernel_traits::SmemLayoutVPacktransposedNoSwizzle{});
    // Tensor sV_params             = make_tensor(make_smem_ptr(shared_storage.smem_Vparams                          .data()), typename Kernel_traits::SmemLayoutVParams{});

    //
    // Copy, Global memory to shared memory
    //

    typename Kernel_traits::GmemTiledCopyQKV    gmem_tiled_copy_QKV;
    typename Kernel_traits::GmemTiledCopySF     gmem_tiled_copy_SF;
    // typename Kernel_traits::GmemTileCopyK_Pack  gmem_tiled_copy_k_pack;
    // typename Kernel_traits::GmemTileCopyV_Pack  gmem_tiled_copy_v_pack;
    // typename Kernel_traits::GmemTileCopyKParams gmem_tiled_copy_k_params;
    // typename Kernel_traits::GmemTileCopyVParams gmem_tiled_copy_v_params;

    auto gmem_thr_copy_QKV         = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    auto gmem_thr_copy_SF          = gmem_tiled_copy_SF.get_thread_slice(tidx);
    

    Tensor tQgQ            = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ            = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tQgSFQ          = gmem_thr_copy_SF.partition_S(gSFQ);
    Tensor tQsSFQ          = gmem_thr_copy_SF.partition_D(sSFQ);

    Tensor tKgK          = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK          = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tKgSFK        = gmem_thr_copy_SF.partition_S(gSFK);
    Tensor tKsSFK        = gmem_thr_copy_SF.partition_D(sSFK);

    #if DEBUG
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // PRINT("mQ", mQ.layout());
        // PRINT("gQ", gQ.layout());
        // PRINT("mSFQ", mSFQ.layout());
        // PRINT("gSFQ", gSFQ.layout());
        // PRINT("sQ", sQ.layout());
        // PRINT("sSFQ", sSFQ.layout());
        // PRINT("tQgSFQ", tQgSFQ.layout());
        // PRINT("tQsSFQ", tQsSFQ.layout());
        PRINT("gK", gK.layout());
        PRINT("sK", sK.layout());
        PRINT("tKgK", tKgK.layout()); 
        PRINT("tKsK", tKsK.layout()); 
        
    }
    #endif

    // Tensor tVgV_pack       = gmem_thr_copy_v_pack.partition_S(gV_pack);
    // Tensor tVsV_pack       = gmem_thr_copy_v_pack.partition_D(sV_pack);
    // Tensor tVgV_params     = gmem_thr_copy_v_params.partition_S(gV_params);
    // Tensor tVsV_params     = gmem_thr_copy_v_params.partition_D(sV_params);

    
    // MMA Atom partitioning
    

    typename Kernel_traits::TiledMmaQK          tiled_mma_qk;

    auto thr_mma_qk         = tiled_mma_qk.get_thread_slice(tidx);

    Tensor tSrQ             = thr_mma_qk.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrSFQ           = flash::partition_fragment_SFA(sSFQ, thr_mma_qk);

    Tensor tSrK             = thr_mma_qk.partition_fragment_B(sK);
    Tensor tSrSFK           = flash::partition_fragment_SFB(sSFK, thr_mma_qk);

    // Tensor tOrVt_dequant     = thr_mma.partition_fragment_B(sVtNoSwizzle_dequant);
    // Tensor tOrVt_pack_tmp    = thr_mma_KV_i4.partition_fragment_B(sVtNoSwizzle_pack);
    // Tensor tOrVt_pack        = make_fragment_like<ElementKVPack>(tOrVt_pack_tmp);

    Tensor acc_o             = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Tensor, ACC
    //

    // auto r2s_tiled_copy_c = make_tiled_copy_C(typename Kernel_traits::R2SCopyAtomAcc{}, tiled_mma);
    // auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(tidx);
    // auto tCsAcc_r2s       = r2s_thr_copy_c.partition_D(sAcc);

    // Tensor tSrAcc            = thr_mma.partition_fragment_A(sAcc);                           // (MMA,MMA_M,MMA_K)
    // auto smem_tiled_copy_Acc = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // auto smem_thr_copy_Acc   = smem_tiled_copy_Acc.get_thread_slice(tidx);
    // Tensor tSsAcc_view       = smem_thr_copy_Acc.partition_S(sAcc);
    // Tensor tSrAcc_view       = smem_thr_copy_Acc.retile_D(tSrAcc);

    //
    // Copy, Shared memory to register
    //
    auto tile_shape_mnk = tile_shape(tiled_mma_qk);

    // Q
    auto smem_tiled_copy_Q          = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma_qk);
    auto smem_thr_copy_Q            = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ                     = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSrQ_copy_view           = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_SFQ = make_tiled_copy_impl(typename Kernel_traits::SmemCopyAtomSF{}, 
                                                    flash::get_layoutSFA_TV(tiled_mma_qk),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                    );
    auto smem_thr_copy_SFQ   = smem_tiled_copy_SFQ.get_thread_slice(tidx);
    Tensor tSsSFQ            = smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ));
    Tensor tSrSFQ_copy_view  = smem_thr_copy_SFQ.retile_D(tSrSFQ);

    // K
    auto smem_tiled_copy_K   = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomK{}, tiled_mma_qk);
    auto smem_thr_copy_K     = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK              = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK_copy_view    = smem_thr_copy_K.retile_D(tSrK);

    auto smem_tiled_copy_SFK = make_tiled_copy_impl(typename Kernel_traits::SmemCopyAtomSF{}, 
                                flash::get_layoutSFB_TV(tiled_mma_qk),
                                make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                );
    auto smem_thr_copy_SFK   = smem_tiled_copy_SFK.get_thread_slice(tidx);
    Tensor tSsSFK            = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
    Tensor tSrSFK_copy_view  = smem_thr_copy_SFK.retile_D(tSrSFK);

    

    // V
    // auto smem_tiled_copy_V          = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    // auto smem_tiled_copy_V_pack     = make_tiled_copy_B(typename Kernel_traits::S2RCopyAtomV_i4{}, tiled_mma_KV_i4);
    // auto smem_thr_copy_V            = smem_tiled_copy_V.get_thread_slice(tidx);
    // auto smem_thr_copy_V_pack       = smem_tiled_copy_V_pack.get_thread_slice(tidx);

    // Tensor tOsVt_pack               = smem_thr_copy_V_pack.partition_S(sVt_pack);
    // Tensor tOrVt_pack_view          = smem_thr_copy_V_pack.retile_D(tOrVt_pack);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ  = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));        // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);                                // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    // 
    // Start of the main loop
    // 

    clear(acc_o);

    // Read Q from gmem to smem, optionally apply rotary embedding.
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
        cute::copy(gmem_tiled_copy_SF, tQgSFQ, tQsSFQ);
    } 
    cute::cp_async_fence();

    flash::Softmax<2 * size<1>(acc_o)> softmax;
    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask_main(binfo.seqlen_k_cache, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    //
    // Main loop
    //

    int n_block = n_block_max - 1;

    
    // Advance gK
    cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cute::copy(gmem_tiled_copy_SF, tKgSFK, tKsSFK);
    cute::cp_async_fence();

    const int n_masking_steps = 1;

    CUTE_UNROLL
    for (int masking_step = 0; masking_step < n_masking_steps && n_block >= n_block_min; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        // if ((masking_step > 0)) {
        //     if (block_table == nullptr) {
        //         tVgV_pack.data()   = tVgV_pack.data()   + (-int(kBlockN * params.v_pack_row_stride));
        //         tVgV_params.data() = tVgV_params.data() + (-int(kBlockN * params.v_params_row_stride));
        //     } else {
        //         const int block_table_idx_cur     = (n_block + 1) * kBlockN / params.page_block_size;
        //         const int block_table_offset_cur  = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
        //         const int block_table_idx_next    = n_block * kBlockN / params.page_block_size;
        //         const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
        //         // tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
        //         tVgV_pack.data() = tVgV_pack.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_pack_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_pack_row_stride;
        //         tVgV_params.data() = tVgV_params.data() + (-int(kBlockN * params.v_params_row_stride));
        //     }
        //     cute::copy(gmem_tiled_copy_v_pack, tVgV_pack, tVsV_pack);
        //     cute::copy(gmem_tiled_copy_v_params, tVgV_params, tVsV_params);
        // } else {
        //     cute::copy(gmem_tiled_copy_v_pack, tVgV_pack, tVsV_pack);
        //     cute::copy(gmem_tiled_copy_v_params, tVgV_params, tVsV_params);
        // }
        // cute::cp_async_fence();

        
        flash::gemm_scale(
            acc_s, 
            tSrQ, tSrSFQ,
            tSrK, tSrSFK, 
            tSsQ, tSrQ_copy_view,
            tSsSFQ, tSrSFQ_copy_view,
            tSsK, tSrK_copy_view,
            tSsSFK, tSrSFK_copy_view,
            tiled_mma_qk, 
            smem_tiled_copy_Q, 
            smem_tiled_copy_SFQ,
            smem_tiled_copy_K,
            smem_tiled_copy_SFK
        );

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        // Mask
        mask_main.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        // flash::cp_async_wait<0>();
        // __syncthreads();

        // if (n_block > n_block_min) {
        //     // Advance gK
        //     if (block_table == nullptr) {
        //         tKgK_pack.data()     = tKgK_pack.data()     + (-int(kBlockP * params.K_pack_row_stride));
        //         tKgK_params.data()   = tKgK_params.data()   + (-int(kBlockK_params * params.k_params_row_stride));
        //     } else {
        //         const int block_table_idx_cur          = n_block * kBlockN / params.page_block_size;
        //         const int block_table_offset_cur       = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
        //         const int block_table_offset_pack_cur  = n_block * kBlockP - block_table_idx_cur * params.page_block_size_pack;
        //         const int block_table_idx_next         = (n_block - 1) * kBlockN / params.page_block_size;
        //         const int block_table_offset_next      = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
        //         const int block_table_offset_pack_next = (n_block - 1) * kBlockP - block_table_idx_next * params.page_block_size_pack;
        //         tKgK_pack.data()   = tKgK_pack.data()   + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.K_pack_batch_stride + (block_table_offset_pack_next - block_table_offset_pack_cur) * params.K_pack_row_stride;
        //         tKgK_params.data() = tKgK_params.data() + (-int(kBlockK_params * params.k_params_row_stride));
        //     }
        //     cute::copy(gmem_tiled_copy_k_pack, tKgK_pack, tKsK_pack);
        //     cute::copy(gmem_tiled_copy_k_params, tKgK_params, tKsK_params);
        //     // This cp_async_fence needs to be in the if block, otherwise the synchronization
        //     // isn't right and we get race conditions.
        //     cute::cp_async_fence();
        // }

        // // We have key_padding_mask so we'll need to Check_inf
        // masking_step == 0
        //     ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, sReduce_tmp, params.scale_softmax_log2)
        //     : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, sReduce_tmp, params.scale_softmax_log2);

        // // Convert acc_s from fp32 to fp16/bf16
        // Tensor acc_s_fp16 = flash::convert_type<Element>(acc_s);

        // auto tCrAcc_r2s = r2s_thr_copy_c.retile_S(acc_s_fp16);  

        // // copy acc_s from register to shared memory
        // clear(tCsAcc_r2s);
        // cute::copy(r2s_tiled_copy_c, tCrAcc_r2s, tCsAcc_r2s);
        // __syncthreads();

        // flash::gemm_Vtensor<num_bits>(
        //     acc_o, tSrAcc, 
        //     tOrVt_pack, tOrVt_dequant, 
        //     tScales_v_h2, tZeros_v_h2, sV_params,
        //     tSsAcc_view, 
        //     tOsVt_pack, 
        //     tiled_mma, 
        //     smem_tiled_copy_Acc, 
        //     smem_tiled_copy_V_pack, 
        //     smem_thr_copy_Acc, 
        //     smem_thr_copy_V_pack,
        //     num_params
        // );  

    }

    // // These are the iterations where we don't need masking on S
    // for (; n_block >= n_block_min; --n_block) {
    //     Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    //     clear(acc_s);
    //     flash::cp_async_wait<0>();
    //     __syncthreads();
        
    //     // Advance gV
    //     if (block_table == nullptr) {
    //         tVgV_pack.data() = tVgV_pack.data()     + (-int(kBlockN * params.v_pack_row_stride));
    //         tVgV_params.data() = tVgV_params.data() + (-int(kBlockN * params.v_params_row_stride));
    //     } else {
    //         const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
    //         const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
    //         const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
    //         const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
    //         tVgV_pack.data() = tVgV_pack.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_pack_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_pack_row_stride;
    //         tVgV_params.data() = tVgV_params.data() + (-int(kBlockN * params.v_params_row_stride));
    //     }

    //     cute::copy(gmem_tiled_copy_v_pack, tVgV_pack, tVsV_pack);
    //     cute::copy(gmem_tiled_copy_v_params, tVgV_params, tVsV_params);
    //     cute::cp_async_fence();

    //     if (Kernel_traits::quant_mode == 1) {
    //         // cute::copy(smem_tiled_copy_K_pack, tSsK_pack, tSrK_pack);
    //         // quant::load_params_Kchannel(tScales_k_h2_c, tZeros_k_h2_c, sK_params, tidx, 0, num_params);
    //         flash::gemm_Kchannel<num_bits>(
    //             acc_s, tSrQ, 
    //             tSrK_pack, tSrK_dequant, 
    //             tScales_k_h2_c, tZeros_k_h2_c, sK_params,
    //             tSsQ, 
    //             tSsK_pack, 
    //             tiled_mma, 
    //             smem_tiled_copy_Q, 
    //             smem_tiled_copy_K_pack,
    //             smem_thr_copy_Q, 
    //             smem_thr_copy_K_pack,
    //             num_params
    //         );
    //     } else {
    //         // cute::copy(smem_tiled_copy_K_pack, tSsK_pack, tSrK_pack);
    //         // quant::load_params_Ktensor(tScales_k_h2_g, tZeros_k_h2_g, sK_params, tidx, num_params);
    //         // flash::gemm_Ktensor(
    //         //     acc_s, tSrQ, 
    //         //     tSrK_pack, tSrK_dequant, 
    //         //     tScales_k_h2_g, tZeros_k_h2_g,
    //         //     tSsQ, 
    //         //     tSsK_pack, 
    //         //     tiled_mma, 
    //         //     smem_tiled_copy_Q, 
    //         //     smem_tiled_copy_K_pack,
    //         //     smem_thr_copy_Q, 
    //         //     smem_thr_copy_K_pack,
    //         //     group_size
    //         // );
    //     }

    //     if constexpr (Is_softcap){
    //         apply_softcap(acc_s, params.softcap);
    //     }

    //     flash::cp_async_wait<0>();
    //     __syncthreads();
        
    //     if (n_block > n_block_min) {
    //         // Advance gK
    //         if (block_table == nullptr) {
    //             tKgK_pack.data()       = tKgK_pack.data()     + (-int(kBlockP * params.K_pack_row_stride));
    //             tKgK_params.data()     = tKgK_params.data()     + (-int(kBlockK_params * params.k_params_row_stride));

    //         } else {
    //             const int block_table_idx_cur          = n_block * kBlockN / params.page_block_size;
    //             const int block_table_offset_cur       = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
    //             const int block_table_offset_pack_cur  = n_block * kBlockP - block_table_idx_cur * params.page_block_size_pack;
    //             const int block_table_idx_next         = (n_block - 1) * kBlockN / params.page_block_size;
    //             const int block_table_offset_next      = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
    //             const int block_table_offset_pack_next = (n_block - 1) * kBlockP - block_table_idx_next * params.page_block_size_pack;
    //             tKgK_pack.data()   = tKgK_pack.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.K_pack_batch_stride + (block_table_offset_pack_next - block_table_offset_pack_cur) * params.K_pack_row_stride;
    //             tKgK_params.data() = tKgK_params.data() + (-int(kBlockK_params * params.k_params_row_stride));
    //         }
    //         cute::copy(gmem_tiled_copy_k_pack, tKgK_pack, tKsK_pack);
    //         cute::copy(gmem_tiled_copy_k_params, tKgK_params, tKsK_params);
    //         // This cp_async_fence needs to be in the if block, otherwise the synchronization
    //         // isn't right and we get race conditions.
    //         cute::cp_async_fence();
    //     }

    //     softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, sReduce_tmp, params.scale_softmax_log2);

    //     Tensor acc_s_fp16 = flash::convert_type<Element>(acc_s);

    //     auto tCrAcc_r2s = r2s_thr_copy_c.retile_S(acc_s_fp16);  

    //     // copy acc_s from register to shared memory
    //     clear(tCsAcc_r2s);
    //     cute::copy(r2s_tiled_copy_c, tCrAcc_r2s, tCsAcc_r2s);
    //     __syncthreads();

    //     flash::gemm_Vtensor<num_bits>(
    //         acc_o, tSrAcc, 
    //         tOrVt_pack, tOrVt_dequant, 
    //         tScales_v_h2, tZeros_v_h2, sV_params,
    //         tSsAcc_view, 
    //         tOsVt_pack, 
    //         tiled_mma, 
    //         smem_tiled_copy_Acc, 
    //         smem_tiled_copy_V_pack, 
    //         smem_thr_copy_Acc, 
    //         smem_thr_copy_V_pack,
    //         num_params
    //     );

    // }

    // // Epilogue
    // __syncthreads();
    // Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, sReduce_tmp, params.scale_softmax);
    // __syncthreads();

    // Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // // Partition sO to match the accumulator partitioning
    // using SmemTiledCopyO = std::conditional_t<
    //     !Split,
    //     typename Kernel_traits::SmemCopyAtomO,
    //     typename Kernel_traits::SmemCopyAtomOaccum
    // >;
    // auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    // auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    // Tensor rO = flash::convert_type<ElementO>(acc_o);
    // Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);             // ((Atom,AtomNum), MMA_M, MMA_N)
    // Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // // sOaccum is larger than sQ, so we need to syncthreads here
    // // TODO: allocate enough smem for sOaccum
    // if constexpr (Split) { __syncthreads(); }

    // cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    // const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
    //     + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    // const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
    //                                      + m_block * kBlockM) * params.d_rounded;
    // const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
    //         ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
    //     ) + m_block * kBlockM;

    // Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
    //                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                              make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    // Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
    //                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    // // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    // GmemTiledCopyO gmem_tiled_copy_Oaccum;
    // auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    // Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);                // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    // __syncthreads();

    // Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    // cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    // Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Tensor taccOcO = thr_mma.partition_C(caccO);                                  // (MMA,MMA_M,MMA_K)
    // static_assert(decltype(size<0>(taccOcO))::value == 4);
    // // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    // if (get<1>(taccOcO_row(0)) == 0) {
    //     #pragma unroll
    //     for (int mi = 0; mi < size(lse); ++mi) {
    //         const int row = get<0>(taccOcO_row(mi));
    //         if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
    //     }
    // }

    // // Construct identity layout for sO
    // Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // // Repeat the partitioning with identity layouts
    // Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // if (!Is_even_K) {
    //     #pragma unroll
    //     for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    // }
    // // Clear_OOB_K must be false since we don't want to write zeros to gmem
    // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    //     gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    // );

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_qpack_1rowblock(const Params &params, const int bidb, const int bidh, const int blockN_idx) {

    // using Element       = typename Kernel_traits::Element;
    // using ElementKVPack = typename Kernel_traits::ElementKVPack;
    // using SharedStorage = typename Kernel_traits::SharedStorage;
    // using index_t       = typename Kernel_traits::index_t;

    // // Shared memory.
    // extern __shared__ char smem_[];
    // SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    // // The thread index.
    // const int tidx = threadIdx.x;

    // constexpr int kBlockN           = Kernel_traits::kBlockN;
    // constexpr int kBlockP           = Kernel_traits::kBlockP;
    // constexpr int kBlockK_params    = Kernel_traits::kBlockK_params;
    // constexpr int kHeadDim          = Kernel_traits::kHeadDim;
    // constexpr int kHeadDim_pack     = Kernel_traits::kHeadDim_pack;
    // constexpr int kHeadDim_k        = Kernel_traits::kHeadDim_k;
    // constexpr int kHeadDim_k_params = Kernel_traits::kHeadDim_k_params;
    // constexpr int kHeadDim_v_params = Kernel_traits::kHeadDim_v_params;
    // constexpr int kNWarps           = Kernel_traits::kNWarps;
    // constexpr int tile_paramsk_j    = Kernel_traits::tile_paramsk_j;
    // constexpr int tile_paramsk_k    = Kernel_traits::tile_paramsk_k;
    // constexpr int tile_paramsk_g    = Kernel_traits::tile_paramsk_g;
    // constexpr int tile_paramsv_k    = Kernel_traits::tile_paramsv_k;
    // constexpr int num_bits          = Kernel_traits::num_bits;
    // constexpr int group_size        = Kernel_traits::group_size;
    // constexpr int num_params        = Kernel_traits::num_params;

    // const BlockInfo binfo(params, bidb);

    // const int bidb_cache              = bidb;
    // const int *block_table            = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    // const int block_table_idx         = block_table        == nullptr ? 0 : blockN_idx * kBlockN / params.page_block_size;
    // const int block_table_offset      = block_table        == nullptr ? 0 : blockN_idx * kBlockN - block_table_idx * params.page_block_size;
    // const int block_table_offset_pack = block_table        == nullptr ? 0 : blockN_idx * kBlockP - block_table_idx * params.page_block_size_pack;

    // const index_t row_offset_k      = block_table == nullptr
    //     ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
    //       + blockN_idx * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
    //     : block_table[block_table_idx] * params.k_batch_stride      + block_table_offset * params.k_row_stride           + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    // const index_t row_offset_k_pack = block_table == nullptr
    //     ?  binfo.k_offset(params.K_pack_batch_stride, params.K_pack_row_stride, bidb_cache)
    //       + blockN_idx * kBlockP * params.K_pack_row_stride + (bidh / params.h_h_k_ratio) * params.K_pack_head_stride
    //     : block_table[block_table_idx] * params.K_pack_batch_stride + block_table_offset_pack * params.K_pack_row_stride + (bidh / params.h_h_k_ratio) * params.K_pack_head_stride;
    // const index_t row_offset_k_params = binfo.k_offset(params.k_params_batch_stride, params.k_params_row_stride, bidb)
    //       + blockN_idx * kBlockK_params * params.k_params_row_stride + (bidh / params.h_h_k_ratio) * params.k_params_head_stride;

    // const index_t row_offset_v        = block_table == nullptr
    //     ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
    //       + blockN_idx * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
    //     : block_table[block_table_idx] * params.v_batch_stride      + block_table_offset * params.v_row_stride     + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    // const index_t row_offset_v_pack   = block_table == nullptr
    //     ? binfo.k_offset(params.v_pack_batch_stride, params.v_pack_row_stride, bidb_cache)
    //       + blockN_idx * kBlockN * params.v_pack_row_stride + (bidh / params.h_h_k_ratio) * params.v_pack_head_stride
    //     : block_table[block_table_idx] * params.v_pack_batch_stride + block_table_offset * params.v_pack_row_stride + (bidh / params.h_h_k_ratio) * params.v_pack_head_stride;
    // const index_t row_offset_v_params = binfo.k_offset(params.v_params_batch_stride, params.v_params_row_stride, bidb)
    //       + blockN_idx * kBlockN * params.v_params_row_stride + (bidh / params.h_h_k_ratio) * params.v_params_head_stride;

    // // Tensor, global memory
    // Tensor gK           = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
    //                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
    //                        make_stride(params.k_row_stride, _1{}));
    // Tensor gK_pack      = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKVPack*>(params.K_pack_ptr) + row_offset_k_pack),
    //                        Shape<Int<kBlockP>, Int<kHeadDim_k>>{},
    //                        make_stride(params.K_pack_row_stride, _1{}));
    // Tensor gK_params    = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.k_params_ptr) + row_offset_k_params),
    //                        Shape<Int<kBlockK_params>, Int<kHeadDim_k_params>>{},
    //                        make_stride(params.k_params_row_stride, params.k_params_dim_stride));

    // Tensor gV           = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v),
    //                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
    //                        make_stride(params.v_row_stride, _1{}));
    // Tensor gV_pack      = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKVPack*>(params.v_pack_ptr) + row_offset_v_pack),
    //                        Shape<Int<kBlockN>, Int<kHeadDim_pack>>{},
    //                        make_stride(params.v_pack_row_stride, _1{}));
    // Tensor gV_params    = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.v_params_ptr) + row_offset_v_params),
    //                        Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{},
    //                        make_stride(params.v_params_row_stride, params.v_params_dim_stride));

    // Tensor sK           = make_tensor(make_smem_ptr(shared_storage.smem_K.data()), typename Kernel_traits::SmemLayoutKV{});
    // Tensor sV           = make_tensor(make_smem_ptr(shared_storage.smem_V.data()), typename Kernel_traits::SmemLayoutKV{});
    // Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    
    // Tensor sK_pack            = make_tensor(make_smem_ptr(shared_storage.smem_Kpack.data()), typename Kernel_traits::SmemLayoutKPack{});
    // Tensor sK_pack_transposed = make_tensor(sK_pack.data(), typename Kernel_traits::SmemLayoutKPacktransposed{});
    // Tensor sV_pack            = make_tensor(make_smem_ptr(shared_storage.smem_Vpack.data()), typename Kernel_traits::SmemLayoutVPack{});
    // Tensor sVt_pack           = make_tensor(sV_pack.data(), typename Kernel_traits::SmemLayoutVPacktransposed{});
    // Tensor sVtNoSwizzle_pack  = make_tensor(sV_pack.data().get(), typename Kernel_traits::SmemLayoutVPacktransposedNoSwizzle{});

    // Tensor sReduce_tmp        = make_tensor(make_smem_ptr(shared_storage.smem_reduce_tmp.data()), typename Kernel_traits::SmemLayoutReduce_tmp{});

    // //
    // // copy: global - shared
    // //
    // typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    // auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    // Tensor tKgK            = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    // Tensor tKsK            = gmem_thr_copy_QKV.partition_D(sK);
    // Tensor tVgV            = gmem_thr_copy_QKV.partition_S(gV);
    // Tensor tVsV            = gmem_thr_copy_QKV.partition_D(sV);

    // typename Kernel_traits::GmemTileCopyK_Pack gmem_tiled_copy_k_pack;
    // auto gmem_thr_copy_k_pack = gmem_tiled_copy_k_pack.get_thread_slice(tidx);
    // Tensor tKsK_pack_s2g      = gmem_thr_copy_k_pack.partition_S(sK_pack);
    // Tensor tKgK_pack_s2g      = gmem_thr_copy_k_pack.partition_D(gK_pack);
    // Tensor tKgK_pack_g2s      = gmem_thr_copy_k_pack.partition_S(gK_pack);
    // Tensor tKsK_pack_g2s      = gmem_thr_copy_k_pack.partition_D(sK_pack);

    // typename Kernel_traits::GmemTileCopyV_Pack gmem_tiled_copy_v_pack;
    // auto gmem_thr_copy_v_pack = gmem_tiled_copy_v_pack.get_thread_slice(tidx);
    // Tensor tVsV_pack_s2g      = gmem_thr_copy_v_pack.partition_S(sV_pack);
    // Tensor tVgV_pack_s2g      = gmem_thr_copy_v_pack.partition_D(gV_pack);
    // Tensor tVgV_pack_g2s      = gmem_thr_copy_v_pack.partition_S(gV_pack);
    // Tensor tVsV_pack_g2s      = gmem_thr_copy_v_pack.partition_D(sV_pack);

    // //
    // // Tensor: Register per thread
    // //
    
    // typename Kernel_traits::TiledMma tiled_mma;
    // typename Kernel_traits::TiledMmaK_i4 tiled_mma_i4;
    // auto thr_mma          = tiled_mma.get_thread_slice(tidx);
    // auto thr_mma_i4       = tiled_mma_i4.get_thread_slice(tidx);
    // Tensor tSrK           = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    // Tensor tSrK_dequant   = thr_mma.partition_fragment_B(sK);
    // Tensor tSrK_pack_tmp  = thr_mma_i4.partition_fragment_B(sK_pack_transposed);                      // (MMA,MMA_N,MMA_K)
    // Tensor tSrK_pack      = make_fragment_like<ElementKVPack>(tSrK_pack_tmp);

    // Tensor tSrV           = thr_mma.partition_fragment_B(sVtNoSwizzle);
    // Tensor tSrV_dequant   = thr_mma.partition_fragment_B(sVtNoSwizzle);
    // Tensor tSrV_pack_tmp  = thr_mma_i4.partition_fragment_B(sVtNoSwizzle_pack);
    // Tensor tSrV_pack      = make_fragment_like<ElementKVPack>(tSrV_pack_tmp);

    // //
    // // copy: shared - register
    // //

    // auto smem_tiled_copy_K       = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // auto smem_thr_copy_K         = smem_tiled_copy_K.get_thread_slice(tidx);
    // Tensor tSsK                  = smem_thr_copy_K.partition_S(sK);
    // Tensor tSrK_view             = smem_thr_copy_K.retile_D(tSrK);
    
    // auto smem_tiled_copy_kv_pack = make_tiled_copy_B(typename Kernel_traits::R2SCopyAtomPack{}, tiled_mma_i4);
    // auto smem_thr_copy_kv_pack   = smem_tiled_copy_kv_pack.get_thread_slice(tidx);
    // Tensor tSrK_pack_r2s_view    = smem_thr_copy_kv_pack.retile_S(tSrK_pack);
    // Tensor tSsK_pack_r2s         = smem_thr_copy_kv_pack.partition_D(sK_pack);
    // Tensor tSrV_pack_r2s_view    = smem_thr_copy_kv_pack.retile_S(tSrV_pack);
    // Tensor tSsV_pack_r2s         = smem_thr_copy_kv_pack.partition_D(sVt_pack);
    // Tensor tSsK_pack_s2r         = smem_thr_copy_kv_pack.partition_S(sK_pack);
    // Tensor tSrK_pack_s2r_view    = smem_thr_copy_kv_pack.retile_D(tSrK_pack);

    // auto smem_tiled_copy_V       = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    // auto smem_tiled_copy_V_pack  = make_tiled_copy_B(typename Kernel_traits::R2SCopyAtomPack{}, tiled_mma_i4);
    // auto smem_thr_copy_V         = smem_tiled_copy_V.get_thread_slice(tidx);
    // auto smem_thr_copy_V_pack    = smem_tiled_copy_V_pack.get_thread_slice(tidx);
    // Tensor tSsV                  = smem_thr_copy_V.partition_S(sVt);
    // Tensor tSrV_view             = smem_thr_copy_V.retile_D(tSrV);
    // Tensor tSsV_pack_s2r         = smem_thr_copy_V_pack.partition_S(sVt_pack);
    // Tensor tSrV_pack_s2r_view    = smem_thr_copy_V_pack.retile_D(tSrV_pack);
 
    // // Advance gK
    // cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    
    // cute::cp_async_fence();
    // flash::cp_async_wait<0>();                               
    // __syncthreads();

    // // Advance gV
    // cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // cute::cp_async_fence();

    // cute::copy(smem_tiled_copy_K, tSsK, tSrK_view);

    // // quantize kv
    // using TensorParamsKC = decltype(make_tensor<half_t>(make_shape(Int<4 * num_params>{}, Int<tile_paramsk_k>{})));
    // using TensorParamsVG = decltype(make_tensor<half_t>(make_shape(Int<num_bits * num_params>{}, Int<tile_paramsv_k>{}))); // TODO: need to change, hardcode num_bits
    // using TensorParamsG  = decltype(make_tensor<half_t>(make_shape(Int<tile_paramsk_g>{})));
    
    // TensorParamsKC tScales_k_c, tZeros_k_c;
    // TensorParamsVG tScales_v_c, tZeros_v_c;
    // TensorParamsG  tScales_k_g, tZeros_k_g;

    // if (Kernel_traits::quant_mode == 1) {
    //     quant::qpack_Kchannel_Vtensor<num_bits>(tSrK, tSrK_pack, tScales_k_c, tZeros_k_c, sReduce_tmp, num_params);
    // } else {
    //     quant::quant_Ktensor(tSrK, tSrK_pack, tScales_k_g, tZeros_k_g, num_params);
    // }
    
    // auto tScales_k_h2_c = cute::recast<__half2>(tScales_k_c);
    // auto tZeros_k_h2_c  = cute::recast<__half2>(tZeros_k_c);
    // auto tScales_k_h2_g = cute::recast<__half2>(tScales_k_g);
    // auto tZeros_k_h2_g  = cute::recast<__half2>(tZeros_k_g);

    // auto tScales_v_h2   = cute::recast<__half2>(tScales_v_c);
    // auto tZeros_v_h2    = cute::recast<__half2>(tZeros_v_c);

    // flash::cp_async_wait<0>();                               
    // __syncthreads();
    // cute::copy(smem_tiled_copy_V, tSsV, tSrV_view);

    // quant::qpack_Kchannel_Vtensor<num_bits>(tSrV, tSrV_pack, tScales_v_c, tZeros_v_c, sReduce_tmp, num_params);

    // const int num_params_2 = num_bits == 2 ? num_params / 2 : num_params;
    // CUTE_UNROLL
    // for (int i = 0; i < size<1>(tScales_v_h2); ++i) {
    //     CUTE_UNROLL
    //     for (int j = 0; j < size<0>(tScales_v_h2); ++j) {
    //         gV_params(128 * (i / 8) + 0  + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2) = tScales_v_h2(j, i);
    //         gV_params(128 * (i / 8) + 64 + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2) = tZeros_v_h2(j, i);
    //     }
    // }

    // if (Kernel_traits::quant_mode == 1) {
    //     CUTE_UNROLL
    //     for (int i = 0; i < size<1>(tScales_k_h2_c); ++i) {
    //         CUTE_UNROLL
    //         for (int j = 0; j < size<0>(tScales_k_h2_c); ++j) {
    //             gK_params(j % num_params, 0  + 8 * i + 4 * (j / num_params) + tidx % 4) = tScales_k_h2_c(j, i);
    //             gK_params(j % num_params, 64 + 8 * i + 4 * (j / num_params) + tidx % 4) = tZeros_k_h2_c(j, i);
    //         }
    //     }
    // } else {
    //     CUTE_UNROLL
    //     for (int j = 0; j < size<0>(tScales_k_h2_g); ++j) {
    //         gK_params(0  + 32 * (j / num_params) + tidx / 4, j % num_params) = tScales_k_h2_g(j);
    //         gK_params(64 + 32 * (j / num_params) + tidx / 4, j % num_params) = tZeros_k_h2_g(j);
    //     }
    // }
    
    // // copy from register to shared memory
    // cute::copy(smem_tiled_copy_kv_pack, tSrK_pack_r2s_view, tSsK_pack_r2s);
    // __syncthreads();
    // if (kHeadDim == 128 && num_bits == 2) {
    //     if (tidx < 64) {
    //         cute::copy(smem_tiled_copy_kv_pack, tSrV_pack_r2s_view, tSsV_pack_r2s);
    //     }
    // } else {
    //     cute::copy(smem_tiled_copy_kv_pack, tSrV_pack_r2s_view, tSsV_pack_r2s);
    // }

    // // copy from shared to global
    // __syncthreads();
    // cute::copy(gmem_tiled_copy_k_pack, tKsK_pack_s2g, tKgK_pack_s2g);
    // __syncthreads();
    // cute::copy(gmem_tiled_copy_v_pack, tVsV_pack_s2g, tVgV_pack_s2g);
    // __syncthreads();
    // // //////////////////////////////////////////////////////////////////////////////
    // // // verify the quantize
    // // // clear(tSrK_pack);
    // // // clear(tSsK_pack_r2s);
    // // // clear(tSrV_pack);
    // // // clear(tSsV_pack_r2s);

    // // // __syncthreads();
    // // // cute::copy(gmem_tiled_copy_k_pack, tKgK_pack_g2s, tKsK_pack_g2s);
    // // // cute::copy(gmem_tiled_copy_v_pack, tVgV_pack_g2s, tVsV_pack_g2s);

    // // // __syncthreads();
    // // // cute::copy(smem_tiled_copy_kv_pack, tSsK_pack_s2r, tSrK_pack_s2r_view);
    // // // cute::copy(smem_tiled_copy_V_pack, tSsV_pack_s2r, tSrV_pack_s2r_view); 

    // // // __syncthreads();
    // // // clear(tScales_k_h2_c);
    // // // clear(tZeros_k_h2_c);
    // // // clear(tScales_k_h2_g);
    // // // clear(tZeros_k_h2_g);
    
    // // // clear(tScales_v_h2);
    // // // clear(tZeros_v_h2);

    // // // CUTE_UNROLL
    // // // for (int i = 0; i < size<1>(tScales_v_h2); ++i) {
    // // //     CUTE_UNROLL
    // // //     for (int j = 0; j < size<0>(tScales_v_h2); ++j) {
    // // //         tScales_v_h2(j, i) = gV_params(128 * (i / 8) + 0  + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2);
    // // //         tZeros_v_h2(j, i)  = gV_params(128 * (i / 8) + 64 + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2);
    // // //     }
    // // // }

    // // CUTE_UNROLL
    // // for (int i = 0; i < size<2>(tSrV_pack); ++i) {  
    // //    quant::dequant_Kchannel_Vtensor<num_bits>(tSrV_pack(_,_,i), tSrV_dequant(_,_,i), tScales_v_c(_,i), tZeros_v_c(_,i), num_params);
    // // }

    // // if (Kernel_traits::quant_mode == 1) {
    //     // CUTE_UNROLL
    //     // for (int i = 0; i < size<1>(tScales_k_h2_c); ++i) {
    //     //     CUTE_UNROLL
    //     //     for (int j = 0; j < size<0>(tScales_k_h2_c); ++j) {
    //     //         tScales_k_h2_c(j, i) = gK_params(j % num_params, 0  + 8 * i + 4 * (j / num_params) + tidx % 4);
    //     //         tZeros_k_h2_c(j, i)  = gK_params(j % num_params, 64 + 8 * i + 4 * (j / num_params) + tidx % 4);
    //     //     }
    //     // }

    //     // CUTE_UNROLL
    //     // for (int i = 0; i < size<2>(tSrK_pack); ++i) {
    //     //     quant::dequant_Kchannel_Vtensor<num_bits>(tSrK_pack(_,_,i), tSrK_dequant(_,_,i), tScales_k_c(_,i), tZeros_k_c(_,i), num_params);
    //     // }
    // // } else {
    //     // CUTE_UNROLL
    //     // for (int j = 0; j < size<0>(tScales_k_h2_g); ++j) {
    //     //     tScales_k_h2_g(j) = gK_params(0  + 32*j + tidx/4, 0);
    //     //     tZeros_k_h2_g(j)  = gK_params(64 + 32*j + tidx/4, 0);
    //     // }

    //     // auto tScales_k_h1_g = cute::recast<__half>(tScales_k_h2_g);
    //     // auto tZeros_k_h1_g = cute::recast<__half>(tZeros_k_h2_g);

    //     // CUTE_UNROLL
    // //     for (int i = 0; i < size<2>(tSrK_pack); ++i) {
    // //         quant::dequantize_Ktensor(tSrK_pack, tSrK_dequant, tScales_k_h2_g, tZeros_k_h2_g, 4, group_size, i);
    // //     }
    // // }

    // // // //////////////////////////////////////////////////////////////////////////////
    // #if DEBUG2
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("num_params = %d\n", num_params);
    //     // PRINT("gK", gK.layout());                                       // PRINTTENSOR("gK", gK);
        
    //     // PRINT("sK", sK.layout());                                    // PRINTTENSOR("sK", sK);
    //     // PRINT("sK_pack", sK_pack.layout());                             // PRINTTENSOR("sK_pack", sK_pack);
    //     // // // PRINT("sK_pack_transposed", sK_pack_transposed.layout());    // PRINTTENSOR("sK_pack_transposed", sK_pack_transposed);
    //     // // PRINT("gK_params", gK_params.layout());
    //     // // PRINT("tScales_k_c", tScales_k_c.layout());                     PRINTTENSOR("tScales_k_c", tScales_k_c);
    //     // // PRINT("tZeros_k_c",  tZeros_k_c.layout());                      PRINTTENSOR("tZeros_k_c", tZeros_k_c);
    //     PRINT("tSrK", tSrK.layout());                                   PRINTTENSOR("tSrK", tSrK);
    //     // PRINT("tSrK_pack", tSrK_pack.layout());                         // PRINTTENSOR("tSrK_pack", tSrK_pack(_,_,0));
    //     PRINT("tSrK_dequant", tSrK_dequant.layout());                   PRINTTENSOR("tSrK_dequant", tSrK_dequant);
    //     PRINT("gK_pack", gK_pack.layout());                             PRINTTENSOR("gK_pack", gK_pack);

    //     // PRINT("gV", gV.layout());
    //     // PRINT("gV_pack", gV_pack.layout());                           // PRINTTENSOR("gV_pack", gV_pack);
    //     // PRINT("sV_pack", sV_pack.layout());                           // PRINTTENSOR("sV_pack", sV_pack);
    //     // PRINT("sV", sV.layout());   // PRINTTENSOR("sV", sV);
        
    //     // PRINT("gV_params", gV_params.layout());
    //     // PRINT("tScales_v_c", tScales_v_c.layout());                   // PRINTTENSOR("tScales_v_c", tScales_v_c);
    //     // PRINT("tZeros_v_c", tZeros_v_c.layout());                     // PRINTTENSOR("tZeros_v_c", tZeros_v_c);
    //     // PRINT("tSrV", tSrV.layout());                                    PRINTTENSOR("tSrV", tSrV);
    //     // PRINT("tSrV_dequant", tSrV_dequant.layout());                    PRINTTENSOR("tSrV_dequant", tSrV_dequant);
    //     // PRINT("tSrV_pack", tSrV_pack.layout());                          PRINTTENSOR("tSrV_pack", tSrV_pack);
    //     // PRINT("gV_pack", gV_pack.layout());                              PRINTTENSOR("gV_pack", gV_pack);
    //     // PRINT("tSrV_pack_r2s_view", tSrV_pack_r2s_view.layout());     // PRINTTENSOR("tSrV_pack_r2s_view", tSrV_pack_r2s_view);
    //     // PRINT("tSsV_pack_r2s", tSsV_pack_r2s.layout());               // PRINTTENSOR("tSsV_pack_r2s", tSsV_pack_r2s);
        
    //     // PRINT("sVt_pack", sVt_pack.layout());         // PRINTTENSOR("sVt_pack", sVt_pack(_,_,0));
    //     // PRINT("tSrV_pack_r2s_view", tSrV_pack_r2s_view.layout());
    //     // PRINT("tSsV_pack_r2s", tSsV_pack_r2s.layout());
    //     // PRINT("tScales_v_h2", tScales_v_h2.layout());
    //     // PRINT("tZeros_v_h2", tZeros_v_h2.layout()); 
    //     printf("#####################################################################################\n");
    // }
    // #endif
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

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Paged_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb    = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh         = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx  = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Paged_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = params.b * params.h * params.seqlen_q;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));

    // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
    // This tensor's layout maps row_offset_lse to {bidb, bidh, q_offset}.
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // This layout maps row_offset_lse to {bidh, q_offset, bidb} or {bidh, bidb, q_offset}.
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = params.seqlenq_ngroups_swapped ? make_stride(params.b, params.seqlen_q * params.b, 1) : make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));

    Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout);

    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then transpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
            gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
        }
    }
    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});

    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        flash::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
        // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }
    // if (cute::thread0()) { print_tensor(tOrO); }

    Tensor rO = flash::convert_type<Element>(tOrO);
    // Write to gO
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.b * params.h * params.seqlen_q) {
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                            Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                    // TODO: Should check if this is using vectorized store, but it seems pretty fast
                    copy(rO(_, m, k), gO);
                    // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
                    // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
                }
            }
        }
    }
}

} // namespace flash