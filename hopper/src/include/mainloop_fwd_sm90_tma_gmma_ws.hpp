/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "flash.h"
#include "named_barrier.hpp"
#include "seqlen.h"
#include "mask.h"
#include "pack_gqa.h"
#include "paged_kv.h"
#include "rotary.h"
#include "utils.h"
#include "dequantize.h"
#include "sm90_pipeline_no_cluster.hpp"

namespace flash {

using namespace cute;

template <int Stages, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKV_, bool AppendKV_,
        bool Mma1_is_RS, bool IntraWGOverlap, bool PackGQA_, bool Split_, bool V_colmajor_,
        int quant_mode_=0, int num_bits_=4, int group_size_=128>
struct CollectiveMainloopFwdSm90 {

    static constexpr int kStages = Stages;
    using index_t = int64_t;
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementKVPack = cute::uint16_t;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKV = PagedKV_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool V_colmajor = V_colmajor_;
    static constexpr bool Transpose_V = Is_FP8 && !V_colmajor;
    static constexpr bool Use_TMA_Q = !PackGQA;
    static constexpr bool Use_TMA_KV = !PagedKV;

    static constexpr int quant_mode = quant_mode_;
    static constexpr int group_size = group_size_;
    static constexpr int num_bits   = num_bits_;
    
    static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
    static_assert(Use_TMA_KV || !V_colmajor, "If not using TMA for KV, V_colmajor is not supported");
    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    static constexpr int pack_num          = 16 / num_bits;
    // static constexpr int kBlockN_Qpack     = num_bits   == 4 ? 128 : 256;
    static constexpr int kBlockN_Qpack     = 128;
    static constexpr int kBlockN_pack      = quant_mode == 1 ? kBlockN / pack_num : kBlockN;
    static constexpr int kBlockN_params    = quant_mode == 1 ? kBlockN / group_size : kBlockN;
    static constexpr int kHeadDim_kpack    = quant_mode == 1 ? kHeadDim : kHeadDim / pack_num;
    static constexpr int kHeadDim_vpack    = kHeadDim / pack_num;
    static constexpr int kHeadDim_k_params = quant_mode == 1 ? kHeadDim : kHeadDim / group_size;
    static constexpr int kHeadDim_v_params = kHeadDim / group_size;
    static constexpr int tile_paramsk_j    = kBlockN / group_size;
    static constexpr int tile_paramsk_m    = kBlockN / kBlockN_Qpack;
    static constexpr int tile_paramsk_g    = kBlockN / 32 * (kBlockN / group_size); // TODO: check
    static constexpr int tile_paramsk_k    = kHeadDim / 16;
    static constexpr int tile_paramsv_k    = kBlockN / 16;  

    static constexpr int num_params = kBlockN_Qpack / group_size; // TODO: check 128

    using TileShape_MNK_Kpack = decltype(make_shape(Int<kBlockM>{}, Int<kBlockN_pack>{}, Int<kHeadDim_kpack>{}));
    using TileShape_MNK_Vpack = decltype(make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kHeadDim_vpack>{}));

    // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
    // Leaving this option here for reference.
    static constexpr bool Mma0_is_RS = false;
    // We can have Mma1 (P @ V) with P in smem in rmem to reduce register pressure at the cost of more smem.
    static_assert(!(!Mma1_is_RS && !IntraWGOverlap), "Mma1 must be RS if IntraWGOverlap is enabled");
    static_assert(!(!Mma1_is_RS && Is_FP8), "Mma1 must be RS if FP8");
    static_assert(!(!Mma1_is_RS && Transpose_V), "Mma1 must be RS if Transpose_V");

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMma0 = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !Mma0_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>())
        >{},
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !Mma1_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                     decltype(select<0, 2, 1>(TileShape_MNK{})), GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                     decltype(select<0, 2, 1>(TileShape_MNK{})), GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutMNK{}));

    /* Dequant */
    using TiledMma0_dequant = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<1>,_4,_1>>,  
        Tile<Int<16>, Int<32>, _16>>;
    using TiledMma0_dequant_r2s = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<1>,_4,_1>>,  
        Tile<Int<16>, Int<128>, _16>>;
    using S2RCopyAtomKPack    = Copy_Atom<SM75_U32x2_LDSM_N, ElementKVPack>;
    using R2SCopyAtomKDequant = Copy_Atom<SM90_U32x4_STSM_N, Element>;
    using S2RCopyAtomVPack    = Copy_Atom<SM75_U16x4_LDSM_T, ElementKVPack>;
    using R2SCopyAtomVDequant = Copy_Atom<SM90_U16x8_STSM_T, Element>;

    static constexpr int NumMmaThreads = size(TiledMma0{});
    static constexpr int NumProducerThreads = !Transpose_V && Use_TMA_KV && Use_TMA_Q ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK     = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomKPack = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementKVPack,
        decltype(cute::get<1>(TileShape_MNK_Kpack{})), decltype(cute::get<2>(TileShape_MNK_Kpack{}))>());
    using SmemLayoutKPack     = decltype(tile_to_shape(
        SmemLayoutAtomKPack{},
        make_shape(shape<1>(TileShape_MNK_Kpack{}), shape<2>(TileShape_MNK_Kpack{}), Int<kStages>{})));

    // using SmemLayoutAtomKPack = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, ElementKVPack,
    //     decltype(cute::get<2>(TileShape_MNK_Kpack{})), decltype(cute::get<1>(TileShape_MNK_Kpack{}))>());
    // using SmemLayoutKPack     = decltype(tile_to_shape(
    //     SmemLayoutAtomKPack{},
    //     make_shape(shape<1>(TileShape_MNK_Kpack{}), shape<2>(TileShape_MNK_Kpack{}), Int<kStages>{})));

    using SmemLayoutKParams_channel = decltype(
        composition(Swizzle<2, 2, 3>{},
                    Layout<Shape<Int<tile_paramsk_j>, Int<kHeadDim>>,
                           Stride<Int<kHeadDim>, Int<1>>>{}));
    using SmemLayoutAtomKParams_group = decltype(
        make_layout(make_shape(Int<32>{}, Int<1>{}),
                    make_stride(Int<1>{}, Int<1>{})));
    using SmemLayoutKParams_group = decltype(tile_to_shape(
        SmemLayoutAtomKParams_group{},
        Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{}));
    using SmemLayoutKParams = SmemLayoutKParams_group;
    
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
        decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));
    using SmemLayoutVtMma = SmemLayoutVt;

    using SmemLayoutAtomVtPack = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, ElementKVPack,
        decltype(cute::get<2>(TileShape_MNK_Vpack{})), decltype(cute::get<1>(TileShape_MNK_Vpack{}))>());
    using SmemLayoutVtPack = decltype(tile_to_shape(
        SmemLayoutAtomVtPack{},
        make_shape(shape<2>(TileShape_MNK_Vpack{}), shape<1>(TileShape_MNK_Vpack{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    using SmemLayoutAtomVParams = decltype(
        make_layout(make_shape(Int<32>{}, Int<1>{}),
                    make_stride(Int<1>{}, Int<1>{})));
    using SmemLayoutVParams = decltype(tile_to_shape(
        SmemLayoutAtomVParams{},
        Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{}));

    // Only used if we're using cp.async to load V
    using SmemLayoutAtomVCpAsync = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutVCpAsync = decltype(tile_to_shape(
        SmemLayoutAtomVCpAsync{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));

    using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

    // Use LDSM.T and STSM to transpose V in the case of FP8 and V being row-major.
    // For FP16/BF16 we don't do any transposing.
    static_assert(!Transpose_V || (kHeadDim % 32 == 0 && kBlockN % 32 == 0));
    static constexpr bool kHeadDim_multiple_64 = kHeadDim % 64 == 0;
    // Either kHeadDim is a multiple of 64 (in which case we use a block size of 64 x 32 for the transpose),
    // or we need kBlockN to be a multiple of 64 (in which case we use a block size of 32 x 64 for the transpose).
    static_assert(!Transpose_V || (kHeadDim_multiple_64 || kBlockN % 64 == 0));
    using LDSM_thread_shape  = std::conditional_t<kHeadDim_multiple_64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
    using LDSM_thread_stride = std::conditional_t<kHeadDim_multiple_64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
    using LDSM_value_shape = Shape<_2, _2, _1, _4>;
    using LDSM_value_stride = Stride<_1, _2, _16, _4>;
    using LDSM_divide_shape = std::conditional_t<kHeadDim_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
    using S2RTiledCopyVt = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
        Layout<LDSM_value_shape, LDSM_value_stride>{}));

    using STSM_thread_shape  = std::conditional_t<kHeadDim_multiple_64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
    using STSM_thread_stride = std::conditional_t<kHeadDim_multiple_64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
    using STSM_value_shape = Shape<_1, _4, _2, _2>;
    using STSM_value_stride = Stride<_0, _1, _4, _8>;
    using STSM_divide_shape = Shape<_8, _16>;
    // These will not permute the columns of V (the kHeadDim dimension) but incur bank conflicts
    // so a little slower (e.g. 1150 TFLOPS for hdim 256 instead of 1200 TFLOPS).
    // Instead we will permute the cols of V, and un-permute the cols of O in the epilogue.
    // using STSM_value_shape = Shape<_2, _4, _1, _2>;
    // using STSM_value_stride = Stride<_4, _1, _0, _8>;
    // using STSM_divide_shape = Shape<_16, _16>;
    using R2STiledCopyV = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
        Layout<STSM_value_shape, STSM_value_stride>{}));

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

    // We use CpAsync for K and V if PagedKV and AppendKV, since TMA doesn't work there
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // We want each thread to have at least 2 loads in the K direction since in the case of non-interleaved
    // rotary (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc), each thread will
    // load twice from the same row.
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKV where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to avoid predication
    static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRow");
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    // using StrideKParams = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideKParams = cute::Stride<_1, int64_t, int64_t, int64_t>;
    using StrideVParams = cute::Stride<_1, int64_t, int64_t, int64_t>;
    using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;
    using StrideDescale = cute::Stride<int64_t, int64_t>;

    using TMA_Q = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}));

    using TMA_K_pack = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<ElementKVPack const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutKPack{}),
        TileShape_MNK_Kpack{},
        ClusterShape{})); 

    // using GmemTileCopyKParams = decltype(
    //     make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, __half2>{},
    //                     make_layout(make_shape(_1{}, _128{}), make_stride(_1{}, _1{})),
    //                     Layout<Shape<_1, _1>>{}));  // Val layout, 4 vals per store

    using GmemTileCopyKParams = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, __half2>{},
                        make_layout(make_shape(_128{}, _1{}), make_stride(_1{}, _1{})),
                        Layout<Shape<_1, _1>>{}));  // Val layout, 4 vals per store
                        
    using GmemTileCopyVParams = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, __half2>{},
                        make_layout(make_shape(_128{}, _1{}), make_stride(_1{}, _1{})),
                        Layout<Shape<_1, _1>>{}));  // Val layout, 4 vals per store

    using TMA_V_pack = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<ElementKVPack const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
        take<0, 2>(SmemLayoutVtPack{}),
        select<2, 1>(TileShape_MNK_Vpack{}),
        size<0>(ClusterShape{})));

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK_original = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{}))     * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK_pack     = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutKPack{})) * cutlass::sizeof_bits_v<ElementKVPack> / 8);
    static constexpr uint32_t TmaTransactionBytesK          = TmaTransactionBytesK_pack;
    static constexpr uint32_t TmaTransactionBytesV_original = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVt{}))     * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV_pack     = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVtPack{})) * cutlass::sizeof_bits_v<ElementKVPack> / 8);
    static constexpr uint32_t TmaTransactionBytesV = TmaTransactionBytesV_pack;
    // static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

    using PipelineTmaAsync = std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1, typename cutlass::PipelineTmaAsyncNoCluster<kStages>, typename cutlass::PipelineTmaAsync<kStages>>;
    using MainloopPipelineK = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineV = std::conditional_t<!Transpose_V && Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineVt = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    // We always use TMA for K_new and V_new
    using MainloopPipelineKVNew = PipelineTmaAsync;
    using PipelineState = cutlass::PipelineState<kStages>;

    // If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q to be aligned
    // and have sQ being position_independent_swizzle_tensor.
    // If !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want smem_k and smem_v to be aligned.
    static constexpr size_t SmemAlignmentQ = Use_TMA_Q && !AppendKV && !Mma0_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
    static constexpr size_t SmemAlignmentK = Use_TMA_KV && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
    static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentVtPack        = cutlass::detail::alignment_for_swizzle(SmemLayoutVtPack{});
    static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128 && SmemAlignmentVtPack >= 128, "Require at least 128B alignment");
    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
    static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

    struct TensorStorageWithoutPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        // cute::array_aligned<ElementKVPack, cute::cosize_v<SmemLayoutKPack>, SmemAlignmentK> smem_k_pack;
        cute::array_aligned<__half2, cute::cosize_v<SmemLayoutKParams>> smem_k_params;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        // cute::array_aligned<ElementKVPack, cute::cosize_v<SmemLayoutVtPack>, SmemAlignmentVtPack> smem_v_pack;
        cute::array_aligned<__half2, cute::cosize_v<SmemLayoutVParams>> smem_v_params;
    };

    using TensorStorage = TensorStorageWithoutPNoTranspose;

    // These are tuned for speed. They don't affect correctness.
    static constexpr bool UseSchedulerBarrier = IntraWGOverlap
        ? (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128)
        : NumMmaWarpGroups == 2;
    static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && (!Is_FP8 || V_colmajor);

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;

        // Element* const ptr_K;  // Not Element const* since we might append to KV cache in-place
        ShapeQKV const shape_K;
        // StrideQK const stride_K;
        ElementKVPack* const ptr_K_pack;
        ShapeQKV const shape_K_pack;
        StrideQK const stride_K_pack;
        __half2* const ptr_K_params;
        ShapeQKV const shape_K_params;
        StrideKParams const stride_K_params;

        // Element* const ptr_V;
        // StrideV const stride_V;
        ElementKVPack* const ptr_V_pack;
        ShapeQKV const shape_V_pack;
        StrideQK const stride_V_pack;
        __half2* const ptr_V_params;
        ShapeQKV const shape_V_params;
        StrideVParams const stride_V_params;
        
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const window_size_left = -1, window_size_right = -1, sink_token_length = 0;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;

        // Element* const ptr_K;
        ShapeQKV const shape_K;
        // StrideQK const stride_K;
        ElementKVPack* const ptr_K_pack;
        ShapeQKV const shape_K_pack;
        StrideQK const stride_K_pack;
        __half2* const ptr_K_params;
        ShapeQKV const shape_K_params;
        StrideKParams const stride_K_params;

        // Element* const ptr_V;
        // StrideV const stride_V;
        ElementKVPack* const ptr_V_pack;
        ShapeQKV const shape_V_pack;
        StrideQK const stride_V_pack;
        __half2* const ptr_V_params;
        ShapeQKV const shape_V_params;
        StrideVParams const stride_V_params;

        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        cutlass::FastDivmod page_size_divmod;
        cutlass::FastDivmod qhead_per_khead_divmod;

        TMA_Q tma_load_Q;

        // TMA_K tma_load_K;
        TMA_K_pack tma_load_K_pack;

        // TMA_V tma_load_V;
        TMA_V_pack tma_load_V_pack;

        // TMA_K tma_load_K_new;
        // TMA_V tma_load_V_new;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        int const window_size_left, window_size_right, sink_token_length;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for Q

        // K
        // Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        // TMA_K tma_load_K = make_tma_copy_B_sm90(
        //     GmemTiledCopyKV{},
        //     mK,
        //     take<0, 2>(SmemLayoutK{}),
        //     TileShape_MNK{},
        //     ClusterShape{}); // mcast along M mode for this N load, if any
        Tensor mK_pack = make_tensor(make_gmem_ptr(args.ptr_K_pack), args.shape_K_pack, args.stride_K_pack);
        TMA_K_pack tma_load_K_pack = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK_pack,
            take<0, 2>(SmemLayoutKPack{}),
            TileShape_MNK_Kpack{},
            ClusterShape{});

        // V Tensor
        // Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), select<1, 0, 2, 3>(args.shape_K), select<1, 0, 2, 3>(args.stride_V));
        // TMA_V tma_load_V = make_tma_copy(
        //     GmemTiledCopyKV{},
        //     mV,
        //     take<0, 2>(SmemLayoutVt{}),
        //     select<2, 1>(TileShape_MNK{}),
        //     size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

        // V_pack Tensor
        Tensor mV_pack = make_tensor(make_gmem_ptr(args.ptr_V_pack), select<1, 0, 2, 3>(args.shape_V_pack), select<1, 0, 2, 3>(args.stride_V_pack));
        TMA_V_pack tma_load_V_pack = make_tma_copy(
            GmemTiledCopyKV{},
            mV_pack,
            take<0, 2>(SmemLayoutVtPack{}),
            select<2, 1>(TileShape_MNK_Vpack{}),
            size<0>(ClusterShape{}));

        // If PackGQA, reshape Q to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        if (get<1>(args.shape_rotary) > 0) {
            assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        }
        assert(args.num_splits >= 1);
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                // args.ptr_K, 
                args.shape_K, 
                // args.stride_K, 
                args.ptr_K_pack, args.shape_K_pack, args.stride_K_pack,
                args.ptr_K_params, args.shape_K_params, args.stride_K_params,
                // args.ptr_V, args.stride_V,
                args.ptr_V_pack, args.shape_V_pack, args.stride_V_pack,
                args.ptr_V_params, args.shape_V_params, args.stride_V_params,
                args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
                args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
                args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
                args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
                cutlass::FastDivmod(int(get<0>(args.shape_K))),
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, 
                // tma_load_K, 
                tma_load_K_pack,
                // tma_load_V, 
                tma_load_V_pack,
                // tma_load_K_new, 
                // tma_load_V_new,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.window_size_left, args.window_size_right, args.sink_token_length,
                !Split ? 1 : args.num_splits,
                args.kv_batch_idx,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA_Q) {
            cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        }
        if constexpr (Use_TMA_KV) {
            cute::prefetch_tma_descriptor(params.tma_load_K_pack.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V_pack.get_tma_descriptor());
        }

    }

    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(Params const& params, SeqlenInfo_t const& seqlen_info,
                                              int m_block, int bidb, int split_idx=0, int num_splits=1) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            int m_idx_max = (m_block + 1) * kBlockM;
            // TODO: check off-by-1 error
            if (PackGQA) { m_idx_max = params.qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div(m_idx_max + seqlen_k - seqlen_q + params.window_size_right, kBlockN));
        }
        int n_block_min = 0;
        if constexpr (Is_local) {
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { m_idx_min = params.qhead_per_khead_divmod.divide(m_idx_min); }
            n_block_min = std::max(int(0), (m_idx_min + seqlen_k - seqlen_q - params.window_size_left) / kBlockN);
        }
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        if constexpr (Split) {
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits);
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
        }
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        return {n_block_min, n_block_max};
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         MainloopPipelineVt pipeline_vt,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int &work_idx
         ) {
        

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_min, n_block_max] = get_n_block_min_max(params, seqlen_info, m_block, bidb, split_idx, params.num_splits);
        // It's possible to have n_block_max <= n_block_min. Loading K can cause illegal memory access.
        
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) {
                scheduler_prefetch();
                return;
            }
        }

        Tensor sQ        = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});

        Tensor sK_pack   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKPack{});
        Tensor sK_params = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k_params.data()), SmemLayoutKParams{});
        
        Tensor sVt       = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutVt{});
        Tensor sVt_pack  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtPack{});
        Tensor sV_params = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v_params.data()), SmemLayoutVParams{});

        // Only used if Transpose_V
        Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutVtMma{}));

        int const thread_idx = threadIdx.x % NumProducerThreads;
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        auto [k_params_row_stride, k_params_dim_stride, k_params_head_stride, k_params_batch_stride] = params.stride_K_params;
        auto [v_params_row_stride, v_params_dim_stride, v_params_head_stride, v_params_batch_stride] = params.stride_V_params;

        const index_t row_offset_k_params = bidb_kv * k_params_batch_stride + bidh_kv * k_params_head_stride
        + (n_block_max - 1) * kBlockN_params * k_params_row_stride;
        const index_t row_offset_v_params = bidb_kv * v_params_batch_stride + bidh_kv * v_params_head_stride
        + (n_block_max - 1) * kBlockN * v_params_row_stride;

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        
        Tensor mQ           = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);

        Tensor mK_TMA_pack  = params.tma_load_K_pack.get_tma_tensor(params.shape_K_pack)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor mVt_TMA_pack = params.tma_load_V_pack.get_tma_tensor(select<1, 0, 2, 3>(params.shape_V_pack))(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor gK_TMA_pack = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK_TMA_pack), select<1, 2>(TileShape_MNK_Kpack{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gK_params   = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.ptr_K_params) + row_offset_k_params),
                           Shape<Int<kBlockN_params>, Int<kHeadDim_k_params>>{},
                           make_stride(k_params_row_stride, k_params_dim_stride));

        Tensor gVt_TMA_pack = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k), mVt_TMA_pack), select<2, 1>(TileShape_MNK_Vpack{}), make_coord(_0{}, _));  // (K, N, _)
        Tensor gV_params    = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.ptr_V_params) + row_offset_v_params),
                                Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{},
                                make_stride(_1{}, v_params_dim_stride));

        auto block_tma_Q      = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ           = group_modes<0, 3>(block_tma_Q.partition_S(gQ));  // (TMA)
        Tensor tQsQ           = group_modes<0, 3>(block_tma_Q.partition_D(sQ));  // (TMA)

        auto block_tma_K_pack = params.tma_load_K_pack.get_slice(cluster_local_block_id.x);
        Tensor tKgK_TMA_pack  = group_modes<0, 3>(block_tma_K_pack.partition_S(gK_TMA_pack));  // (TMA, k)
        Tensor tKsK_TMA_pack  = group_modes<0, 3>(block_tma_K_pack.partition_D(sK_pack));  // (TMA, PIPE)

        auto block_tma_V_pack = params.tma_load_V_pack.get_slice(cluster_local_block_id.x);
        Tensor tVgVt_TMA_pack = group_modes<0, 3>(block_tma_V_pack.partition_S(gVt_TMA_pack));  // (TMA, k)
        Tensor tVsVt_TMA_pack = group_modes<0, 3>(block_tma_V_pack.partition_D(sVt_pack));  // (TMA, PIPE)

        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumProducerThreads, Element, Transpose_V || !IntraWGOverlap /*KV_Same_Iter*/>;

        uint16_t mcast_mask_kv = 0;

        auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            pipeline_k.producer_acquire(smem_pipe_write);
            copy(params.tma_load_K_pack.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                tKgK_TMA_pack(_, n_block), tKsK_TMA_pack(_, smem_pipe_write.index()));
        };

        auto load_V = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
            pipeline_v_load.producer_acquire(smem_pipe_write);
            copy(params.tma_load_V_pack.with(*pipeline_v_load.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                tVgVt_TMA_pack(_, n_block), tVsVt_TMA_pack(_, smem_pipe_write.index()));

        };

        int n_block = n_block_max - 1;

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // If this is true, we're guaranteed that only the first warp will execute this function
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = !Use_TMA_KV || ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync());

        if (should_load_KV) {
            if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/); }
            load_K(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
        }

        // load Q
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        Tensor mQ_s = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
        using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumProducerThreads, Element>;
        PackGQAt::load_Q(mQ_s, sQ_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
        auto &barrier_Q = shared_storage.pipelines.barrier_Q;

        cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Q));
        barrier_Q.arrive();

        // Wait for the MMA WGs to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        // if (thread_idx == 0) { printf("Producer: main load, before barrier_O, work_idx = %d\n", work_idx);}
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        // if (thread_idx == 0) { printf("Producer: main load, after barrier_O\n");}

        if constexpr (!Transpose_V && !IntraWGOverlap) {
            if (should_load_KV) { load_V(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/); }
        }
        int n_block_prev = n_block;
        --n_block;
        #pragma unroll (!Transpose_V && Use_TMA_KV ? 2 : 1)
        for (; n_block >= n_block_min; --n_block) {
            PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
            ++smem_pipe_write;
            if (should_load_KV) {
                if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/); }
                load_K(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                if constexpr (!Transpose_V) {
                    if constexpr (IntraWGOverlap) {
                        load_V(n_block_prev, smem_pipe_write_v, cute::true_type{} /*Seqlenk_mask*/);
                    } else {
                        load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                    }
                }
            }
            n_block_prev = n_block;
        }

        scheduler_prefetch();
        if constexpr (!Transpose_V && IntraWGOverlap) {
            if (should_load_KV) { load_V(n_block_prev, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/); }
        }
        ++smem_pipe_write;
        // At the end, all threads have the correct smem_pipe_write.
        ++work_idx;
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt,
              PipelineState& smem_pipe_write, SharedStorage &shared_storage, int const work_idx) {

        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
            if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (UseSchedulerBarrier) {
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            int const cur_WG = flash::canonical_warp_group_idx_nosync() - 1;
            int const next_WG = NumMmaWarpGroups == 2
                ? 1 - cur_WG
                : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG /*id*/);
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // Tell producers that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        // cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);
        if constexpr (UseSchedulerBarrier) {
            // We have NamedBarrier for up to 3 WGs
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            // WG1 needs the very first signal to start
            if (flash::canonical_warp_group_idx_nosync() == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v,
        PipelineState& smem_pipe_read,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        int &work_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // can't use auto [m_block, ...] = block_coord since structured binding cannot be captured in lambda
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        auto [n_block_min, n_block_max] = get_n_block_min_max(params, seqlen_info, m_block, bidb, split_idx, params.num_splits);

        #if DEBUG
        if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("n_block_min: %d, n_block_max: %d\n", n_block_min, n_block_max);
        }
        #endif

        // It's possible to have n_block_max <= n_block_min. We don't want to load Q or change any barrier
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) { return false; }
        }

        auto [k_params_row_stride, k_params_dim_stride, k_params_head_stride, k_params_batch_stride] = params.stride_K_params;
        auto [v_params_row_stride, v_params_dim_stride, v_params_head_stride, v_params_batch_stride] = params.stride_V_params;

        const index_t row_offset_k_params = bidb_kv * k_params_batch_stride + bidh_kv * k_params_head_stride
        + (n_block_max - 1) * kBlockN_params * k_params_row_stride;
        const index_t row_offset_v_params = bidb_kv * v_params_batch_stride + bidh_kv * v_params_head_stride
        + (n_block_max - 1) * kBlockN * v_params_row_stride;

        Tensor gK_params   = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.ptr_K_params) + row_offset_k_params),
                           Shape<Int<kBlockN_params>, Int<kHeadDim_k_params>>{},
                           make_stride(k_params_row_stride, k_params_dim_stride));
        Tensor gV_params    = make_tensor(make_gmem_ptr(reinterpret_cast<__half2*>(params.ptr_V_params) + row_offset_v_params),
                                Shape<Int<kBlockN>, Int<kHeadDim_v_params>>{},
                                make_stride(_1{}, v_params_dim_stride));

        Tensor sQ        = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});

        Tensor sK        = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sK_pack   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKPack{});
        Tensor sK_params = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k_params.data()), SmemLayoutKParams{});

        Tensor sV        = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
        Tensor sV_pack   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtPack{});
        Tensor sV_params = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v_params.data()), SmemLayoutVParams{});

        Tensor sP = [&] {
            if constexpr (Mma1_is_RS) {
                // We might not have smem_p if !Mma1_is_RS1, just use smem_q as a placeholder since we don't use it
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
            }
        }();

        if constexpr (!Mma0_is_RS) {
            static_assert(stride<0>(typename TiledMma0::ALayout{}) == 0 and
                        stride<0>(typename TiledMma0::BLayout{}) == 0 and
                        size<0>(typename TiledMma0::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                        size<0>(typename TiledMma0::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        }
        constexpr int MmaWarpGroups = size(TiledMma0{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMma0 tiled_mma0;
        TiledMma1 tiled_mma1;
        auto wg_mma0         = tiled_mma0.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma1         = tiled_mma1.get_slice(warp_group_thread_layout(warp_group_idx));

        auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma0);
        auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors"
        Tensor tSrQ = wg_mma0.partition_fragment_A(sQ);
        Tensor tSrK = wg_mma0.partition_fragment_B(sK);
        
        Tensor tOrV = wg_mma1.partition_fragment_B(sV);
        Tensor tOsP = wg_mma1.partition_fragment_A(sP);
        Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));

        /* Dequant K */
        TiledMma0_dequant tiled_mma0_dequant;
        TiledMma0_dequant_r2s tiled_mma0_dequant_r2s;
        auto mma0_dequant    = tiled_mma0_dequant.get_slice(thread_idx);
        // Tensor tSrK_f        = mma0_dequant.partition_fragment_B(sK);
        Tensor tSrK_dequant  = mma0_dequant.partition_fragment_B(sK);
        Tensor tSrK_pack_tmp = mma0_dequant.partition_fragment_B(sK_pack);
        Tensor tSrK_pack     = make_fragment_like<ElementKVPack>(tSrK_pack_tmp);
        
        // S2R
        auto smem_tiled_copy_K_pack = make_tiled_copy_B(S2RCopyAtomKPack{}, tiled_mma0_dequant);
        auto smem_thr_copy_K_pack   = smem_tiled_copy_K_pack.get_slice(thread_idx);
        Tensor tSsK                 = smem_thr_copy_K_pack.partition_S(sK);
        // Tensor tSrK_view            = smem_thr_copy_K_pack.retile_D(tSrK_f);
        Tensor tSsK_pack            = smem_thr_copy_K_pack.partition_S(sK_pack);
        Tensor tSrK_pack_view       = smem_thr_copy_K_pack.retile_D(tSrK_pack);

        // R2S
        auto smem_tiled_copy_K_dequant = make_tiled_copy_B(R2SCopyAtomKDequant{}, tiled_mma0_dequant_r2s);
        auto smem_thr_copy_K_dequant   = smem_tiled_copy_K_dequant.get_slice(thread_idx);
        // Tensor tSrK_r2s                = smem_thr_copy_K_dequant.retile_S(tSrK_f);
        Tensor tSsK_r2s                = smem_thr_copy_K_dequant.partition_D(sK);
        Tensor tSrK_dequant_r2s        = smem_thr_copy_K_dequant.retile_S(tSrK_dequant);

        /* Dequant V */
        Tensor tSrV_dequant  = mma0_dequant.partition_fragment_B(sV);
        Tensor tSrV_pack_tmp = mma0_dequant.partition_fragment_B(sV_pack);
        Tensor tSrV_pack     = make_fragment_like<ElementKVPack>(tSrV_pack_tmp);

        // S2R
        auto smem_tiled_copy_V_pack = make_tiled_copy_B(S2RCopyAtomVPack{}, tiled_mma0_dequant);
        auto smem_thr_copy_V_pack   = smem_tiled_copy_V_pack.get_slice(thread_idx);
        Tensor tSsV                 = smem_thr_copy_V_pack.partition_S(sV);
        Tensor tSsV_pack            = smem_thr_copy_V_pack.partition_S(sV_pack);
        Tensor tSrV_pack_view       = smem_thr_copy_V_pack.retile_D(tSrV_pack);

        // R2S
        auto smem_tiled_copy_V_dequant = make_tiled_copy_B(R2SCopyAtomVDequant{}, tiled_mma0_dequant_r2s);
        auto smem_thr_copy_V_dequant   = smem_tiled_copy_V_dequant.get_slice(thread_idx);
        // Tensor tSrV_r2s                = smem_thr_copy_V_dequant.retile_S(tSrV_pack);
        Tensor tSsV_r2s                = smem_thr_copy_V_dequant.partition_D(sV);
        Tensor tSrV_dequant_r2s        = smem_thr_copy_V_dequant.retile_S(tSrV_dequant);

        // KParams
        // cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);
        GmemTileCopyKParams gmem_tiled_copy_k_params;
        auto gmem_thr_copy_k_params = gmem_tiled_copy_k_params.get_thread_slice(thread_idx);
        Tensor tKgK_params          = gmem_thr_copy_k_params.partition_S(gK_params);
        Tensor tKsK_params          = gmem_thr_copy_k_params.partition_D(sK_params);
        cute::copy(gmem_tiled_copy_k_params, tKgK_params, tKsK_params);
        cute::cp_async_fence();
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);

        // VParams
        GmemTileCopyVParams gmem_tiled_copy_v_params;
        auto gmem_thr_copy_v_params = gmem_tiled_copy_v_params.get_thread_slice(thread_idx);
        Tensor tVgV_params          = gmem_thr_copy_v_params.partition_S(gV_params);
        Tensor tVsV_params          = gmem_thr_copy_v_params.partition_D(sV_params);

        using TensorParamsKC = decltype(make_tensor<half_t>(make_shape(Int<4 * num_params>{}, Int<tile_paramsk_m>{}, Int<tile_paramsk_k>{})));
        using TensorParamsVG = decltype(make_tensor<half_t>(make_shape(Int<num_bits * num_params>{}, Int<tile_paramsv_k>{})));
        using TensorParamsG  = decltype(make_tensor<half_t>(make_shape(Int<tile_paramsk_g>{})));
        TensorParamsKC tScales_k_c, tZeros_k_c;
        TensorParamsVG tScales_v_c, tZeros_v_c;
        TensorParamsG  tScales_k_g, tZeros_k_g;
        auto tScales_k_h2_c  = cute::recast<__half2>(tScales_k_c);
        auto tZeros_k_h2_c   = cute::recast<__half2>(tZeros_k_c);
        auto tScales_k_h2_g  = cute::recast<__half2>(tScales_k_g);
        auto tZeros_k_h2_g   = cute::recast<__half2>(tZeros_k_g);
        auto tScales_v_h2    = cute::recast<__half2>(tScales_v_c);
        auto tZeros_v_h2     = cute::recast<__half2>(tZeros_v_c);

        /******/

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        // Need to initialize tOrO in the case of RescaleOBeforeGemm where we will scale tOrO even in the 1st iter
        clear(tOrO);

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int n_block = n_block_max - 1;

        flash::Mask<kBlockM, kBlockN, PackGQA, TiledMma0> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, params.sink_token_length,
            params.qhead_per_khead_divmod
        );

        float softcap_val = params.softcap_val;
        if constexpr (Has_softcap && Is_FP8) {
            float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
            float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
            softcap_val *= q_descale * k_descale;
        }
        // Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
        // -inf to e.g. -50.0, which can affect the attention softmax.
        auto scoremod_premask_fn = [&](auto& tSrS) {
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        };

        auto &barrier_Q = shared_storage.pipelines.barrier_Q;
        barrier_Q.wait(work_idx % 2);
        
        // No intra-WG overlap
        warp_scheduler_barrier_sync();
        auto fwd_step = [&](int const n_block, int const n_block_min, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
            static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
            static constexpr bool Check_inf = decltype(check_inf_type)::value;
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));

            // cp_async_wait<0>();
            // cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);
            consumer_wait(pipeline_k, smem_pipe_read);
            
            /* Load Vparams */
            if (!is_first_iter_type) {
                tVgV_params.data() = tVgV_params.data() + (-int(kBlockN * v_params_row_stride));
            }
            cute::copy(gmem_tiled_copy_v_params, tVgV_params, tVsV_params);
            cute::cp_async_fence();
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);

            /* Dequant K */
            cute::copy(smem_tiled_copy_K_pack, tSsK_pack, tSrK_pack_view);
            quant::load_params_Ktensor(tScales_k_h2_g, tZeros_k_h2_g, sK_params, thread_idx, num_params);
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tSrK_pack); ++i) {
                quant::dequantize_Ktensor(tSrK_pack, tSrK_dequant, tScales_k_h2_g, tZeros_k_h2_g, 4, group_size, i);
            }
            cute::copy(smem_tiled_copy_K_dequant, tSrK_dequant_r2s, tSsK_r2s);

            #if DEBUG
            if (is_first_iter_type && threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                PRINT("sK_pack", sK_pack.layout()); PRINTTENSOR("sK_pack", sK_pack);
                PRINT("tSrK_pack", tSrK_pack.layout()); PRINTTENSOR("tSrK_pack", tSrK_pack);
                PRINT("tSrK", tSrK.layout()); // PRINTTENSOR("tSrK", tSrK);
                PRINTTENSOR("sK", sK)
            }
            #endif
            /******/

            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<0>();

            pipeline_k.consumer_release(smem_pipe_read);  // release K

            scoremod_premask_fn(tSrS);
            mask_fn(tSrS, n_block);
            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
            if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }

            // cp_async_wait<0>();
            // cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);
            consumer_wait(pipeline_v, smem_pipe_read);

            /* Load Kparams */
            if (n_block > n_block_min) {
                tKgK_params.data()   = tKgK_params.data() + (-int(kBlockN_params * k_params_row_stride));
                cute::copy(gmem_tiled_copy_k_params, tKgK_params, tKsK_params);
                cute::cp_async_fence();
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);
            }
            
            /* Dequant V */
            cute::copy(smem_tiled_copy_V_pack, tSsV_pack, tSrV_pack_view);

            CUTE_UNROLL
            for (int i = 0; i < size<2>(tSrV_pack); ++i) {
                quant::load_params_Vtensor<num_bits>(tScales_v_h2, tZeros_v_h2, sV_params, thread_idx, i, num_params);
                quant::dequant_Kchannel_Vtensor<num_bits>(tSrV_pack(_,_,i,_0{}), tSrV_dequant(_,_,i,_0{}), tScales_v_h2(_,i), tZeros_v_h2(_,i), num_params);
            }
            cute::copy(smem_tiled_copy_V_dequant, tSrV_dequant_r2s, tSsV_r2s);

            #if DEBUG
            if (is_first_iter_type && threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                PRINT("sV_pack", sV_pack.layout());
                PRINT("tSrV_pack", tSrV_pack.layout());
                PRINT("tOrV", tOrV.layout());
            }
            #endif
            /******/

            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            pipeline_v.consumer_release(smem_pipe_read);  // release V
            ++smem_pipe_read;
        };

        auto first_iter_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
        fwd_step(n_block, n_block_min, first_iter_mask_fn, cute::true_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
        --n_block;
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : params.qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        auto no_mask_fn = [](auto& tSrS, int n_block) { };
        #pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            fwd_step(n_block, n_block_min, no_mask_fn, cute::false_type{} /*is_first_iter*/, cute::false_type{} /*check_inf*/);
        }

        warp_scheduler_barrier_arrive();
        // Tell producers that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::KParamsEmpty) /*id*/);
        float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
        Tensor scores_scale = softmax.finalize(v_descale);
        softmax.rescale_o(tOrO, scores_scale);
        if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
        
        ++work_idx;
        return true;
    }
};

} // namespace flash
