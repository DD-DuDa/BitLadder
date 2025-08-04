/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockN_, int kNWarps_, int quantmode_=0, int num_bits_=4, int group_size_=128, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, 32, kBlockN_, kNWarps_, elem_type> >
struct Flash_qpack_traits : public Base {
    using Element                = typename Base::Element;
    using ElementKVPack          = cute::uint16_t;
    using index_t                = typename Base::index_t;
    using SmemCopyAtom           = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr int quant_mode = quantmode_;
    static constexpr int group_size = group_size_;
    static constexpr int num_bits   = num_bits_;
    static constexpr int pack_num   = 16 / num_bits;

    // The number of threads.
    static constexpr int kNWarps   = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    
    static constexpr int kBlockN           = kBlockN_;
    static constexpr int kBlockN_pack      = num_bits   == 4 ? 128 : 256;
    static constexpr int kBlockP           = quant_mode == 1 ? kBlockN / pack_num : kBlockN;
    static constexpr int kBlockK_params    = quant_mode == 1 ? kBlockN / group_size : kBlockN;
    static constexpr int kHeadDim          = kHeadDim_;
    static constexpr int kHeadDim_pack     = kHeadDim / pack_num; // TODO
    static constexpr int kHeadDim_k        = quant_mode == 1 ? kHeadDim : kHeadDim_pack;
    static constexpr int kHeadDim_k_params = quant_mode == 1 ? kHeadDim : kHeadDim / group_size;
    static constexpr int kHeadDim_v_params = kHeadDim / group_size;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle    = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int tile_paramsk_g = kBlockN / 32 * (kBlockN / group_size); // TODO: check
    static constexpr int tile_paramsk_j = kBlockN / group_size;
    static constexpr int tile_paramsk_k = kHeadDim / 16;
    static constexpr int tile_paramsv_k = kBlockN / 16;    // TODO: check 128

    static constexpr int num_params = kBlockN_pack / group_size;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<1>,_4,_1>>,  
        Tile<Int<16>, _128, _16>>;

    using TiledMmaK_i4 = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<1>,_4,_1>>,  
        Tile<Int<16>, Int<32>, _16>>;

    using SmemLayoutAtomKV_SW = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomK_tiled = decltype(
        make_layout(make_shape(Int<8>{}, Int<kHeadDim>{}),
                    make_stride(Int<kHeadDim>{}, Int<1>{})));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK_tiled{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutKPack = decltype(
        make_layout(make_shape(Int<kBlockP>{}, Int<kHeadDim_k>{}),
                    make_stride(Int<kHeadDim_k>{}, Int<1>{})));
    using SmemLayoutKPacktransposed_ = decltype(
        composition(SmemLayoutKPack{}, make_layout(Shape<Int<kHeadDim_k>, Int<kBlockP>>{}, GenRowMajor{})));
    using SmemLayoutKPacktransposed = std::conditional_t<
        quant_mode == 1,
        SmemLayoutKPack,
        SmemLayoutKPacktransposed_
    >;

    using SmemLayoutAtomV = decltype(
        make_layout(make_shape(Int<8>{}, Int<kHeadDim_pack>{}),
                    make_stride(Int<kHeadDim_pack>{}, Int<1>{})));
    using SmemLayoutVPack = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDim_pack>>{}));
    using SmemLayoutVPacktransposed = decltype(
        composition(SmemLayoutVPack{}, make_layout(Shape<Int<kHeadDim_pack>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVPacktransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVPacktransposed{}));

    // TODO: 32 of x can be determined. 
    using SmemLayoutReduce_tmp = decltype(
        make_layout(make_shape(Int<32>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{})));

    using R2SCopyAtom = Copy_Atom<DefaultCopy, Element>;
    using R2SCopyAtomPack = Copy_Atom<DefaultCopy, ElementKVPack>;

    struct SharedStorage
    {
        array_aligned<Element, cosize_v<SmemLayoutKV>> smem_K;
        array_aligned<Element, cosize_v<SmemLayoutKV>> smem_V;
        array_aligned<ElementKVPack, cosize_v<SmemLayoutKPack>> smem_Kpack;
        array_aligned<ElementKVPack, cosize_v<SmemLayoutVPack>> smem_Vpack;
        array_aligned<Element, cosize_v<SmemLayoutReduce_tmp>> smem_reduce_tmp;
    };
    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTileCopyK_Pack = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementKVPack>{},
                        make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTileCopyV_Pack = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementKVPack>{},
                        make_layout(make_shape(_64{}, _2{}), make_stride(_2{}, _1{})),
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};

////////////////////////////////////////////////////////////////////////////////////////////////////