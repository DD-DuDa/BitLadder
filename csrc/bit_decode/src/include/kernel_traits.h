/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

#include "blockscaled_layout.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#define DEBUG 0
#define DEBUG1 0
#define DEBUG2 0

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

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, int quant_mode_=0, int num_bits_=4, int group_size_=128, typename elem_type=cutlass::half_t,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    static constexpr bool Has_cp_async = Base::Has_cp_async;

    // TODO
    using Element = cutlass::float_e2m1_t;
    using ElementSF = cutlass::float_ue4m3_t;
    using ElementKVPack = cute::uint16_t;
    using ElementFP16 = cute::half_t;
    using ElementAccum = float;
    using ElementOut = float;
    using index_t      = typename Base::index_t;
    
    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomK = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomV = Copy_Atom<SM75_U32x2_LDSM_N, Element>;


    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs   = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int quant_mode = quant_mode_;
    static constexpr int group_size = group_size_;
    static constexpr int full_len   = 32;
    static constexpr int num_bits   = num_bits_;
    static constexpr int pack_num   = 16 / num_bits;

    static constexpr int residual_block_size = num_bits == 4 ? 128 : 256;

    // The number of threads.
    static constexpr int kNWarps    = kNWarps_;
    static constexpr int kNThreads  = kNWarps * 32;
    
    static constexpr int kBlockM            = kBlockM_;
    static constexpr int kBlockN            = kBlockN_;
    static constexpr int kBlockN_pack       = num_bits   == 4 ? 128 : 256;
    static constexpr int kBlockN_residual   = kBlockN_pack;
    static constexpr int kBlockP            = quant_mode == 1 ? kBlockN / pack_num : kBlockN;
    static constexpr int kBlockP_new_pack   = quant_mode == 1 ? kBlockN_pack / pack_num : kBlockN_pack;
    static constexpr int kBlockK_params     = quant_mode == 1 ? kBlockN / group_size : kBlockN;
    static constexpr int kBlockK_params_new = quant_mode == 1 ? kBlockN_pack / group_size : kBlockN_pack;
    static constexpr int kHeadDim           = kHeadDim_;
    static constexpr int kHeadDim_pack      = kHeadDim / pack_num; 
    static constexpr int kHeadDim_k         = quant_mode == 1 ? kHeadDim : kHeadDim_pack;
    static constexpr int kHeadDim_k_params  = quant_mode == 1 ? kHeadDim : kHeadDim / group_size;
    static constexpr int kHeadDim_v_params  = kHeadDim / group_size;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int k_pack_div         = quant_mode == 1 ? pack_num : 1;
    // static constexpr int k_params_div       = quant_mode == 1 ? group_size : 1;

    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle    = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int tile_paramsk_g   = kBlockN / 32 * (kBlockN / group_size); // TODO: check
    static constexpr int tile_paramsk_g_r = kBlockN_residual / 32 * (kBlockN_residual / group_size); // TODO: check
    static constexpr int tile_paramsk_j   = kBlockN / group_size;
    static constexpr int tile_paramsk_m   = kBlockN / kBlockN_pack;
    static constexpr int tile_paramsk_k   = kHeadDim / 16;
    static constexpr int tile_paramsv_k   = kBlockN / 16;    
    static constexpr int tile_paramsv_k_r = kBlockN_residual / 16;

    static constexpr int num_params = kBlockN_pack / group_size; // TODO: check 128

    static constexpr auto SFVectorSize = 16;
    //
    // Tiled MMA
    //

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<1>,_4,_1>>,  
        Tile<Int<16>, Int<128>, _16>>;

    using TiledMmaKV_i4 = TiledMMA<
        typename Base::MMA_Atom_Arch, // MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>
        Layout<Shape<Int<1>,_4,_1>>,  
        Tile<Int<16>, Int<32>, _16>>;

    using AtomLayoutMNK = Layout<Shape<Int<1>,_4,_1>>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::rr_blockscaled_op_selector_sm120<Element,
                                               Element,
                                               ElementAccum,
                                               ElementSF,
                                               SFVectorSize,
                                               false
                                               >(),
        AtomLayoutMNK{},
        Tile<Int<16>, Int<32>, _64>{}
    ));

    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::rr_blockscaled_op_selector_sm120<Element,
                                               Element,
                                               ElementAccum,
                                               ElementSF,
                                               SFVectorSize,
                                               false
                                               >(),
        AtomLayoutMNK{},
        Tile<Int<16>, Int<32>, _64>{}
    ));
    //
    // SF GMem Layout
    //
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;

    using LayoutP = decltype(
        make_layout(
            make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
            make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
        )
    );
    using LayoutSFP = decltype(
        make_layout(
            make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
            make_stride(make_stride(_0{}, _1{}), _0{}, _4{})
        )
    );

    //
    // Shared memory layout
    //

    // Q
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
    ));

    // K
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))));

    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutSFK = decltype(make_layout(
        shape(SmemLayoutAtomSFK{}),
        stride(SmemLayoutAtomSFK{})
    ));


    // V
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))));

    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutSFV = decltype(make_layout(
        shape(SmemLayoutAtomSFV{}),
        stride(SmemLayoutAtomSFV{})
    ));

    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<1>(TileShape_MNK{}))>());
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}))));
    
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));
    using SmemLayoutSFVt = decltype(make_layout(
        shape(SmemLayoutAtomSFVt{}),
        stride(SmemLayoutAtomSFVt{})
    ));

    // acc
    using SmemLayoutAtomACC = decltype(composition(
        Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                        make_stride(Int<kBlockN>{}, Int<1>{}))));
    using SmemLayoutAcc = decltype(tile_to_shape(
        SmemLayoutAtomACC{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemLayoutAtomACC_residual = decltype(composition(
        Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN_residual>{}),
                                        make_stride(Int<kBlockN_residual>{}, Int<1>{}))));
    using SmemLayoutAcc_residual = decltype(
        make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN_residual>{}),
                    make_stride(Int<kBlockN_residual>{}, Int<1>{})));
    using R2SCopyAtomAcc = Copy_Atom<UniversalCopy<int>, ElementFP16>;

    // O
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    // Reduce tmp
    // TODO: check 32
    using SmemLayoutReduce_tmp = decltype(
        composition(Swizzle<1, 3, 3>{},
                    Layout<Shape<_8, Int<32>>,
                        Stride<Int<32>, _1>>{}));

    struct SharedStorage
    {
        array_aligned<Element, cosize_v<SmemLayoutQ>> smem_Q;
        array_aligned<ElementSF, cosize_v<SmemLayoutSFQ>> smem_SFQ;
        array_aligned<Element, cosize_v<SmemLayoutK>> smem_K;
        array_aligned<ElementSF, cosize_v<SmemLayoutSFK>> smem_SFK;
        array_aligned<Element, cosize_v<SmemLayoutV>> smem_V;
        array_aligned<ElementSF, cosize_v<SmemLayoutSFV>> smem_SFV;
        array_aligned<ElementFP16, cosize_v<SmemLayoutAcc>> smem_acc;

    };
    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    //
    // Copy Atom, global to shared memory
    //

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(cutlass::half_t);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>; // (16, 8)

    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        Layout<Shape <Int<32>, Int<4>>,
                        Stride<Int<32>, _1>>{},
                        Layout<Shape<_1, _32>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopySF = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementSF>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _1>>{}));  // Val layout, 8 vals per read


//  using GmemTileCopyKV_Residual = decltype(
//      make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
//                      GmemLayoutAtom{},
//                      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

//  // TODO: check 32, 4
//  // KV
//  using GmemTileCopyK_Pack = decltype(
//      make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVPack>{},
//                      make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
//                      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
//  using GmemTileCopyV_Pack_2bit = decltype(
//      make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVPack>{},
//                      make_layout(make_shape(_64{}, _2{}), make_stride(_2{}, _1{})),
//                      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
//  using GmemTileCopyV_Pack_4bit = decltype(
//      make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVPack>{},
//                      make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
//                      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
//  using GmemTileCopyV_Pack = std::conditional_t<
//      num_bits == 2,
//      GmemTileCopyV_Pack_2bit,
//      GmemTileCopyV_Pack_4bit
//  >;
//  using GmemTileCopyKV_NewPack = decltype(
//      make_tiled_copy(Copy_Atom<DefaultCopy, ElementKVPack>{},
//                      make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
//                      Layout<Shape<_1, _4>>{}));  // Val layout, 8 vals per read

//  // KV params
//  using GmemTileCopyKParams_BN128 = decltype(
//      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, __half2>{},
//                      make_layout(make_shape(_1{}, _128{}), make_stride(_1{}, _1{})),
//                      Layout<Shape<_1, _1>>{}));  // Val layout, 4 vals per store
//  using GmemTileCopyKParams_BN256 = decltype(
//      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint64_t>, __half2>{},
//                      make_layout(make_shape(_1{}, _128{}), make_stride(_1{}, _1{})),
//                      Layout<Shape<_2, _1>>{}));  // Val layout, 4 vals per store
//  using GmemTileCopyKParams_channel = std::conditional_t<
//      kBlockN == 256,
//      GmemTileCopyKParams_BN128,  // TODO: check
//      GmemTileCopyKParams_BN256
//  >;



//  using GmemTileCopyVParams_BN128 = decltype(
//      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, __half2>{},
//                      make_layout(make_shape(_128{}, _1{}), make_stride(_1{}, _1{})),
//                      Layout<Shape<_1, _1>>{}));  // Val layout, 4 vals per store
//  using GmemTileCopyVParams_BN256 = decltype(
//      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint64_t>, __half2>{},
//                      make_layout(make_shape(_128{}, _1{}), make_stride(_1{}, _1{})),
//                      Layout<Shape<_2, _1>>{}));  // Val layout, 4 vals per store
//  using GmemTileCopyVParams = std::conditional_t<
//      kBlockN == 256,
//      GmemTileCopyVParams_BN256,
//      GmemTileCopyVParams_BN128
//  >;

//  using GmemTileCopyKParams = std::conditional_t<
//      quant_mode == 1,
//      GmemTileCopyKParams_channel,
//      GmemTileCopyVParams
//  >;

//  // using GmemTileCopyKParams = GmemTileCopyVParams_BN128;


    // O
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementOut>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
// //  using GmemLayoutAtomRotcossin = GmemLayoutAtom;
// //  using GmemTiledCopyRotcossin = decltype(
// //      make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
// //                      GmemLayoutAtomRotcossin{},
// //                      Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
// //  using GmemTiledCopyRotcossinCont = decltype(
// //      make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
// //                      GmemLayoutAtomRotcossin{},
// //                      Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load

//  //
//  // Copy Atom, shared to register
//  //

//  using S2RCopyOpQ       = SM75_U32x2_LDSM_N;
//  using S2RCopyTraitsQ   = Copy_Traits<S2RCopyOpQ>;

//  using S2RCopyAtomK     = Copy_Atom<S2RCopyTraitsQ , Element>;
//  using S2RCopyAtomK_i4  = Copy_Atom<S2RCopyTraitsQ , ElementKVPack>;

//  using S2RCopyAtomV     = SmemCopyAtomTransposed;
//  using S2RCopyAtomV_i4  = Copy_Atom<SM75_U16x4_LDSM_T, ElementKVPack>;

//  using SmemCopyAtomTransposed_i4 = Copy_Atom<SM75_U16x4_LDSM_T, ElementKVPack>;
//  // using SmemCopyAtomTransposed_i4 = Copy_Atom<DefaultCopy, ElementKVPack>;
//  using SmemCopyAtomTransposed_residual = SmemCopyAtomTransposed_i4;

//  using R2SCopyAtomPack = Copy_Atom<DefaultCopy, ElementKVPack>;

//  // using R2SCopyAtomPack = Copy_Atom<DefaultCopy, cute::uint8_t>;
//  // using ElementKVPack = cute::uint16_t;

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