#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"

#include "include/flash.h"
#include "include/static_switch.h"
#include "include/tile_size.h"
#include "include/flash_fwd_kernel_sm90.h"
#include "include/mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "include/epilogue_fwd.hpp"
#include "include/tile_scheduler.hpp"
#include "include/kernel_traits.h"
#include "include/flash_qpack_kernel.h"

#include <cstdio>

using namespace cute;

template <int Arch, int kHeadDim, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKV, bool AppendKV,
          bool PackGQA, bool Split, bool V_colmajor>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
    static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");
    static constexpr bool FP8_TransposeV = false;

    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd_sm90(kHeadDim, Is_causal, Is_local, sizeof(Element) /*element_size*/, V_colmajor, PagedKV, Has_softcap);
    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, Is_causal, Is_local, sizeof(Element) /*element_size*/, PagedKV, Varlen && Split, Has_softcap, AppendKV);
    static constexpr int kBlockM = Arch >= 90 ? std::get<0>(kBlockMN_RS_IntraWGOverlap) : std::get<0>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kBlockN = Arch >= 90 ? std::get<1>(kBlockMN_RS_IntraWGOverlap) : std::get<1>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool Mma1_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    // static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = false;
    static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
    // static constexpr int kStages = Arch >= 90 ? 2 : std::get<3>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kStages = 1;
    static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape  = cute::Shape<Int<ClusterM>, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<kStages, ClusterShape, TileShape_MNK, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKV, AppendKV, Mma1_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK, ClusterShape, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, FP8_TransposeV>;

    static constexpr int NumProducerThreads = Arch >= 90 ? CollectiveMainloop::NumProducerThreads : CollectiveMainloop::NumMmaThreads;
    using SchedulerPersistent = flash::StaticPersistentTileScheduler<Split>;
    using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
    using Scheduler = SchedulerSingleTile;
    using AttnKernel = flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;
    using ElementKVPack = typename CollectiveMainloop::ElementKVPack;

    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    bool const is_varlen_k_new = params.cu_seqlens_knew;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
    int batch_k_pack = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},   // stride_Q

        // static_cast<Element*>(params.k_ptr),                                                           // K_ptr
        {!PagedKV ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
        params.d, params.h_k, !PagedKV ? batch_k : params.num_pages},                                 // shape_K
        // {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},   // stride_K
        static_cast<ElementKVPack*>(params.K_pack_ptr),                                                // K_pack_ptr
        {params.seqlen_k_pack, params.d_kpack, params.h_k, batch_k},                                   // shape_K_pack
        {params.K_pack_row_stride, _1{}, params.K_pack_head_stride, params.K_pack_batch_stride},       // stride_K_pack
        static_cast<__half2*>(params.k_params_ptr),                                                    // K_params_ptr
        {params.seqlen_k_params, params.d, params.h_k, batch_k},                                       // shape_K_params
        // {params.k_params_row_stride, _1{}, params.k_params_head_stride, params.k_params_batch_stride}, // stride_K_params
        {_1{}, params.k_params_dim_stride, params.k_params_head_stride, params.k_params_batch_stride},

        // static_cast<Element*>(params.v_ptr),                                                            // V_ptr        
        // v_strides,                                                                                      // stride_V
        static_cast<ElementKVPack*>(params.v_pack_ptr),                                                 // V_pack_ptr
        {params.seqlen_k, params.d_vpack, params.h_k, batch_k},                                           // shape_V_pack
        {params.v_pack_row_stride, _1{}, params.v_pack_head_stride, params.v_pack_batch_stride},        // stride_V_pack
        static_cast<__half2*>(params.v_params_ptr),                                                     // V_params_ptr
        {params.seqlen_k, params.d_vparams, params.h_k, batch_k},                                       // shape_V_params
        {_1{}, params.v_params_dim_stride, params.v_params_head_stride, params.v_params_batch_stride},  // stride_V_params

        static_cast<Element const*>(params.knew_ptr),
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},  // shape_K_new
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},  // stride_K_new
        static_cast<Element const*>(params.vnew_ptr),
        {params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? params.vnew_batch_stride : 0}, // stride_V_new
        static_cast<Element const*>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2},  // shape_rotary, the seqlen shape doesn't matter
        {params.rotary_dim / 2, _1{}},  // stride_rotary_cos
        static_cast<Element const*>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}},  // stride_rotary_sin
        params.is_rotary_interleaved,
        params.page_table,
        // if page_size is not set, avoid dividing by zero
        {params.kv_batch_idx ? params.b_k : params.b, !PagedKV ? 0 : params.seqlen_k / params.page_size}, // shape_page_table
        {params.page_table_batch_stride, _1{}},  // stride_page_table
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.window_size_left, params.window_size_right, params.sink_token_length,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k,
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(!Split ? params.o_ptr : params.oaccum_ptr),
        {seqlen_q, params.d, params.h, batch_q, params.num_splits},  // shape_O
        {!Split ? params.o_row_stride : params.oaccum_row_stride,
         _1{},
         !Split ? params.o_head_stride : params.oaccum_head_stride,
         !is_varlen_q ? (!Split ? params.o_batch_stride : params.oaccum_batch_stride) : 0,
         !Split ? 0 : params.oaccum_split_stride},  // stride_O
        static_cast<float*>(!Split ? params.softmax_lse_ptr : params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, !Split ? 0 : params.h * seqlen_q * batch_q},  // stride_LSE
        params.h_k,
        params.cu_seqlens_q, params.seqused_q
    };

    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q
    };

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });
    
    dim3 grid_dims  = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size   = AttnKernel::SharedStorageSize;

    #if DEBUG
        printf("Arch: %d kHeadDim: %d ClusterM: %d\n", Arch, kHeadDim, ClusterM);
        printf("kStages: %d\n", kStages);
        printf("grid_dims: %d %d %d\n", grid_dims.x, grid_dims.y, grid_dims.z);
        printf("block_dims: %d %d %d\n", block_dims.x, block_dims.y, block_dims.z);
    #endif

    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    } else {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    }
    CHECK_CUDA_KERNEL_LAUNCH();

}

template<int Arch, typename T, int kHeadDim, bool Split, bool PagedKV, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = false;
    using T_out = std::conditional_t<!Split, std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>, float>;

    // CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    //     VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor_, [&] {
    //         static constexpr bool V_colmajor = V_colmajor_ && sizeof(T) == 1;
    //         VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
    //             // Only needed here to decide if we should use cluster
    //             static constexpr int kBlockM = Arch >= 90 ? std::get<0>(tile_size_fwd_sm90(kHeadDim, Is_causal, Is_local, sizeof(T) /*element_size*/, V_colmajor, PagedKV, Has_softcap)) : 128;

    //             static constexpr bool Enable_cluster = Arch >= 90 && (sizeof(T) == 2 ? (kHeadDim >= 128) : (kHeadDim == 192)) && !Is_causal && !Is_local && !Split && !PagedKV && !Varlen;
    //             APPENDKV_SWITCH(params.knew_ptr, AppendKV, [&] {
    //                 // Only use Cluster if number of tiles along seqlen_q is even and not varlen
    //                 CLUSTER_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, Use_cluster, [&] {
    //                     static constexpr int ClusterM = Enable_cluster && Use_cluster ? 2 : 1;

    //                     printf("ClusterM: %d Is_causal: %d Is_local: %d Split: %d PagedKV: %d Varlen: %d AppendKV: %d PackGQA: %d V_colmajor: %d\n", ClusterM, Is_causal, Is_local, Split, PagedKV, Varlen, AppendKV, PackGQA, V_colmajor);

                        

                        static constexpr int ClusterM = 1;
                        static constexpr bool Is_causal = false;
                        static constexpr bool Is_local = false;
                        static constexpr bool Varlen = false;
                        static constexpr bool AppendKV = false;
                        static constexpr bool V_colmajor = false;

                        auto tile_size = tile_size_fwd_sm90(kHeadDim, Is_causal, Is_local, sizeof(T) /*element_size*/, V_colmajor, PagedKV, Has_softcap);
                        // printf("ClusterM: %d tile_size: %d %d %d %d\n", ClusterM, std::get<0>(tile_size), std::get<1>(tile_size), std::get<2>(tile_size), std::get<3>(tile_size));

                        run_flash_fwd<Arch, kHeadDim, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKV, AppendKV && Varlen, PackGQA, Split, V_colmajor>(params, stream);
    //                 });
    //             });
    //         });
    //     });
    // });

}


////////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

#define DEFINE_FLASH_QPACK_KERNEL(kernelName, ...) \
template<typename Kernel_traits> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_QPACK_KERNEL(flash_qpack_kernel) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_qpack<Kernel_traits>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits>
void run_flash_qpack(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    dim3 grid(num_n_block, params.b, params.h);
    
    auto kernel = &flash_qpack_kernel<Kernel_traits>;

    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();

}

template<typename T, int quant_mode, int num_bits, int group_size>
void run_kvcache_qpack_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockN = num_bits == 4 ? 128 : 256;

    run_flash_qpack<Flash_qpack_traits<Headdim, kBlockN, 4, quant_mode, num_bits, group_size, T>>(params, stream);
}