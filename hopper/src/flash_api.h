#pragma once

#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "include/flash.h"
#include "include/heuristics.h"
#include "include/tile_size.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k, const size_t seqlen_k_pack, const size_t seqlen_k_params,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d, const size_t d_kpack, const size_t d_vpack,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k, const at::Tensor k_pack, const at::Tensor k_params,
                      const at::Tensor v, const at::Tensor v_pack, const at::Tensor v_params,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.f,
                      const int sm_margin=0) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

    // Set the pointers and strides.
    params.q_ptr        = q.data_ptr();
    params.k_ptr        = k.data_ptr();
    params.K_pack_ptr   = k_pack.data_ptr();
    params.k_params_ptr = k_params.data_ptr();
    params.v_ptr        = v.data_ptr();
    params.v_pack_ptr   = v_pack.data_ptr();
    params.v_params_ptr = v_params.data_ptr();

    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.K_pack_row_stride = k_pack.stride(-3);
    params.k_params_row_stride = k_params.stride(-1);
    params.v_row_stride = v.stride(-3);
    params.v_pack_row_stride = v_pack.stride(-3);
    params.v_params_row_stride = v_params.stride(-1);

    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.K_pack_head_stride = k_pack.stride(-2);
    params.k_params_head_stride = k_params.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.v_pack_head_stride = v_pack.stride(-2);
    params.v_params_head_stride = v_params.stride(-2);

    params.v_dim_stride = v.stride(-1);
    params.k_params_dim_stride = k_params.stride(-3);
    params.v_params_dim_stride = v_params.stride(-3);

    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.o_batch_stride = out.stride(0);
    }
    if (cu_seqlens_k_d == nullptr) {
        params.k_batch_stride = k.stride(0);
        params.K_pack_batch_stride = k_pack.stride(0);
        params.k_params_batch_stride = k_params.stride(0);
        params.v_batch_stride = v.stride(0);
        params.v_pack_batch_stride = v_pack.stride(0);
        params.v_params_batch_stride = v_params.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_k_pack = seqlen_k_pack;
    params.seqlen_k_params = seqlen_k_params;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_kpack = d_kpack;
    params.d_vpack = d_vpack;
    params.d_vparams = v_params.size(1);
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    return num_splits_heuristic(params.b * (!params.pack_gqa ? params.h : params.h_k) * num_m_blocks, params.num_sm, num_n_blocks, 128);
    // return num_splits_heuristic(params.b * params.h_k * num_m_blocks, params.b * params.h_k,
    //                             params.num_sm, num_n_blocks, 128, params.d_rounded);
    #endif
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKV or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || params.page_table || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}


void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    return run_mha_fwd_<90, cutlass::half_t, 128, true, false, false, true>(params, stream);
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream) {
    return run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream);
}

template <int num_bits>
void run_kvcache_qpack(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.quant_mode == "k-channel") {
        if (params.group_size == 32) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 1, num_bits, 32>(params, stream);
        } else if (params.group_size == 64) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 1, num_bits, 64>(params, stream);
        } else if (params.group_size == 128) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 1, num_bits, 128>(params, stream);
        }
    } else {
        if (params.group_size == 32) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 0, num_bits, 32>(params, stream);
        } else if (params.group_size == 64) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 0, num_bits, 64>(params, stream);
        } else if (params.group_size == 128) {
            run_kvcache_qpack_<cutlass::half_t, 128, 0, num_bits, 128>(params, stream);
        }
    }
}

template<int num_bits>
at::Tensor
mha_fwd_kvcache(at::Tensor &q, 
                const at::Tensor &k, const at::Tensor &k_pack, const at::Tensor &k_params,
                const at::Tensor &v, const at::Tensor &v_pack, const at::Tensor &v_params,
                const float softmax_scale,
                const int num_splits=0) {
    
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major >= 8;
    TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    if (dprops->major < 9) {
        TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                    "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type");
    }
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

    CHECK_DEVICE(q); // CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor page_table;
    const bool paged_KV = false;
    at::Tensor cu_seqlens_q;
    bool const is_varlen_q = false;
    at::Tensor cu_seqlens_k;
    bool const is_varlen_k = false;
    bool const is_varlen = false;

    auto const sizes = q.sizes();
    const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;
    int seqlen_q = sizes[1];
    int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
    int num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const head_size_kpack = k_pack.size(-1);
    int const head_size_vpack = v_pack.size(-1);
    int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
    int const num_pages = !paged_KV ? 0 : k.size(0);
    int const page_size = !paged_KV ? 1 : k.size(1);
    int const seqlen_k  = !paged_KV ? k.size(1) : max_num_pages_per_seq * page_size;
    int const seqlen_k_pack = k_pack.size(1);
    int const seqlen_k_params = k_params.size(-1);
    int const total_k   = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
    int const num_heads_k  = k.size(-2);
    int const batch_size_k = k.size(0);

    int window_size_left = -1;
    int window_size_right = -1;

    bool is_causal = window_size_left < 0 && window_size_right == 0;

    int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;

    auto opts = q.options();
    at::Tensor out;
    out = torch::empty_like(q);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded = round_up_headdim(head_size);
    int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    at::Tensor softmax_lse;
    if (!is_varlen_q) {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    } else {
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k, seqlen_k_pack, seqlen_k_params,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_kpack, head_size_vpack,
                     head_size_rounded,
                     q, 
                     k, k_pack, k_params,
                     v, v_pack, v_params,
                     out,
                     !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                     !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right);
    params.total_q = total_q;
    params.total_k = total_k;
    params.sink_token_length = 0;
    params.b_k = batch_size_k;
    
    params.page_size = page_size;
    params.num_pages = num_pages;

    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    params.pack_gqa = get_pack_gqa(params);
    if (params.num_splits == 1) {
        params.num_splits = 2;
    }
    // printf("num_splits: %d\n", params.num_splits);
    // printf("pack_gqa: %d\n", params.pack_gqa);

    params.rotary_dim = 0;
    at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = at::ScalarType::Float;
    if (params.num_splits > 1) {
        TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        if (!is_varlen_q) {
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            params.oaccum_batch_stride = out_accum.stride(1);
            params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
        } else {
            out_accum = torch::empty({params.num_splits, num_heads, total_q, head_size}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
        }
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_row_stride = out_accum.stride(-2);
        params.oaccum_head_stride = out_accum.stride(-3);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_head_stride = softmax_lse_accum.stride(-2);
    }

    at::Tensor tile_count_semaphore;
    // We don't use the persistent scheduler if Split and not Varlen
    bool const persistent_scheduler = params.arch >= 90
        ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
        : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
    if (persistent_scheduler) {
        tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    } else {
        params.tile_count_semaphore = nullptr;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
    if (params.num_splits > 1) {
        if (is_varlen_q) {
            params.b = 1;
            params.seqlen_q = total_q;
        }
        run_mha_fwd_combine(params, stream);
    }

    return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// QPacking
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_params_fprop_qpack(Flash_fwd_params &params,
                            // sizes
                            const size_t b,
                            const size_t seqlen_k,
                            const size_t h, const size_t h_k,
                            const size_t d,
                            // device pointers
                            const at::Tensor k, at::Tensor k_pack, at::Tensor k_params,
                            const at::Tensor v, at::Tensor v_pack, at::Tensor v_params,
                            void *cu_seqlens_k_d,
                            const std::string quant_mode,
                            const int group_size
                            ) {

    // Reset the parameters
    params = {};

    params.is_bf16 = k.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.k_ptr = k.data_ptr();
    params.K_pack_ptr = k_pack.data_ptr();
    params.k_params_ptr = k_params.data_ptr();
    params.v_ptr = v.data_ptr();
    params.v_pack_ptr = v_pack.data_ptr();
    params.v_params_ptr = v_params.data_ptr();
    // All stride are in elements, not bytes.
    params.k_row_stride = k.stride(-3);
    params.K_pack_row_stride = k_pack.stride(-3);
    params.k_params_row_stride = k_params.stride(-1);
    params.v_row_stride = v.stride(-3);
    params.v_pack_row_stride = v_pack.stride(-3);
    params.v_params_row_stride = v_params.stride(-1);

    params.k_params_dim_stride = k_params.stride(-3);
    params.v_params_dim_stride = v_params.stride(-3);

    params.k_head_stride = k.stride(-2);
    params.K_pack_head_stride = k_pack.stride(-2);
    params.k_params_head_stride = k_params.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.v_pack_head_stride = v_pack.stride(-2);
    params.v_params_head_stride = v_params.stride(-2);

    // params.k_batch_stride = k.stride(0);
    params.k_batch_stride = seqlen_k * k.size(-2) * k.size(-1);
    params.K_pack_batch_stride = k_pack.stride(0);
    params.k_params_batch_stride = k_params.stride(0);
    // params.v_batch_stride = v.stride(0);
    params.v_batch_stride = seqlen_k * v.size(-2) * v.size(-1);
    params.v_pack_batch_stride = v_pack.stride(0);
    params.v_params_batch_stride = v_params.stride(0);

    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_k = seqlen_k;
    params.d = d;

    params.quant_mode = quant_mode;
    params.group_size = group_size;
}

template <int num_bits>
void kvcache_qpack(const at::Tensor &k,  
                   at::Tensor &k_pack,
                   at::Tensor &k_params,
                   const at::Tensor &v,  
                   at::Tensor &v_pack,
                   at::Tensor &v_params,
                   c10::optional<at::Tensor> &block_table_,
                   const at::Tensor &cu_seqlens_k, 
                   const int max_seqlen_k,
                   const std::string quant_mode,
                   const int group_size
                   ) {

    auto k_dtype = k.dtype();
    TORCH_CHECK(k_dtype == torch::kFloat16 || k_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_k);
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }
    
    const auto sizes = k.sizes();

    const int batch_size  = cu_seqlens_k.numel() - 1;
    int num_heads         = paged_KV ? sizes[2] : sizes[1];
    const int head_size   = paged_KV ? sizes[3] : sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks             = !paged_KV ? 0 : k.size(0);
    const int page_block_size        = !paged_KV ? 1 : k.size(1);
    const int page_block_size_pack   = !paged_KV ? 0 : k_pack.size(1);
    const int seqlen_k               = !paged_KV ? k.size(1) : max_num_blocks_per_seq * page_block_size;
    const int batch_size_c           = !paged_KV ? k.size(0) : batch_size;

    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)k.get_device()};

    Flash_fwd_params params;
    set_params_fprop_qpack(params,
                           batch_size,
                           max_seqlen_k,
                           num_heads, num_heads_k,
                           head_size,
                           k, k_pack, k_params,
                           v, v_pack, v_params,
                           /*cu_seqlens_k_d=*/nullptr,
                           quant_mode,
                           group_size
                           );


    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_kvcache_qpack<num_bits>(params, stream);
    } 

    return;
}