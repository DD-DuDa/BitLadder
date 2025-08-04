#include <cstdio>
#include <fstream>

#include "flash_api.h"

torch::Tensor single_mha(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, int head_dim) {
    const float sm_scale = 1.f / std::sqrt(float(head_dim));
    auto scaled_q = q * sm_scale;
    
    auto scores = torch::einsum("bthd,bshd->bhts", {scaled_q, k});
    auto attention = torch::softmax(scores, -1).to(v.dtype());
    auto output = torch::einsum("bhts,bshd->bthd", {attention, v});
    return output;
}

template <int num_heads, int num_heads_kv, int head_dim, int num_bits>
double TestDecodingKernelPerformance(int seqlen_kv, int bs, const std::string quant_mode, const int group_size, const int repeat, const int num_splits=0) {
    const int seqlen_q = 4;
    const int pack_nums = 16 / num_bits;

    torch::Tensor Q_host = torch::rand({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::ones({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::ones({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    // torch::Tensor K_device = K_host.to(torch::kCUDA);
    // torch::Tensor V_device = V_host.to(torch::kCUDA);
    
    at::Tensor k_pack, k_params, v_pack, v_params;
    if (quant_mode == "k-channel") {
        k_pack   = torch::empty({bs, seqlen_kv / pack_nums,   num_heads_kv, head_dim}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params = torch::empty({bs, seqlen_kv / group_size, num_heads_kv, head_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    } else {
        k_pack   = torch::empty({bs, seqlen_kv, num_heads_kv, head_dim / pack_nums}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    }
    v_pack   = torch::empty({bs, seqlen_kv, num_heads_kv, head_dim / pack_nums}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
    v_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    // Convert K, V to unpadded format
    // torch::Tensor K_unpad = K_device.reshape({bs * seqlen_kv, num_heads_kv, head_dim});
    // torch::Tensor V_unpad = V_device.reshape({bs * seqlen_kv, num_heads_kv, head_dim});

    // auto cu_seqlens_k = torch::arange(0, (bs + 1) * seqlen_kv, seqlen_kv, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    // std::optional<at::Tensor> opt_block_table = std::nullopt;

    // kvcache_qpack<num_bits>(
    //     K_unpad, k_pack, k_params,
    //     V_unpad, v_pack, v_params,
    //     opt_block_table,
    //     cu_seqlens_k,              
    //     seqlen_kv,
    //     quant_mode,
    //     group_size
    // );

    const float sm_scale = 1 / std::sqrt(float(head_dim));

    // Warm up
    for (int i = 1; i < 5; ++i)
        mha_fwd_kvcache<num_bits>(Q_device, 
                                  K_host, k_pack, k_params,
                                  V_host, v_pack, v_params,
                                  sm_scale);

    // Benchmark
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        // sm_scale = 1 / std::sqrt(float(head_dim));
        mha_fwd_kvcache<num_bits>(Q_device, 
                                  K_host, k_pack, k_params,
                                  V_host, v_pack, v_params,
                                  sm_scale,
                                  num_splits);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    msec = msec / repeat;

    return msec;
}

int main() {
    const int num_heads = 128;
    const int num_heads_kv = 32;
    const int head_dim = 128;
    const int num_bits = 4;
    const std::string quant_mode = "k-tensor";
    const int group_size = 128;
    const int test_num = 10;

    int len_list[test_num];
    len_list[0] = 1024;
    for (int i = 1; i < test_num; i++) {
        len_list[i] = len_list[i - 1] * 2;
    }

    int bs_list[7];
    bs_list[0] = 2;
    for (int i = 1; i < 7; i++) {
        bs_list[i] = bs_list[i - 1] * 2;
    }

    const int outer_repeat = 1, inner_repeat = 1;

    // printf("\n######## Benchmark single decode ########\n");
    // for (int j = 0; j < test_num; j++) {

    //     int seqlen_kv = len_list[j];
    //     double max_msec = 0.0;
    //     double min_msec = DBL_MAX;
    //     double total_msec = 0.0;

    //     for (int k = 0; k < outer_repeat; k++) {
    //         double this_sec = TestDecodingKernelPerformance<num_heads, num_heads_kv, head_dim, num_bits>(seqlen_kv, quant_mode, group_size, inner_repeat);
    //         max_msec = max(max_msec, this_sec);
    //         min_msec = min(min_msec, this_sec);
    //         total_msec += this_sec;
    //     }

    //     double avg_msec = total_msec / outer_repeat;
    //     printf("seqlen_kv num_heads head_dim = %6d %6d %6d, ", seqlen_kv, num_heads, head_dim);
    //     printf("Time = %12.8lf %12.8lf %12.8lf ms, \n", min_msec, avg_msec, max_msec);
    // }

    printf("\n######## Benchmark single decode with different num_splits ########\n");
    for (int j = 0; j < test_num; j++) {
        int bs = 1;
        int seqlen_kv = len_list[j];
        double best_time = DBL_MAX;
        int best_splits = 2;
        
        printf("\nTesting seqlen_kv=%d:\n", seqlen_kv);
        printf("num_splits  min_time(ms)  avg_time(ms)  max_time(ms)\n");
        printf("------------------------------------------------\n");
        
        // Test different num_splits values
        for (int splits = 0; splits <= 20; splits++) {
            double max_msec = 0.0;
            double min_msec = DBL_MAX;
            double total_msec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = TestDecodingKernelPerformance<num_heads, num_heads_kv, head_dim, num_bits>(
                    seqlen_kv, bs, quant_mode, group_size, inner_repeat, splits);
                max_msec = max(max_msec, this_sec);
                min_msec = min(min_msec, this_sec);
                total_msec += this_sec;
            }

            double avg_msec = total_msec / outer_repeat;
            printf("%9d  %11.4f  %11.4f  %11.4f\n", splits, min_msec, avg_msec, max_msec);
            
            if (min_msec < best_time) {
                best_time = min_msec;
                best_splits = splits;
            }
            if (j < 2) {
                break;
            } else if (j < 5 && splits > 5) {
                break;
            }
        }
        
        printf("\nBest result for seqlen_kv=%d: num_splits=%d, time=%.4f ms\n", 
               seqlen_kv, best_splits, best_time);
    }

    // printf("\n######## Benchmark single decode with different num_splits ########\n");
    // for (int j = 0; j < 7; j++) {
    //     int bs = bs_list[j];
    //     int seqlen_kv = 32768;
    //     double best_time = DBL_MAX;
    //     int best_splits = 0;
        
    //     printf("\nTesting batch_size=%d:\n", bs);
    //     printf("num_splits  min_time(ms)  avg_time(ms)  max_time(ms)\n");
    //     printf("------------------------------------------------\n");
        
    //     // Test different num_splits values
    //     for (int splits = 0; splits <= 10; splits++) {
    //         double max_msec = 0.0;
    //         double min_msec = DBL_MAX;
    //         double total_msec = 0.0;

    //         for (int k = 0; k < outer_repeat; k++) {
    //             double this_sec = TestDecodingKernelPerformance<num_heads, num_heads_kv, head_dim, num_bits>(
    //                 seqlen_kv, bs, quant_mode, group_size, inner_repeat, splits);
    //             max_msec = max(max_msec, this_sec);
    //             min_msec = min(min_msec, this_sec);
    //             total_msec += this_sec;
    //         }

    //         double avg_msec = total_msec / outer_repeat;
    //         printf("%9d  %11.4f  %11.4f  %11.4f\n", splits, min_msec, avg_msec, max_msec);
            
    //         if (min_msec < best_time) {
    //             best_time = min_msec;
    //             best_splits = splits;
    //         }

    //         if (j < 2) {
    //             break;
    //         } else if (j < 5 && splits > 5) {
    //             break;
    //         }
    //     }
        
    //     printf("\nBest result for seqlen_kv=%d: num_splits=%d, time=%.4f ms\n", 
    //            seqlen_kv, best_splits, best_time);
    // }

    return 0;
}