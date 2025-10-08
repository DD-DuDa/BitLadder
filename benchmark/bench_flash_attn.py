import torch
import math
import triton
from flash_attn import flash_attn_with_kvcache


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(10, 20, 1)],
        line_arg='provider',
        line_vals=['flash-attn-v2'],
        line_names=['Flash Attention v2'],
        styles=[('blue', '-')],
        ylabel="Time (ms)",
        plot_name="Flash Attention KV Cache Benchmark",
        args={},
    )
)
def benchmark(seq_len, provider):
    torch.random.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    batch_size = 1
    nheads = 32
    nheads_k = 32
    d = 128

    q = torch.randn(batch_size, 1, nheads, d, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seq_len, nheads_k, d, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seq_len, nheads_k, d, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flash-attn-v2':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_with_kvcache(q, k_cache, v_cache),
            quantiles=quantiles
        )
    
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)