import bitblas
import torch
import time
import numpy as np

# uncomment to enable debug output
# bitblas.set_log_level("Debug")

# Prefill
n_heads = 1
seq_len = 128
dim = 128
matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=n_heads*seq_len,  # N dimension
    K=dim,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="int4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)

matmul = bitblas.Matmul(config=matmul_config)

# Create input matrices
# input_tensor = torch.rand((1, dim), dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, 7, (n_heads*seq_len, dim), dtype=torch.int8).cuda()

# Warmup runs
print("\nWarming up...")
for _ in range(5):
    _ = matmul.transform_weight(weight_tensor)
    torch.cuda.synchronize()

# Timing runs
num_runs = 10
times = []

print(f"\nRunning {num_runs} timing iterations...")

for i in range(num_runs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    weight_tensor_int4 = matmul.transform_weight(weight_tensor)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    times.append(elapsed_time)
    
    if (i + 1) % 20 == 0:
        print(f"  Completed {i + 1}/{num_runs} runs")

times = np.array(times)
mean_time = np.mean(times)

print(f"Mean time: {mean_time} ms")