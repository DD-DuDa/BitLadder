import argparse
import dataclasses
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from llama import LlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer

@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
#   device: str = dataclasses.field(default="cuda:0")


def load_model(args):
    # device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    config = LlamaConfig.from_pretrained(args.model_path)
    config.attn_backend = args.attn_backend
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.residual_block_size = 128 if args.num_bits == 4 else 256

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        device_map="auto",
        torch_dtype=dtype
    )
    return model

@torch.inference_mode()
def benchmark_throughput():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="llama3-8b-instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_len", type=int, default=2*1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--attn_backend", type=str, default="flash_attention_2")
    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--quant_mode", type=str, default="k-channel")
    parser.add_argument("--group_size", type=int, default=128)
    
    args = parser.parse_args()

    model = load_model(args)

    context_len = args.context_len
    decode_len = args.decode_len
    batch_size = args.batch_size
    
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    hidden_size = model.config.hidden_size

    prefill_latency = []
    decode_latency = []

    for iter_idx in tqdm(range(args.iteration)):
        # clear cuda cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Prefill Stage
        ts = time.perf_counter()
        hidden_states = torch.randn(batch_size, context_len, hidden_size, dtype=dtype, device=device)
        out = model(
            inputs_embeds=hidden_states,
            use_cache=True
        )
        torch.cuda.synchronize()
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        
        # Memory stats after prefill
        if iter_idx == 0:
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB")

        # Warm up for decode
        for _ in range(5):
            hidden_states = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device)
            model(
                inputs_embeds=hidden_states,
                past_key_values=out.past_key_values,
                use_cache=True,
            )

        # Decode Stage - measure total time for all tokens
        ts_decode_total = time.perf_counter()
        for _ in range(decode_len):
            hidden_states = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device)
            out = model(
                inputs_embeds=hidden_states,
                past_key_values=out.past_key_values,
                use_cache=True,
            )
        torch.cuda.synchronize()
        te_decode_total = time.perf_counter()
        decode_latency.append(te_decode_total - ts_decode_total)
    
    # Calculate metrics
    avg_prefill_latency = np.mean(prefill_latency)
    avg_decode_latency = np.mean(decode_latency)
    # avg_decode_latency -= 0.0019366741180 * 32
    
    # Calculate throughput
    prefill_throughput = (batch_size * context_len) / avg_prefill_latency
    decode_throughput = (batch_size * decode_len) / avg_decode_latency
    
    # Print results in a table format
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Model: {args.model_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Context Length: {context_len}")
    print(f"Decode Length: {decode_len}")
    print(f"Quantization: {args.num_bits}-bit {args.quant_mode}")
    print("\n--- Latency ---")
    print(f"Avg Prefill Latency: {avg_prefill_latency:.4f} s")
    print(f"Avg Decode Latency (total): {avg_decode_latency:.4f} s")
    print(f"Avg Decode Latency (per token): {avg_decode_latency/decode_len:.4f} s")
    print("\n--- Throughput ---")
    print(f"Prefill Throughput: {prefill_throughput:.2f} tokens/s")
    print(f"Decode Throughput: {decode_throughput:.2f} tokens/s")
    
    # CSV format for easy parsing
    print("\n--- CSV Format ---")
    print("batch_size,context_len,decode_len,prefill_latency,decode_latency,prefill_throughput,decode_throughput")
    print(f"{batch_size},{context_len},{decode_len},{avg_prefill_latency:.4f},{avg_decode_latency:.4f},{prefill_throughput:.2f},{decode_throughput:.2f}")

if __name__ == "__main__":
    benchmark_throughput()