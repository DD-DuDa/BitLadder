# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
from llama import LlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLaMA model with KIVI')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--max_length', type=int, default=131072, help='Maximum length of the input sequence')
    parser.add_argument('--num_bits', type=int, default=4, help='Number of bits for quantization')
    parser.add_argument('--quant_mode', type=str, default='k-channel', help='Quantization mode')
    parser.add_argument('--group_size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--attn_backend', type=str, default='flash_attention_2', help='Attention implementation')
    args = parser.parse_args()

    # For reproducibility 
    random.seed(0)
    torch.manual_seed(0)

    config = LlamaConfig.from_pretrained(args.model_path)

    config.attn_backend = args.attn_backend
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.residual_block_size = 128 if args.num_bits == 4 else 256

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    enc = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='left',  # Add this line
        pad_token='</s>'      # Add this line
    )

    dataset = load_dataset('gsm8k', 'main')

    prompt = ''
    for i in range(15):
        prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
    prompt += "Arnel had ten boxes of pencils with the same number of pencils in each box. He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?"

    inputs = enc(
        prompt,
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_attention_mask=True
    ).to('cuda')

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=enc.pad_token_id,
        max_new_tokens=125
    )
    config_str = f"# prompt tokens: {inputs.input_ids.shape[1]}"

    print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nOutput:")
    # print("\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nOutput:")
    print(enc.decode(output[0].tolist()[inputs.input_ids.shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
