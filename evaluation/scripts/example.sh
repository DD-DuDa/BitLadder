python example.py \
    --model_path Qwen/Qwen3-8B \
    --max_length 131072 \
    --num_bits 4 \
    --quant_mode k-channel \
    --group_size 128 \
    --attn_backend bit_decoding # flash_attention_2, flash_decoding, bit_decoding


# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen3-8B