# BUDGET_POOL=('1024' '2048' '4096' '8192' '16384' '32768') 
# BATCH_SIZE=('1' '2' '4' '8' '16' '32')

BUDGET_POOL=('16384') 
BATCH_SIZE=('1')

for batch_size in ${BATCH_SIZE[@]}; do
    for budget in ${BUDGET_POOL[@]}; do
        python3 bench_throughput.py \
            --model_path meta-llama/Llama-3.1-70B-Instruct \
            --batch_size $batch_size \
            --context_len $budget \
            --decode_len 100 \
            --iteration 1 \
            --num_bits 4 \
            --quant_mode k-channel \
            --group_size 128 \
            --attn_backend flash_attention_2 
    done
done


# flash_attention_2, flash_decoding, bit_decoding