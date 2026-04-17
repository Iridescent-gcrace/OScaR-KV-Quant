PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
DTYPE="${DTYPE:-auto}"
NUM_BITS="${NUM_BITS:-4}"
QUANT_MODE="${QUANT_MODE:-k-channel}"
GROUP_SIZE="${GROUP_SIZE:-128}"
ATTN_BACKEND="${ATTN_BACKEND:-bit_decoding}"

"${PYTHON_BIN}" example.py \
    --model_path "${MODEL_PATH}" \
    --max_length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --num_bits "${NUM_BITS}" \
    --quant_mode "${QUANT_MODE}" \
    --group_size "${GROUP_SIZE}" \
    --attn_backend "${ATTN_BACKEND}" # flash_attention_2, flash_decoding, bit_decoding


# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen3-8B
