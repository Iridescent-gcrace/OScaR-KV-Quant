# Qwen3 2-Bit Hadamard Norm Runbook

Last updated: 2026-05-10

This runbook records the commands and benchmark artifacts used for the current
Qwen3 8B 2-bit BitDecoding path with K Hadamard plus K norm.

## Repository State

- Working directory: `/home/yangrui55/BitDecoding`
- Model path:
  `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/`
- Active branch: `Iridescent-gcrace/2bit-preprocess-r256`
- Key code commit before this runbook:
  `0a745d146a69661fe55012c44c1bab78bb0c9243`

## Build And Sanity Commands

```bash
cd /home/yangrui55/BitDecoding

.venv-local/bin/python -m py_compile \
  evaluation/qwen3.py \
  evaluation/bench_throughput.py \
  evaluation/example.py \
  evaluation/scripts/run_qwen3_bitdecoding_suite.py \
  evaluation/scripts/run_qwen3_gsm8k_accuracy.py \
  bit_decode/bit_decode_interface.py \
  bit_decode/models/cache_utils.py

.venv-local/bin/python setup.py build_ext --inplace
```

The build completed successfully in the local uv environment. The last build
reported `ninja: no work to do` and relinked/copied `bit_decode_cuda`.

## GSM8K Accuracy Commands

Use the deterministic GSM8K slice runner for quick regression checks:

```bash
cd /home/yangrui55/BitDecoding

CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python \
  evaluation/scripts/run_qwen3_gsm8k_accuracy.py \
  --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/ \
  --variant hada_norm \
  --indices 0-9 \
  --dtype bfloat16 \
  --device cuda:0 \
  --num_bits 2 \
  --quant_mode k-channel \
  --group_size 32
```

Previously validated small-slice result:

- `hada_norm`: `8/10`
- `plain`: `8/10`
- `fa2`: `8/10`

## End-To-End Throughput Commands

The throughput numbers below use `evaluation/bench_throughput.py`, not an
attention-only microbenchmark. The script performs full-model prefill and decode
with random `inputs_embeds`.

Common settings:

- GPU: H800
- `batch_size=1`
- `decode_len=8`
- `iteration=1`
- `dtype=bfloat16`
- Context list:
  `1024 2048 4096 8192 16384 32768 49152 65536 98304 131072`

FA2 command template:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python evaluation/bench_throughput.py \
  --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/ \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 1 \
  --context_len ${CTX} \
  --decode_len 8 \
  --iteration 1 \
  --attn_backend flash_attention_2 \
  --num_bits 2 \
  --quant_mode k-channel \
  --group_size 32 \
  --kv_rotation none \
  --kv_norm 0
```

2-bit Hadamard + norm command template:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python evaluation/bench_throughput.py \
  --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/ \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 1 \
  --context_len ${CTX} \
  --decode_len 8 \
  --iteration 1 \
  --attn_backend bit_decoding \
  --num_bits 2 \
  --quant_mode k-channel \
  --group_size 32 \
  --kv_rotation hadamard \
  --kv_norm 1
```

Loop used for manual reproduction:

```bash
for CTX in 1024 2048 4096 8192 16384 32768 49152 65536 98304 131072; do
  CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python evaluation/bench_throughput.py \
    --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/ \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch_size 1 \
    --context_len "${CTX}" \
    --decode_len 8 \
    --iteration 1 \
    --attn_backend flash_attention_2 \
    --num_bits 2 \
    --quant_mode k-channel \
    --group_size 32 \
    --kv_rotation none \
    --kv_norm 0

  CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python evaluation/bench_throughput.py \
    --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/ \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch_size 1 \
    --context_len "${CTX}" \
    --decode_len 8 \
    --iteration 1 \
    --attn_backend bit_decoding \
    --num_bits 2 \
    --quant_mode k-channel \
    --group_size 32 \
    --kv_rotation hadamard \
    --kv_norm 1
done
```

## End-To-End Throughput Result Snapshot

CSV artifacts committed with this runbook:

- `evaluation/results/e2e_context_sweep_20260509_211434/e2e_context_sweep_all.csv`
- `evaluation/results/e2e_context_sweep_20260509_211434/e2e_context_sweep_compare.csv`
- `evaluation/results/e2e_context_sweep_20260509_211434/e2e_context_sweep_low_mid.csv`
- `evaluation/results/e2e_context_sweep_20260509_211434/e2e_context_sweep_high.csv`

Summary:

| context | FA2 decode tps | HN 2bit decode tps | HN / FA2 | FA2 peak MB | HN peak MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 11.20 | 9.26 | 0.83x | 16893.98 | 16778.52 |
| 2048 | 10.59 | 9.42 | 0.89x | 17373.44 | 17135.45 |
| 4096 | 9.89 | 8.86 | 0.90x | 18331.84 | 17848.78 |
| 8192 | 9.40 | 8.70 | 0.93x | 20246.54 | 19273.33 |
| 16384 | 7.99 | 7.70 | 0.96x | 24078.04 | 22124.54 |
| 32768 | 6.17 | 7.23 | 1.17x | 31741.03 | 27826.96 |
| 49152 | 5.05 | 6.73 | 1.33x | 39404.02 | 33529.38 |
| 65536 | 4.29 | 6.23 | 1.45x | 47067.02 | 39231.80 |
| 98304 | 3.25 | 5.33 | 1.64x | 62393.00 | 50636.63 |
| 131072 | OOM | 4.81 | n/a | OOM | 62041.47 |

Notes:

- FA2 at `context_len=131072` failed during prefill `lm_head` allocation.
- HN 2bit completed at `context_len=131072`.
- Qwen3-8B config has `max_position_embeddings=40960`; contexts above that are
  throughput stress tests rather than normal model accuracy settings.

## Remote Push Commands

Push commands executed from `/home/yangrui55/BitDecoding`:

```bash
git push iridescent HEAD:Iridescent-gcrace/2bit-preprocess-r256
git push origin HEAD:Iridescent-gcrace/2bit-preprocess-r256
git push origin HEAD:e2e
git push sankuai HEAD:Iridescent-gcrace/2bit-preprocess-r256
git push sankuai HEAD:yang/bitdecoding-e2e-tc128-hada128
```

Push results on 2026-05-10:

- `iridescent HEAD:Iridescent-gcrace/2bit-preprocess-r256`: succeeded; new branch created.
- `origin HEAD:Iridescent-gcrace/2bit-preprocess-r256`: failed because `origin` uses HTTPS and no GitHub username / credential was available in this environment.
- `origin HEAD:e2e`: failed for the same HTTPS credential reason.
- `sankuai HEAD:Iridescent-gcrace/2bit-preprocess-r256`: succeeded; new branch created.
- `sankuai HEAD:yang/bitdecoding-e2e-tc128-hada128`: succeeded; fast-forwarded from `4d85fce` to the current head.

Branches intentionally not overwritten:

- `iridescent/BitDecoding-e2e`: not fast-forwardable to the current head.
- `iridescent/Iridescent-gcrace/2bit-residual-github`: not fast-forwardable to the current head.
- `sankuai/yang/bitdecoding-e2e-2bit`: not fast-forwardable to the current head.

Use only normal fast-forward pushes. Do not force-push unless the branch owner explicitly requests it.
