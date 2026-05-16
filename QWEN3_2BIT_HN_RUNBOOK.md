# Qwen3 2-Bit Hadamard Norm Runbook

Last updated: 2026-05-18

This runbook records the commands and benchmark artifacts used for the current
Qwen3 8B 2-bit BitDecoding path with K Hadamard plus K norm.

## Repository State

- Working directory: `/home/yangrui55/bitdecoding`
- Model path used on this host: `/home/yangrui55/models/Qwen3-8B`
- Original source path:
  `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-llm/zhichen/huggingface.co/Qwen/Qwen3-8B/`
- Active branch: `experiment/qwen3-hn2bit-residual128`
- Validated code commit before this runbook-only update:
  `6788b0fe4b5d0a01d09e7f50fb63d7a5b6535d48`

## Verified UV Environment

The commands below were re-validated on 2026-05-11 on an H20 host.

```bash
cd /home/yangrui55/bitdecoding

uv venv --python 3.10 .venv-local
source .venv-local/bin/activate

git submodule update --init --recursive

uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
uv pip install -r requirements.txt
uv pip install --no-build-isolation flash-attn
```

Validated package versions:

- Python `3.10.17`
- PyTorch `2.6.0+cu124`
- `flash-attn 2.8.3`
- `transformers 4.57.6`

Important notes:

- `requirements.txt` must include `transformers`, `datasets`, `accelerate`, and `tqdm`.
- Install PyTorch before `flash-attn`.
- Install `flash-attn` with `--no-build-isolation`.
- `libs/cutlass` must be initialized before `setup.py build_ext --inplace`.

## Build And Sanity Commands

```bash
cd /home/yangrui55/bitdecoding

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

## Current Reproduction Method

Use the local model mirror on this machine. In this shell, the original
DolphinFS path is not the path used by the working commands.

```bash
cd /home/yangrui55/bitdecoding

export MODEL=/home/yangrui55/models/Qwen3-8B
export PY=.venv-local/bin/python
export CUDA_VISIBLE_DEVICES=1

# With CUDA_VISIBLE_DEVICES=1, pass --device cuda:0 to scripts.
```

Build before running accuracy or throughput:

```bash
$PY -m py_compile \
  evaluation/qwen3.py \
  evaluation/bench_throughput.py \
  eval_longbench_bitdecoding.py \
  bit_decode/bit_decode_interface.py

$PY setup.py build_ext --inplace
```

Operator-path Qasper accuracy, using Qwen3 2-bit K-channel Hadamard + norm,
offline V Hadamard, residual block size fixed at 128, and residual evict size
256:

```bash
CUDA_LAUNCH_BLOCKING=1 $PY eval_longbench_bitdecoding.py \
  --model_path "$MODEL" \
  --datasets qasper_e \
  --max_input_len 32768 \
  --dtype bfloat16 \
  --device cuda:0 \
  --residual_evict_size 256 \
  --offline_v_hadamard \
  --output_dir pred_e/qwen3_8b_hn2bit_offline_v_r128_ev256_qasper \
  --log_every 1 \
  --resume
```

FA2 Qasper reference:

```bash
$PY eval_longbench_batch.py \
  --model qwen3_8b \
  --model_path "$MODEL" \
  --mode baseline \
  --dataset qasper_e \
  --max_input_len 32768 \
  --output_dir pred_e

$PY eval_long_bench.py \
  --path pred_e/qwen3_8b_baseline \
  --e
```

Pseudo-quant Qasper reference aligned with `kv_cache_compression/qwen3_model.py`:

```bash
$PY eval_longbench_batch.py \
  --model qwen3_8b \
  --model_path "$MODEL" \
  --mode oscar2_rsqrt \
  --dataset qasper_e \
  --residual_length 128 \
  --offline_v_hadamard \
  --max_input_len 32768 \
  --output_dir pred_e

$PY eval_long_bench.py \
  --path pred_e/qwen3_8b_oscar2_rsqrt \
  --e
```

## Current 2026-05-18 Status

This branch is the current Qwen3-8B 2-bit BitDecoding experiment branch:

- Branch: `experiment/qwen3-hn2bit-residual128`
- Validated code commit before this runbook-only update:
  `6788b0fe4b5d0a01d09e7f50fb63d7a5b6535d48`
- Extension used for the latest runs:
  `bit_decode_cuda.cpython-310-x86_64-linux-gnu.so`, built on 2026-05-18.
- Runtime configuration:
  `attn_backend=bit_decoding`, `num_bits=2`, `quant_mode=k-channel`,
  `group_size=32`, `kv_rotation=hadamard`, `kv_norm=1`,
  `residual_block_size=128`, `residual_evict_size=256`,
  `offline_v_hadamard=true`, `dtype=bfloat16`.

The main throughput fix restores the 2-bit head-dim-128 V path from scalar PV
accumulation to a tensor-core path. The default path now gathers/dequantizes the
2-bit packed V tile into the MMA B fragment and runs:

```cpp
cute::gemm(tiled_mma, tCrA(_, _, i), tCrB_dequant(_, _, i), acc);
```

The scalar helper remains available only as a debug fallback behind
`FLASH_FORCE_2BIT_SCALAR_PV=1`; default builds use
`gemm_Vtensor_2bit_gather`.

`residual_evict_size=128` is not currently a safe parameter-only change for
Qwen3 2-bit k-channel. The main packed K cache is laid out in 256-token tiles,
and `evaluation/qwen3.py` intentionally rejects `residual_evict_size < 256` to
avoid appending a 128-token local tile into the middle of a 256-token global
tile. Supporting 128 would require coordinated changes to cache flush logic,
packed K/V layout, qpack, and CUDA kernel tile assumptions.

## Qasper-E Accuracy Snapshot

Current comparison artifacts on this host:

| path | mode | residual | offline V | 0-4k | 4-8k | 8k+ |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `pred_e/qwen3_8b_hn2bit_offline_v_r128_ev256_qasper` | BitDecoding HN 2-bit operator | block 128, evict 256 | yes | 11.30 | 9.81 | 7.11 |
| `pred_e/qasper_full_2bit_hn_20260516` | BitDecoding HN 2-bit operator | block 128, evict 256 | yes | 11.26 | 9.43 | 6.92 |
| `pred_e/longbench_full_2bit_hn_20260518` | BitDecoding HN 2-bit operator | block 128, evict 256 | yes | 11.26 | 9.43 | 7.29 |
| `evaluation/results/2026-05-13_qasper_pseudoq_aligned_r256/qwen3_8b_oscar2_rsqrt` | pseudo-quant `oscar2_rsqrt` | 256 | yes | 10.66 | 10.03 | 6.69 |
| `evaluation/results/2026-05-13_qasper_fa2_qwen3_8b/qwen3_8b_baseline` | FA2 baseline | n/a | no | 11.27 | 10.38 | 5.67 |

Notes:

- All three rows are full `qasper_e` runs with `224/224` samples,
  `max_input_len=32768`, and `num_tokens=128`.
- The pseudo-quant comparison row is the existing aligned artifact with
  `residual_length=256`; rerun pseudo-quant with `--residual_length 128` if an
  exact residual-length match to the operator tile size is required.
- The `pred_e/longbench_full_2bit_hn_20260518` row was produced as part of the
  full LongBench-E run. It should be treated as the latest Qasper number for
  the current compiled branch.

## Full LongBench-E Accuracy

Latest full run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python eval_longbench_bitdecoding.py \
  --model_path /home/yangrui55/models/Qwen3-8B \
  --datasets all \
  --max_input_len 32768 \
  --dtype bfloat16 \
  --device cuda:0 \
  --residual_evict_size 256 \
  --offline_v_hadamard \
  --output_dir pred_e/longbench_full_2bit_hn_20260518 \
  --log_every 10 \
  --resume
```

All 13 LongBench-E datasets completed. The table reports per-length-bucket
scores plus the per-dataset weighted average using the actual sample counts in
each bucket.

| dataset | 0-4k | 4-8k | 8k+ | counts | weighted avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| qasper_e | 11.26 | 9.43 | 7.29 | 100/100/24 | 10.02 |
| multifieldqa_en_e | 26.26 | 25.74 | 22.91 | 67/70/13 | 25.73 |
| hotpotqa_e | 13.25 | 13.66 | 12.79 | 100/100/100 | 13.23 |
| 2wikimqa_e | 13.66 | 11.53 | 9.61 | 100/100/100 | 11.60 |
| gov_report_e | 26.62 | 28.99 | 30.10 | 100/100/100 | 28.57 |
| multi_news_e | 22.78 | 20.59 | 20.84 | 100/100/94 | 21.41 |
| trec_e | 69.00 | 74.00 | 72.00 | 100/100/100 | 71.67 |
| triviaqa_e | 93.19 | 89.86 | 91.16 | 100/100/100 | 91.40 |
| samsum_e | 41.67 | 41.60 | 44.86 | 100/100/100 | 42.71 |
| passage_count_e | 19.25 | 11.87 | 8.14 | 100/100/100 | 13.09 |
| passage_retrieval_en_e | 92.25 | 94.33 | 92.10 | 100/100/100 | 92.89 |
| lcc_e | 73.00 | 75.31 | 69.84 | 100/100/100 | 72.72 |
| repobench-p_e | 67.38 | 60.69 | 57.57 | 100/100/100 | 61.88 |

Summary:

- Per-dataset weighted macro average: `42.84`
- Sample-weighted overall average across 3668 examples: `44.25`

Single-batch throughput sweep for FA2 and Hadamard + norm 2-bit:

Note: `evaluation/bench_throughput.py` does not currently expose
`--offline_v_hadamard`; this sweep measures the BitDecoding Hadamard + norm
operator path and FA2 reference path.

```bash
OUT=evaluation/results/qwen3_hn2bit_vs_fa2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

set -o pipefail
for CTX in 1024 2048 4096 8192 16384 32768 49152 65536 98304 131072; do
  $PY evaluation/bench_throughput.py \
    --model_path "$MODEL" \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch_size 1 \
    --context_len "$CTX" \
    --decode_len 8 \
    --iteration 1 \
    --attn_backend flash_attention_2 \
    --num_bits 2 \
    --quant_mode k-channel \
    --group_size 32 \
    --kv_rotation none \
    --kv_norm 0 \
    2>&1 | tee "$OUT/fa2_ctx${CTX}.log"

  $PY evaluation/bench_throughput.py \
    --model_path "$MODEL" \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch_size 1 \
    --context_len "$CTX" \
    --decode_len 8 \
    --iteration 1 \
    --attn_backend bit_decoding \
    --num_bits 2 \
    --quant_mode k-channel \
    --group_size 32 \
    --kv_rotation hadamard \
    --kv_norm 1 \
    --residual_evict_size 256 \
    2>&1 | tee "$OUT/hn2bit_r128_ev256_ctx${CTX}.log"
done
```

Current HN 2-bit throughput result from this host:

- Output directory:
  benchmark log was captured from `evaluation/bench_throughput.py` on the
  current compiled extension.
- Command settings: Qwen3-8B, `batch_size=1`, `decode_len=8`, `iteration=1`,
  `dtype=bfloat16`, `attn_backend=bit_decoding`, `num_bits=2`,
  `quant_mode=k-channel`, `group_size=32`, `kv_rotation=hadamard`, `kv_norm=1`,
  `residual_block_size=128`, `residual_evict_size=256`.
- `evaluation/bench_throughput.py` does not expose `--offline_v_hadamard`, so
  this measures the current BitDecoding Hadamard + norm operator path only.
- The current CUDA path dispatches 2-bit head-dim-128 PV through
  `gemm_Vtensor_2bit_gather` and tensor-core `cute::gemm`. The scalar path is
  only a compile-time fallback.

128k single-batch result:

| context | decode len | prefill latency | decode latency/token | prefill tps | decode tps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 131072 | 8 | 72.1089 s | 0.0367 s | 1817.69 | 27.23 |

Smoke/full suite wrapper:

```bash
$PY evaluation/scripts/run_qwen3_bitdecoding_suite.py \
  --mode smoke \
  --model_path "$MODEL" \
  --device cuda:0 \
  --dtype bfloat16
```

Current caveat: the current validation focuses on full 256-token packed tiles.
Keep `evaluation/test_v_pack_2bit_layout.py`, the random q=0 oracle, Qasper,
and the 128k throughput benchmark as regression gates before publishing new
kernel changes.

## GSM8K Accuracy Commands

Use the deterministic GSM8K slice runner for quick regression checks:

```bash
cd /home/yangrui55/bitdecoding

CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python \
  evaluation/scripts/run_qwen3_gsm8k_accuracy.py \
  --model_path "$MODEL" \
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
  --model_path "$MODEL" \
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
  --model_path "$MODEL" \
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
    --model_path "$MODEL" \
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
    --model_path "$MODEL" \
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

Push commands executed from `/home/yangrui55/bitdecoding`:

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
