# Qwen3 2-Bit K-Only Tasks

Last updated: 2026-04-21

This document is the source of truth for the current Qwen3 2-bit integration work in this repo. It is intended to preserve task context and reduce reliance on conversational memory.

## Scope

- Model: Qwen3
- Path under active development: `attn_backend=bit_decoding`, `num_bits=2`
- Current focus: K-side operations only
- Do not change prefill attention math itself
- Do not spend time on V-side Hadamard fusion for now

## Current Design Decisions

### Prefill

- Prefill is the `q_len != 1` path in [`evaluation/qwen3.py`](/home/yangrui55/BitDecoding-e2e/evaluation/qwen3.py).
- Prefill attention computation itself should remain unchanged.
- Prefill attention still uses:
  - `flash_attention_2` when available
  - otherwise the eager attention fallback
- Only the KV cache build part after prefill attention is meant to change.

### Decode

- Decode is the `q_len == 1` path in [`evaluation/qwen3.py`](/home/yangrui55/BitDecoding-e2e/evaluation/qwen3.py).
- Decode attention should continue to use `fwd_kvcache_int(...)`.
- Residual KV should still accumulate in floating-point cache.
- When residual reaches `residual_block_size`, it should flush into packed cache.
- For the current `2-bit + k-channel` path, `residual_block_size=128` is both the user-visible flush threshold and the CUDA pack / flush block size.

### K-side math

- K uses Hadamard before quantization.
- K uses token-wise L2 norm across `head_num * head_dim`.
- Cache stores normalized K, denoted conceptually as `K_unit`.
- Token-wise K norm is stored separately as metadata `k_norm`.
- During decode attention:
  - low-bit K is dequantized to `K_unit`
- `QK_unit` is computed
- token-wise `k_norm` is multiplied back on the logits

### Hadamard implementation note

- Kernel-side Hadamard should follow the HadaCore design from [`HadaCore: Tensor Core Accelerated Hadamard Transform Kernel`](https://arxiv.org/abs/2412.08832), not a naive scalar FWHT implementation.
- Prefer a tensor-core-friendly decomposition that preserves Hadamard recursion but uses matrix-shaped work decomposition and reorder/transpose steps instead of materializing the Hadamard matrix.
- For the current `head_dim=128` path, treat the transform as `128 = 16 x 8`:
  - use HadaCore-style size-16 Tensor Core stages as the main building block
  - finish with one extra size-8 stage for the non-power-of-16 remainder, following the paper's mixed-radix handling
- Keep the implementation in registers / shared memory where possible and avoid a Python-side FWHT fallback on the hot path.

### V-side math

- V norm is no longer part of the plan.
- V Hadamard is not the current focus.
- V Hadamard is currently disabled on the Qwen3 bit-decoding path.
- V Hadamard could be absorbed into `W_v` later, but this is explicitly out of scope for the current phase.

## What Has Already Been Changed

### 1. K norm metadata path is wired through decode

- `K` token-wise norm metadata is now produced by the dedicated K preprocess kernel in [`csrc/bit_decode/src/k_preprocess.cu`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/k_preprocess.cu), and consumed from Python in [`evaluation/qwen3.py`](/home/yangrui55/BitDecoding-e2e/evaluation/qwen3.py).
- Cache support for residual and packed `k_norm` exists in [`bit_decode/models/cache_utils.py`](/home/yangrui55/BitDecoding-e2e/bit_decode/models/cache_utils.py).
- Python/C++ interface support for `k_norm` exists in:
  - [`bit_decode/bit_decode_interface.py`](/home/yangrui55/BitDecoding-e2e/bit_decode/bit_decode_interface.py)
  - [`csrc/bit_decode/src/include/flash.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/include/flash.h)
  - [`csrc/bit_decode/decode_api.cpp`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/decode_api.cpp)
- Decode kernel consumes `k_norm` in:
  - residual path
  - packed path
  in [`csrc/bit_decode/src/flash_fwd_kernel.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/flash_fwd_kernel.h)

### 1.5. K preprocess now has a dedicated CUDA kernel

- A dedicated K-only preprocess kernel now exists in [`csrc/bit_decode/src/k_preprocess.cu`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/k_preprocess.cu).
- It handles:
  - optional K Hadamard for the current `head_dim=128` path
  - token-wise L2 norm across `head_num * head_dim`
  - normalized `K_unit` output
  - `k_norm` output
- The current kernel-side Hadamard is an interim shared-memory FWHT, not yet the final HadaCore-style tensor-core implementation.
- [`evaluation/qwen3.py`](/home/yangrui55/BitDecoding-e2e/evaluation/qwen3.py) now uses this kernel for K-side cache preprocessing in both:
  - prefill cache build
  - decode residual staging
- This removes the Python-side K FWHT / K norm hot path, but it is still a separate preprocess kernel, not yet fused into qpack itself.

### 1.6. Prefill packed K no longer materializes preprocess results in Python

- A combined C++ / pybind entry now exists for `K preprocess -> qpack`:
  - it launches the dedicated K preprocess kernel
  - then immediately feeds the transformed K into the existing qpack path
  - and returns `k_norm`
- [`evaluation/qwen3.py`](/home/yangrui55/BitDecoding-e2e/evaluation/qwen3.py) now uses that combined path for the packed portion of prefill KV cache build.
- The remaining standalone K preprocess use on the hot path is now limited to:
  - prefill residual tail tokens that stay in floating-point residual cache
  - decode residual staging before `fwd_kvcache_int(...)`

### 2. Decode flush path status

- The old decode flush path used to override kernel outputs with a Python-side repack.
- That extra Python repack has been removed for the full kernel-flush path.
- The current `2-bit + k-channel + residual_block_size=128` path flushes directly with the CUDA kernel and no longer keeps the old `128 -> 256` compatibility staging in the wrapper.

### 3. V norm has been removed

- `_pack_kv_cache_block()` no longer computes `value_norm`.
- It no longer does:
  - `value_states / value_norm`
  - denorm folding into `v_params`
- Current meaning:
  - K has Hadamard + token-wise norm
  - V does not use norm

## Important Clarification

The current implementation is **not yet a true K-side fused qpack implementation**.

What is already inside CUDA:

- K-only preprocess kernel for Hadamard + token-wise norm
- standard low-bit dequant/pack logic
- K-side `qk` denorm on logits via `k_norm`

What is still outside CUDA:

- full fusion of K preprocess into qpack / residual-flush kernels
- V-side Hadamard, if we still choose to keep it at all

So the current state is:

- correctness-first K norm metadata path exists
- decode can consume `k_norm`
- K-side pre-quant transform is now produced by a standalone CUDA preprocess kernel
- but that preprocess is not yet fused into qpack / residual-flush kernels

## Current Hotspots / Why Performance Is Still Worse Than Expected

The removed Python re-pack only affected residual flush. That was not the only hot path.

Remaining hot costs include:

- Python-side staging of residual tensors before `fwd_kvcache_int(...)`
- CUDA-side `apply_token_norm(...)` still being a simple post-GEMM scaling path rather than a deeper fusion into dequant/MMA

## Immediate Goal

Move the K-side pre-quant transform into kernel code while keeping the overall prefill/decode control flow unchanged.

In practical terms:

- Prefill attention remains unchanged
- Decode attention remains `fwd_kvcache_int(...)`
- Only K-side cache-build / flush qpack behavior should change

## Task List

### Completed

- [x] Wire `k_norm` through cache, Python wrapper, C++ params, and decode kernel
- [x] Remove decode flush Python repack
- [x] Remove V norm from current path
- [x] Remove Python-side V Hadamard from the Qwen3 bit-decoding path
- [x] Keep prefill attention math unchanged
- [x] Align `2-bit + k-channel` pack / residual flush on `128`-token blocks
- [x] Move packed-prefill K preprocess out of Python into a combined C++ pack path

### In Progress

- [ ] Replace the remaining Python-side cache preprocessing with kernel-side logic
- [ ] Align kernel-side Hadamard implementation with the HadaCore-style design

### Next Tasks

- [ ] Fuse the dedicated K preprocess kernel into prefill qpack:
  - remove the standalone K preprocess launch
  - keep `k_norm` writeout
  - keep compatibility with `group_size=32`
- [ ] Fuse the same K-side logic into decode residual-flush qpack
- [ ] Re-check whether `_pack_kv_cache_block()` can be fully removed or reduced to a compatibility fallback
- [ ] Benchmark again after kernel-side K transform is in place

### Later Tasks

- [ ] Optimize CUDA-side `apply_token_norm(...)`
- [ ] Consider whether V Hadamard should be absorbed into `W_v`
- [ ] Revisit accuracy evaluation after K-side kernel integration is complete

## File Map

### Python / integration

- [`evaluation/qwen3.py`](/home/yangrui55/BitDecoding-e2e/evaluation/qwen3.py)
- [`bit_decode/models/cache_utils.py`](/home/yangrui55/BitDecoding-e2e/bit_decode/models/cache_utils.py)
- [`bit_decode/bit_decode_interface.py`](/home/yangrui55/BitDecoding-e2e/bit_decode/bit_decode_interface.py)

### C++ / CUDA entrypoints

- [`csrc/bit_decode/decode_api.cpp`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/decode_api.cpp)
- [`csrc/bit_decode/src/include/flash.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/include/flash.h)

### Kernels

- [`csrc/bit_decode/src/flash_fwd_kernel.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/flash_fwd_kernel.h)
- [`csrc/bit_decode/src/k_preprocess.cu`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/k_preprocess.cu)
- [`csrc/bit_decode/src/include/qpack.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/include/qpack.h)
- [`csrc/bit_decode/src/include/dequantize.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/include/dequantize.h)
- [`csrc/bit_decode/src/include/utils.h`](/home/yangrui55/BitDecoding-e2e/csrc/bit_decode/src/include/utils.h)

## Validation Snapshot

These checks have already succeeded with the current code state:

- Extension builds successfully with:
  - `/home/yangrui55/mllm/.venv/bin/python setup.py build_ext --inplace`
- `example.sh` runs successfully for:
  - `Qwen3-8B`
  - `bit_decoding`
  - `2-bit`
  - `KV_ROTATION=hadamard`
  - `KV_NORM=1`
- `bench_throughput.sh` also runs successfully on the same path

Current performance should not be treated as final because K-side transform generation is still partly on the Python hot path.

## Do Not Re-open These Questions Unless Requirements Change

- Prefill attention math should remain unchanged.
- Current phase is K-only.
- V norm is out.
- V Hadamard fusion into `W_v` is deferred.
- Decode still uses `fwd_kvcache_int(...)`; do not redesign the whole decode path.
