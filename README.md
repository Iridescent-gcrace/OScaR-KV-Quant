# BitDecoding
[![arXiv](https://img.shields.io/badge/arXiv-2410.13276-b31b1b.svg)](https://arxiv.org/abs/2503.18773)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Setup
```bash
git clone --recursive https://github.com/DD-DuDa/BitDecoding.git
cd BitDecoding

conda create -n bitdecode python=3.10 -y
conda activate bitdecode

pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Run
Set `MODEL_PATH` to your local Qwen3 checkpoint path before running the suite.

### Smoke Test
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/run_qwen3_bitdecoding_suite.py \
  --mode smoke \
  --python_bin "$(which python)" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16
```

### Full Benchmark Sweep
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/run_qwen3_bitdecoding_suite.py \
  --mode full \
  --python_bin "$(which python)" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16
```

If the extension has already been built and Python files have already been checked, you can skip those steps:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/run_qwen3_bitdecoding_suite.py \
  --mode full \
  --skip_build \
  --skip_py_compile \
  --python_bin "$(which python)" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16
```

### Single Example
```bash
MODEL_PATH="${MODEL_PATH}" \
DTYPE=bfloat16 \
NUM_BITS=2 \
QUANT_MODE=k-channel \
GROUP_SIZE=32 \
KV_ROTATION=hadamard \
KV_NORM=1 \
ATTN_BACKEND=bit_decoding \
bash evaluation/scripts/example.sh
```

BitDecoding is a high-performance, GPU-optimized system
designed to accelerate long-context LLMs decoding with a low-bit KV
cache. Achieve **3-9x speedup** than Flash Attention v2.
![overview](imgs/overview.png)
![scheme](imgs/scheme.png)

## Benchmark
* Kernel Performance in RTX4090
![overview](imgs/4090.png)
* Kernel Performance in A100
![overview](imgs/a100.png)

## Citation
If you find BitDecoding useful or want to use in your projects, please kindly cite our paper:
```
@misc{du2025bitdecodingunlockingtensorcores,
      title={BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache}, 
      author={Dayou Du and Shijie Cao and Jianyi Cheng and Ting Cao and Mao Yang},
      year={2025},
      eprint={2503.18773},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2503.18773}, 
}
```

## Acknowledgement
BitDecoding is inspired by many open-source libraries, including (but not limited to) [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main), [flute](https://github.com/HanGuo97/flute), [Atom](https://github.com/efeslab/Atom), [omniserve](https://github.com/mit-han-lab/omniserve), [KIVI](https://github.com/jy-yuan/KIVI).
