# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import torch
import random
import argparse
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_decode import DynamicCache, StaticCache, Cache
import transformers.cache_utils
transformers.cache_utils.DynamicCache = DynamicCache
transformers.cache_utils.StaticCache = StaticCache
transformers.cache_utils.Cache = Cache

from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset


def resolve_model_components(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "llama":
        from llama import LlamaForCausalLM
        return config, LlamaForCausalLM
    if model_type == "qwen3":
        from qwen3 import Qwen3ForCausalLM
        return config, Qwen3ForCausalLM
    raise ValueError(f"Unsupported model_type: {model_type}")


def resolve_torch_dtype(config, dtype_name):
    if dtype_name == "auto":
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            return getattr(torch, config_dtype)
        if isinstance(config_dtype, torch.dtype):
            return config_dtype
        return torch.float16
    return getattr(torch, dtype_name)


def trim_gsm8k_answer(text):
    match = re.search(r"####\s*-?[\d,]+(?:\.\d+)?", text)
    if match is None:
        return text
    return text[: match.end()].rstrip()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLaMA model with KIVI')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--max_length', type=int, default=131072, help='Maximum length of the input sequence')
    parser.add_argument('--dtype', type=str, default='auto', help='Torch dtype: auto, float16, bfloat16, float32')
    parser.add_argument('--num_bits', type=int, default=4, help='Number of bits for quantization')
    parser.add_argument('--quant_mode', type=str, default='k-channel', help='Quantization mode')
    parser.add_argument('--group_size', type=int, default=None, help='Group size for quantization')
    parser.add_argument('--kv_rotation', type=str, default='none', help='KV rotation mode, e.g. none or hadamard')
    parser.add_argument('--kv_norm', type=str, default='0', help='KV norm mode, e.g. 0 or 1')
    parser.add_argument('--attn_backend', type=str, default='flash_attention_2', help='Attention implementation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Model/device placement')
    args = parser.parse_args()

    # For reproducibility 
    random.seed(0)
    torch.manual_seed(0)

    if args.group_size is None:
        args.group_size = 32 if args.num_bits == 2 else 128

    config, model_cls = resolve_model_components(args.model_path)
    dtype = resolve_torch_dtype(config, args.dtype)

    config._attn_implementation = "flash_attention_2"
    config.attn_backend = args.attn_backend
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.kv_rotation = args.kv_rotation
    config.kv_norm = args.kv_norm
    config.residual_block_size = 256 if args.num_bits == 2 else 128

    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map={"": args.device}
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
    ).to(args.device)

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=enc.pad_token_id,
        max_new_tokens=125
    )
    config_str = f"# prompt tokens: {inputs.input_ids.shape[1]}"

    # print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nOutput:")
    # print("\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nOutput:")
    generated = enc.decode(output[0].tolist()[inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(trim_gsm8k_answer(generated))

if __name__ == "__main__":
    main()
