import argparse
import os
import sys

import pdb
import torch
from peft import LoraConfig, get_peft_model

from transformers import AutoProcessor      # do not remove this line
from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    # Env
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )

    # Model
    parser.add_argument("--version", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--model_max_length", default=4096, type=int)

    # Lora
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="qkv_proj", type=str)

    # Training and save
    parser.add_argument("--weight", type=str, required=True)
    return parser.parse_args(args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head"], verbose=True):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    # lora_namespan_exclude += ["vision_model", "img_projection", "visual_model"]
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def main(args):
    args = parse_args(args)
    args.save_path = os.path.dirname(args.weight) + "/merged_model"

    # Create processor
    processor = Qwen2VLProcessor.from_pretrained(args.version,
                                               padding_side='right',
                                               model_max_length=args.model_max_length)

    # use unk rather than eos token to prevent endless generation
    # processor.tokenizer.pad_token = processor.tokenizer.unk_token
    # processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    # processor.tokenizer.padding_side = 'right'

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        _attn_implementation="eager",
        # **model_args
    )

    model.config.use_cache = False
    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    lora_r = args.lora_r
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_target_linear_names(model, lora_namespan_exclude=["visual"])
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    state_dict = torch.load(args.weight, map_location="cpu")
    # model.load_state_dict(state_dict, strict=True)
    model.load_state_dict(state_dict, strict=False)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict, safe_serialization=False)
    processor.save_pretrained(args.save_path)

if __name__ == "__main__":
    main(sys.argv[1:])