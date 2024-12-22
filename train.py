import argparse
import re
import os
import shutil
import sys
import pdb
import time
import json
import wandb
from functools import partial
from datetime import datetime

import deepspeed
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoProcessor, BitsAndBytesConfig      # do not remove this line
from main.trainer import train
from main.evaluator import validate as validate_default
from main.eval_mind2web import validate_mind2web
from main.eval_omniact_nav import validate_omniact_nav
from main.eval_aitw import validate_aitw
from main.eval_aitz import validate_aitz
from main.eval_screenspot import validate_screenspot
from main.eval_odyssey import validate_odyssey
from main.eval_guiworld import validate_guiworld

from model.utils import find_target_linear_names
from data.dataset import HybridDataset, collate_fn
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_args_to_json, create_log_dir

def env_init(distributed=True):
    print("Init Env for Distributed Training")
    if distributed:
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", 'localhost')
            os.environ['MASTER_PORT']  = os.environ.get("MASTER_PORT", "12875")
            os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
            os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
            os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

            print(f"OMPI_COMM_WORLD_SIZE: {os.environ['OMPI_COMM_WORLD_SIZE']}")
            print(f"OMPI_COMM_WORLD_RANK: {os.environ['OMPI_COMM_WORLD_RANK']}")
            print(f"OMPI_COMM_WORLD_LOCAL_RANK: {os.environ['OMPI_COMM_WORLD_LOCAL_RANK']}")
            print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
            print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        elif 'WORLD_SIZE' in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", 'localhost')
            os.environ['MASTER_PORT']  = os.environ.get("MASTER_PORT", "12875")

            print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
            print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
        else:
            return
    else:
        return

# a tricky way to broadcast timestamp to all ranks
def broadcast_timestamp(src=0, local_rank=0):
    if dist.get_rank() == src:
        timestamp = torch.tensor([datetime.now().timestamp()], dtype=torch.float64).to(f'cuda:{local_rank}')
    else:
        timestamp = torch.zeros(1, dtype=torch.float64).to(f'cuda:{local_rank}')

    dist.broadcast(timestamp, src=src)
    time_str = datetime.fromtimestamp(timestamp.item()).strftime('%Y-%m-%d_%H-%M-%S')
    return time_str

def parse_args(args):
    parser = argparse.ArgumentParser(description="ShowUI Model Training")
    # Env
    parser.add_argument("--wandb_key", default=None, type=str, help="wandb key to monitor training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--ds_zero", choices=['zero1', 'zero2', 'zero3'], default='zero2', help="deepspeed zero stage")

    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--attn_imple", choices=["eager", "flash_attention_2", "sdpa"], default="eager")
    parser.add_argument("--liger_kernel", action="store_true", default=False)

    # Model & Ckpt
    parser.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct", choices=["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"])
    parser.add_argument("--version", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--model_max_length", default=8192, type=int)
    parser.add_argument("--local_weight", action="store_true", default=False)
    parser.add_argument("--local_weight_dir",  default=".", help="default path to load the model weight")

    parser.add_argument("--num_crops", default=16, type=int) # only available for Phi3.5v; 16 as the default setting that aligns with the Phi3v;
    parser.add_argument("--min_visual_tokens", default=256, type=int) # only available for qwen2-vl-2b;
    parser.add_argument("--max_visual_tokens", default=1280, type=int) # 
    parser.add_argument("--tune_visual_encoder", action="store_true", default=False)
    parser.add_argument("--tune_visual_encoder_projector", action="store_true", default=False)
    parser.add_argument("--freeze_lm_embed", action="store_true", default=False)
    parser.add_argument("--decay_factor", default=1.0, type=float)  # history modeling
    # vid
    parser.add_argument("--num_frames", default=1, type=int)
    parser.add_argument("--max_frames", default=16, type=int)
    parser.add_argument("--frame_sampling", default="uniform", choices=["uniform", "random", "keyframe"])  

    # Data
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--dataset", default="seeclick", type=str)

    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument("--uniform_sample", action="store_true", default=False)
    parser.add_argument("--random_sample", action="store_true", default=False)
    parser.add_argument("--record_sample", action="store_true", default=False)

    # ui setting for modelling
    parser.add_argument("--layer_skip_ratio", default=0, type=float)
    parser.add_argument("--layer_skip_type", default='[1,28,0]', type=str)  # qwen2-vl-2b-llm
    parser.add_argument("--vis_layer_skip_ratio", default=0, type=float)  # qwen2-vl-2b-clip-encoder
    parser.add_argument("--vis_layer_skip_type", default='[1,32,0]', type=str)  # qwen2-vl-2b-clip-encoder
    parser.add_argument("--vis_layer_skip_keep", action="store_true", default=False)
    parser.add_argument("--merge_style", type=str, default='s0')

    # ui setting for preprocessor
    parser.add_argument("--merge_pre_assign", action="store_true", default=False)
    # ui setting for preprocessor & modelling
    parser.add_argument("--layer_skip_rand", action="store_true", default=False) # only work for without pre-assign

    # ui setting in dataloder;
    parser.add_argument("--merge_patch", default=0, type=float)
    parser.add_argument("--merge_threshold", default=0, type=float)
    parser.add_argument("--merge_inference", action="store_true", default=False)
    parser.add_argument("--merge_random", type=str, choices=["grid", "shuffle"], default=None)

    # PT / SFT    
    parser.add_argument("--assistgui_data", default="hf_train_full", type=str)
    parser.add_argument("--seeclick_data", default="hf_train", type=str)
    parser.add_argument("--synthesis_data", default="hf_train_10k", type=str)
    parser.add_argument("--showui_data", default="hf_train", type=str)
    parser.add_argument("--guiexp_data", default="hf_train_ground", type=str)
    parser.add_argument("--guiexpweb_data", default="hf_train_v1", type=str)
    parser.add_argument("--guienv_data", default="hf_train", type=str)
    parser.add_argument("--guiact_data", default="hf_train_web-single_v2", type=str)
    parser.add_argument("--guiact_g_data", default="hf_train_web-single_ground", type=str)
    parser.add_argument("--guichat_data", default="hf_train", type=str)
    parser.add_argument("--ricosca_data", default="hf_train_ricosca", type=str)
    parser.add_argument("--widget_data", default="hf_train_widget", type=str)
    parser.add_argument("--screencap_data", default="hf_train_screencap", type=str)
    parser.add_argument("--amex_data", default="hf_train", type=str)
    parser.add_argument("--amexcap_data", default="hf_train_cap", type=str)
    parser.add_argument("--xlam_data", default="hf_train", type=str)
    parser.add_argument("--llava_data", default="llava_v1_5_mix665k", type=str)
    parser.add_argument("--act2cap_data", default="hf_train", type=str)
    parser.add_argument("--omniact_data", default="hf_train_showui_desktop", type=str)
    parser.add_argument("--omniact_nav_data", default="hf_train", type=str)
    parser.add_argument("--osatlas_data", default="hf_desktop", type=str)

    # Downstream train.
    parser.add_argument("--miniwob_data", default="hf_train", type=str)
    parser.add_argument("--aitw_data", default="hf_train", type=str)
    parser.add_argument("--aitz_data", default="hf_train", type=str)
    parser.add_argument("--mind2web_data", default="hf_train", type=str)
    parser.add_argument("--odyssey_data", default="hf_train_random", type=str)
    parser.add_argument("--guiworld_data", default="hf_train", type=str)
    # Downstream val.
    parser.add_argument("--val_sample_rates", default="1", type=str)
    parser.add_argument("--val_dataset", default="mind2web", type=str)
    parser.add_argument("--val_mind2web_data", default="hf_test_full", type=str)
    parser.add_argument("--val_aitw_data", default="hf_test", type=str)
    parser.add_argument("--val_aitz_data", default="hf_test", type=str)
    parser.add_argument("--val_guiact_data", default="hf_test_web-single", type=str)
    parser.add_argument("--val_screenspot_data", default="hf_test_full", type=str)
    parser.add_argument("--val_odyssey_data", default="hf_test_random", type=str)
    parser.add_argument("--val_guiworld_data", default="hf_test_mcq", type=str)
    parser.add_argument("--val_omniact_nav_data", default="hf_test", type=str)

    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--num_turn", default=1, type=int)
    parser.add_argument("--text2point", default=1, type=float)
    parser.add_argument("--text2bbox", default=0, type=float)
    parser.add_argument("--point2text", default=0, type=float)
    parser.add_argument("--bbox2text", default=0, type=float)
    parser.add_argument("--shuffle_image_token", action="store_true", default=False, help="shuffle image token for training")
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--xy_int", action="store_true", default=False)
    parser.add_argument("--uniform_prompt", action="store_true", default=False)

    parser.add_argument("--skip_readme_train", action="store_true", default=False)
    parser.add_argument("--skip_readme_test", action="store_true", default=False)

    parser.add_argument("--num_history", default=4, type=int)
    parser.add_argument("--interleaved_history", default='tttt',  choices=['tttt', 'vvvv', 'vtvt', 'tvtv', 'vvtt', 'ttvv'])
    parser.add_argument("--draw_history", default=0, type=int)
    
    # aitz-coat
    parser.add_argument("--prob_plan", default=0, type=float)
    parser.add_argument("--prob_cap", default=1, type=float)
    parser.add_argument("--prob_res", default=1, type=float)
    parser.add_argument("--prob_think", default=1, type=float)
    # grounding
    parser.add_argument("--crop_min", default=1, type=float)
    parser.add_argument("--crop_max", default=1, type=float)

    # Lora
    parser.add_argument("--use_qlora", action="store_true", default=False)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="qkv_proj", type=str)

    # Training
    parser.add_argument("--exp_id", default="debug", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--log_base_dir", default="../runs", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--warmup_type", default="linear", type=str)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)

    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--print_freq", default=1, type=int)

    parser.add_argument("--num_zoom_in", default=0, type=int)

    parser.add_argument("--debug", action="store_true", default=False) # for debugging, will not save model and monitor
    return parser.parse_args(args)

def main(args):
    env_init()
    args = parse_args(args)
    args.global_rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.attn_imple in ["eager", "sdpa"]:
        # suggested by https://github.com/Lightning-AI/litgpt/issues/327
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if args.global_rank == 0 else None

    args.distributed = args.world_size > 1
    # ensure all rank share the same timestamp
    if args.distributed:
        print(f"Using distributed training with {args.world_size} GPUs, with rank {os.environ['RANK']}")
        deepspeed.init_distributed(dist_backend="nccl", rank=args.global_rank, world_size=args.world_size)
        timestamp = broadcast_timestamp(0, args.local_rank)

    args.log_dir = os.path.join(args.log_base_dir, args.exp_id, timestamp)
    args.tmp_dir = os.path.join(args.log_dir, "tmp")

    # must provide wandb-key
    assert args.wandb_key is not None
    wandb.login(key=args.wandb_key)

    writer = None
    if args.global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True)
        save_args_to_json(args, os.path.join(args.log_dir, "args.json"))        # save args to json
        if not args.debug:
            writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))       # init. tensorboard writer
            # init. wandb monitor
            wandb.init(
                project="ShowUI",
                group=args.exp_id,
                name=f'{args.exp_id}_{timestamp}',
                dir=args.log_dir,
                config=args
            )
    print(f"Start job {args.exp_id}")

    # Create processor
    if args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"]:
        from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        model_id = args.model_id.replace("Qwen/", "")
        if args.local_weight:
            model_url = f"{args.local_weight_dir}/{model_id}"
        else:
            model_url = args.model_id
        processor = Qwen2VLProcessor.from_pretrained(
                                                    model_url,
                                                    min_pixels=args.min_visual_tokens *28*28, 
                                                    max_pixels=args.max_visual_tokens *28*28,
                                                    model_max_length=args.model_max_length,
                                                    merge_pre_assign=args.merge_pre_assign,
                                                    layer_skip_rand=args.layer_skip_rand,
                                                    layer_skip_ratio=args.layer_skip_ratio,
                                                    )
        processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    # Create model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    # follow by https://github.com/GaiZhenbiao/Phi3V-Finetuning/blob/main/train_phi3v.py
                    llm_int8_skip_modules=["img_projection"],
                ) if args.use_qlora else None

    if args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"]:
        # from transformers import Qwen2VLForConditionalGeneration
        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        qwen_layer_lm = 28
        qwen_layer_vis = 32
        def parse_layer_type(str_ranges, L, default=0):
            result = [default] * L
            matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', str_ranges)
            for start, end, value in matches:
                start, end, value = int(start) - 1, int(end) - 1, int(value)
                if end >= L:
                    end = L - 1
                result[start:end + 1] = [value] * (end - start + 1)
            return result
        layer_skip_type = parse_layer_type(args.layer_skip_type, qwen_layer_lm)
        vis_layer_skip_type = parse_layer_type(args.vis_layer_skip_type, qwen_layer_vis)

        model_id = args.model_id.replace("Qwen/", "")
        if args.local_weight:
            model_url = f"{args.local_weight_dir}/{model_id}"
        else:
            model_url = args.model_id

        if args.liger_kernel:
            print("Apply liger kernel to Qwen2-VL")
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
            apply_liger_kernel_to_qwen2_vl()

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            # args.version,
            # args.model_id,
            model_url,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            _attn_implementation=args.attn_imple,
            quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
            layer_skip_rand=args.layer_skip_rand,
            layer_skip_ratio=args.layer_skip_ratio,
            layer_skip_type=layer_skip_type,
            vis_layer_skip_ratio=args.vis_layer_skip_ratio,
            vis_layer_skip_type=vis_layer_skip_type,
            vis_layer_skip_keep=args.vis_layer_skip_keep,
            merge_style=args.merge_style,
        )
        if args.version != args.model_id:
            state_dict = torch.load(args.version, map_location="cpu")
            # please remove the self-defined layer for avoid error;
            model.load_state_dict(state_dict, strict=False)
    model.config.use_cache = False
    # pdb.set_trace()
    # if only for evaluation, no need to prepare lora
    if not args.eval_only and args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # Config lora using peft library
    lora_r = args.lora_r
    # if not args.eval_only and lora_r > 0:
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        if args.model_id in ["microsoft/Phi-3-vision-128k-instruct", "microsoft/Phi-3.5-vision-instruct"]:
            exclude_module = ["vision_model", "img_projection", "visual_model"] if not args.tune_visual_encoder else []
        elif args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"]:
            exclude_module = ["visual"] if not args.tune_visual_encoder else []
            exclude_module += ["lm_head"] if args.freeze_lm_embed else exclude_module
            # this might be applied for the style variant; should be removed in future;
            exclude_module += ["weight_layer"]

        lora_target_modules = find_target_linear_names(model, 
                                                    lora_namespan_exclude=exclude_module)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if args.global_rank == 0:
            model.print_trainable_parameters()

        model_child = model.model.model
    else:
        model_child = model.model

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if not args.tune_visual_encoder:
        if args.model_id in ["microsoft/Phi-3-vision-128k-instruct", "microsoft/Phi-3.5-vision-instruct"]:
            # model_child.embed_tokens.weight.requires_grad?
            for p in model_child.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
            for p in model_child.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        elif args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"]:
            if args.lora_r > 0:
                for p in model.base_model.model.visual.parameters():
                    p.requires_grad = False
            elif args.lora_r == 0:
                for p in model.visual.parameters():
                    p.requires_grad = False
            
    if args.tune_visual_encoder_projector:
        for k, p in model.named_parameters():
            if 'visual.merger' in k:
                p.requires_grad = True

    if args.freeze_lm_embed:
        if args.model_id in ["microsoft/Phi-3-vision-128k-instruct", "microsoft/Phi-3.5-vision-instruct"]:
            raise ValueError("Not supported for Phi-3-vision-128k-instruct and Phi-3.5-vision-instruct")
        elif args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"]:
            if args.lora_r > 0:
                for p in model_child.embed_tokens.parameters():
                    p.requires_grad = False
            elif args.lora_r == 0:
                for p in model_child.embed_tokens.parameters():
                    p.requires_grad = False

    if args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B"]:
        if args.merge_style != 's0':
            for n, p in model.named_parameters():
                if 'weight_layer' in n:
                    p.requires_grad = True

    # Check trainable parameters
    list_of_params_to_optimize = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if args.global_rank == 0:
                print("[Name]", n, " [Shape]", p.shape)
            list_of_params_to_optimize.append(p)
    
    # Create dataset
    train_dataset = HybridDataset(
        args.dataset_dir,
        processor,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * args.world_size,
        precision=args.precision,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        miniwob_data=args.miniwob_data,
        assistgui_data=args.assistgui_data,
        seeclick_data=args.seeclick_data,
        showui_data=args.showui_data,
        aitw_data=args.aitw_data,
        aitz_data=args.aitz_data,
        mind2web_data=args.mind2web_data,
        odyssey_data=args.odyssey_data,
        ricosca_data=args.ricosca_data,
        widget_data=args.widget_data,
        screencap_data=args.screencap_data,
        guienv_data=args.guienv_data,
        guiact_data=args.guiact_data,
        guiact_g_data=args.guiact_g_data,
        guichat_data=args.guichat_data,
        guiworld_data=args.guiworld_data,
        guiexp_data=args.guiexp_data,
        guiexpweb_data=args.guiexpweb_data,
        act2cap_data=args.act2cap_data,
        omniact_data=args.omniact_data,
        omniact_nav_data=args.omniact_nav_data,
        osatlas_data=args.osatlas_data,
        amex_data=args.amex_data,
        amexcap_data=args.amexcap_data,
        xlam_data=args.xlam_data,
        llava_data=args.llava_data,
        inference=False,
        num_turn=args.num_turn,
        text2point=args.text2point,
        text2bbox=args.text2bbox,
        point2text=args.point2text,
        bbox2text=args.bbox2text,
        shuffle_image_token=args.shuffle_image_token,
        prob_plan=args.prob_plan,
        prob_cap=args.prob_cap,
        prob_res=args.prob_res,
        prob_think=args.prob_think,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        num_frames=args.num_frames,
        max_frames=args.max_frames,
        frame_sampling=args.frame_sampling,
        num_history=args.num_history,
        interleaved_history=args.interleaved_history,
        draw_history=args.draw_history,
        uniform_sample=args.uniform_sample,
        random_sample=args.random_sample,
        record_sample=args.record_sample,
        decay_factor=args.decay_factor,
        merge_patch=args.merge_patch,
        merge_threshold=args.merge_threshold,
        merge_inference=args.merge_inference,
        merge_random=args.merge_random,
        xy_int=args.xy_int,
        uniform_prompt=args.uniform_prompt,
        skip_readme_train=args.skip_readme_train,
        skip_readme_test=args.skip_readme_test,
    )

    val_dataset = HybridDataset(
        args.dataset_dir,
        processor,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * args.world_size,
        precision=args.precision,
        dataset=args.val_dataset,
        sample_rate=[float(x) for x in args.val_sample_rates.split(",")],
        # seeclick_data=args.seeclick_data,
        guiact_data=args.val_guiact_data,
        aitw_data=args.val_aitw_data,
        aitz_data=args.val_aitz_data,
        mind2web_data=args.val_mind2web_data,
        odyssey_data=args.val_odyssey_data,
        guiworld_data=args.val_guiworld_data,
        screenspot_data=args.val_screenspot_data,
        omniact_nav_data=args.val_omniact_nav_data,
        inference=True,
        prob_plan=args.prob_plan,
        prob_cap=args.prob_cap,
        prob_res=args.prob_res,
        prob_think=args.prob_think,
        num_frames=args.num_frames,
        max_frames=args.max_frames,
        frame_sampling=args.frame_sampling,
        num_history=args.num_history,
        interleaved_history=args.interleaved_history,
        draw_history=args.draw_history,
        decay_factor=args.decay_factor,
        merge_patch=args.merge_patch,
        merge_threshold=args.merge_threshold,
        merge_inference=args.merge_inference,
        merge_random=args.merge_random,
        xy_int=args.xy_int,
        uniform_prompt=args.uniform_prompt,
        skip_readme_train=args.skip_readme_train,
        skip_readme_test=args.skip_readme_test,
    )

    if args.val_dataset == "mind2web":
        validate = validate_mind2web
    elif args.val_dataset == "screenspot":
        validate = validate_screenspot
    elif args.val_dataset == "aitw":
        validate = validate_aitw
    elif args.val_dataset == "aitz":
        validate = validate_aitz
    elif args.val_dataset == "odyssey":
        validate = validate_odyssey
    elif args.val_dataset == "guiworld":
        validate = validate_guiworld
    elif args.val_dataset == "omniact_nav":
        validate = validate_omniact_nav
    else:
        validate = validate_default

    if not args.random_sample:
        args.steps_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)
        # args.steps_per_epoch = len(train_loader)

    # Build deepspeed config and initialize deepspeed
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "warmup_type": args.warmup_type,
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        }
    }

    config_url = f'ds_configs/{args.ds_zero}.json'
    with open(config_url, 'r') as file:
        ds_json = json.load(file)
    ds_config.update(ds_json)

    if lora_r > 0:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=list_of_params_to_optimize,
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                processor=processor
            ),
            config=ds_config,
        )
    # full tunning
    elif lora_r == 0 and not args.eval_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=list_of_params_to_optimize,
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                processor=processor
            ),
            config=ds_config,
        )
    elif lora_r == 0 and args.eval_only:
        for param in model.parameters():
            param.requires_grad = False 
        model_engine = model
    else:
        raise ValueError("Invalid setting")

    # Resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        if args.global_rank == 0:
            print(
                "resume training from {}, start from epoch {}".format(
                    args.resume, args.start_epoch
                )
            )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                processor=processor
            ),
        )
    else:
        val_loader = None

    if args.eval_only:
        local_rank = args.local_rank
        model_engine = model_engine.to(f'cuda:{local_rank}')
        validate(val_loader, model_engine, processor, 0, 0, writer, args)
        exit()

    train_iter = iter(train_loader)
    best_score = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter, global_step = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False and val_loader is not None:
            score = validate(val_loader, model_engine, processor, epoch, global_step, writer, args)
            is_best = score > best_score
            best_score = max(score, best_score)
        else:
            is_best = True
            best_score = 0

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.global_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        save_dir,
                        "meta_log_epo{:.0f}_score{:.2f}.pth".format(
                            epoch, best_score
                        ),
                    ),
                )
            torch.distributed.barrier()
            try:
                model_engine.save_checkpoint(save_dir)
            except Exception as e:
                print("Failed to save checkpoint (): ", e)
    
    if args.global_rank == 0:
        if not args.debug:
            wandb.finish()
            writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])