import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, unpack
from src.flux.util import configs, load_ae, load_flow_model2

from image_datasets.dataset_cc3m import loader as cc3m_loader
from image_datasets.dataset_mimic import loader as mimic_loader
from torchvision import transforms

from clip_models.build_CLIP import load_clip_model_SigLIP
from clip_models.sampling import prepare_clip

from peft import LoraConfig, get_peft_model
from copy import deepcopy

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


OPENAI_DATASET_MEAN = 0.5   # NOTE!!! 0.5 for SigLIP
OPENAI_DATASET_STD = 0.5   # NOTE!!! 0.5 for SigLIP
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    args.clip_config.seq_t5 = 256 if is_schnell else 512   # NOTE!!!
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit = load_flow_model2(args.model_name, device="cpu")
    vae = load_ae(args.model_name, device=accelerator.device)
    clip_vis = load_clip_model_SigLIP(args.clip_config, device=accelerator.device)
    
    # set LoRA
    lora_config = LoraConfig(
        r=args.lora_config.r,
        lora_alpha=args.lora_config.lora_alpha,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2'],   # NOTE!!!
        #target_modules='all-linear',
        lora_dropout=args.lora_config.lora_dropout,
        bias=args.lora_config.bias,
    )
    clip_vis.model = get_peft_model(clip_vis.model, lora_config)
    clip_vis.model.print_trainable_parameters()


    print('loading projection params...')
    load_path_project_clip = os.path.join(args.load_dir, f"checkpoint-project-clip-{args.load_step}.bin")
    clip_vis.project_clip.load_state_dict(torch.load(load_path_project_clip, map_location=torch.device('cpu')))
    load_path_project_t5 = os.path.join(args.load_dir, f"checkpoint-project-t5-{args.load_step}.bin")
    clip_vis.project_t5.load_state_dict(torch.load(load_path_project_t5, map_location=torch.device('cpu')))
    print('loading successfully!')

    print('loading dit params...')
    load_path_dit = os.path.join(args.load_dir, f"checkpoint-dit-{args.load_step}.bin")
    dit.load_state_dict(torch.load(load_path_dit))
    print('loading successfully!')

    vae.requires_grad_(False)
    dit.requires_grad_(False)
    dit = dit.to(torch.bfloat16)
    dit.to(accelerator.device)
    clip_vis.train()
    dit.train()

    for name_, param in clip_vis.named_parameters():
        if 'project_clip' in name_ or 'project_t5' in name_:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        [p for p in clip_vis.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 根据数据集类型选择相应的数据加载器
    if hasattr(args.data_config, 'dataset_type') and args.data_config.dataset_type == 'custom':
        # 为数据加载器添加batch size参数
        data_config_with_batch = args.data_config.copy()
        data_config_with_batch['train_batch_size'] = args.train_batch_size
        train_dataloader = mimic_loader(**data_config_with_batch)
    else:
        # 默认使用CC3M数据集
        data_config_with_batch = args.data_config.copy()
        data_config_with_batch['train_batch_size'] = args.train_batch_size
        train_dataloader = cc3m_loader(**data_config_with_batch)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    
    # 获取实际数据集大小
    if hasattr(args.data_config, 'dataset_type') and args.data_config.dataset_type == 'custom':
        # 对于自定义数据集，我们需要从数据加载器获取实际大小
        # 先创建一个临时的数据加载器来获取数据集大小
        temp_data_config = args.data_config.copy()
        temp_data_config['train_batch_size'] = args.train_batch_size
        temp_dataloader = mimic_loader(**temp_data_config)
        dataset_size = len(temp_dataloader.dataset)
        logger.info(f"Dataset size: {dataset_size}")
    else:
        # 默认使用CC3M数据集大小
        dataset_size = int(3e6)
        logger.info(f"Using default CC3M dataset size: {dataset_size}")
    
    num_update_steps_per_epoch = math.ceil(dataset_size / args.train_batch_size) / args.gradient_accumulation_steps
    logger.info(f"Number of update steps per epoch: {num_update_steps_per_epoch}")
    
    # 优先使用epoch数，如果设置了epoch数，则计算对应的步数
    if hasattr(args, 'num_train_epochs') and args.num_train_epochs is not None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"Using epoch-based training: {args.num_train_epochs} epochs = {args.max_train_steps} steps")
    elif args.max_train_steps is None:
        # 如果没有设置epoch数也没有设置步数，使用默认epoch数
        if not hasattr(args, 'num_train_epochs') or args.num_train_epochs is None:
            args.num_train_epochs = 10  # 默认10个epoch
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"Using default epoch-based training: {args.num_train_epochs} epochs = {args.max_train_steps} steps")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    clip_vis, optimizer, _, lr_scheduler = accelerator.prepare(
        clip_vis, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    if overrode_max_train_steps:
        # 基于epoch数计算步数，无需额外处理
        pass
    else:
        # 当用户明确设置了max_train_steps时，计算需要的epoch数
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        logger.info(f"Calculated number of epochs needed: {args.num_train_epochs}")
        logger.info(f"Total training steps will be: {args.max_train_steps}")

    if accelerator.is_main_process:
        # 初始化wandb跟踪器
        if args.report_to == "wandb":
            # 直接初始化wandb
            if is_wandb_available():
                wandb.init(
                    project=args.tracker_project_name,
                    name=f"GenHancer_SigLIP_{args.clip_config.clip_image_size}_stage2",
                    tags=["GenHancer", "SigLIP", "stage2", f"img_size_{args.clip_config.clip_image_size}"],
                    config={
                        "model_name": args.model_name,
                        "clip_image_size": args.clip_config.clip_image_size,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.train_batch_size,
                        "max_train_steps": args.max_train_steps,
                        "mixed_precision": args.mixed_precision,
                        "dataset_type": getattr(args.data_config, 'dataset_type', 'cc3m'),
                        "output_dir": args.output_dir,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "lora_config": args.lora_config,
                    }
                )
                logger.info(f"Initialized wandb with project: {args.tracker_project_name}")
                logger.info(f"Experiment name: GenHancer_SigLIP_{args.clip_config.clip_image_size}_stage2")
            else:
                logger.warning("wandb is not available, falling back to tensorboard")
                accelerator.init_trackers(args.tracker_project_name, {"test": None})
        else:
            accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, int(args.num_train_epochs)):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(clip_vis):
                original_img, prompts = batch['image'], batch['text']
                original_img = original_img.to(accelerator.device)
                
                with torch.no_grad():
                    x_1 = vae.encode(NORMALIZE_VAE(original_img).to(torch.float32))

                inp = prepare_clip(clip=clip_vis, original_img=NORMALIZE_CLIP(original_img).to(weight_dtype), img=x_1.to(weight_dtype))
                x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                bs = original_img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=accelerator.device) * args.scale_factor)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                # Predict the noise residual and compute loss
                model_pred = dit(img=x_t.to(weight_dtype),
                                 img_ids=inp['img_ids'].to(weight_dtype),
                                 txt=inp['txt'].to(weight_dtype),
                                 txt_ids=inp['txt_ids'].to(weight_dtype),
                                 y=inp['vec'].to(weight_dtype),
                                 timesteps=t.to(weight_dtype),
                                 guidance=guidance_vec.to(weight_dtype),)

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(clip_vis.parameters(), args.max_grad_norm)   # NOTE!!!
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # 记录训练指标
                if args.report_to == "wandb" and accelerator.is_main_process and is_wandb_available():
                    # 直接使用wandb记录
                    wandb.log({
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/loss": train_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "train/mixed_precision": args.mixed_precision,
                    }, step=global_step)
                else:
                    # 使用accelerator.log记录（tensorboard等）
                    accelerator.log({
                        "train_loss": train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                    }, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or global_step in [50, 100, 200, 300, 500, 1000, 2000, 3000]:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"siglip-so400m-patch14-{args.clip_config.clip_image_size}-{global_step}")
                        unwrapped_clip_vis = accelerator.unwrap_model(clip_vis)
                        save_model = deepcopy(unwrapped_clip_vis.model).merge_and_unload()
                        save_model.save_pretrained(save_path, safe_serialization=False)
                        logger.info(f"Saved ckpts to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= int(args.max_train_steps):
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"siglip-so400m-patch14-{args.clip_config.clip_image_size}-{global_step}")
                    unwrapped_clip_vis = accelerator.unwrap_model(clip_vis)
                    save_model = deepcopy(unwrapped_clip_vis.model).merge_and_unload()
                    save_model.save_pretrained(save_path, safe_serialization=False)
                break

    accelerator.wait_for_everyone()
    
    # 结束wandb记录
    if args.report_to == "wandb" and accelerator.is_main_process and is_wandb_available():
        wandb.finish()
        logger.info("Wandb logging finished")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
