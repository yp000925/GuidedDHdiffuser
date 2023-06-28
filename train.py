from functools import partial
import os
import argparse
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    create_model,create_gaussian_diffusion,
    dict_to_args,
    add_dict_to_argparser)
from improved_diffusion.train_util import TrainLoop
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config',default='configs/model_config.yaml', type=str)
    parser.add_argument('--diffusion_config',default='configs/diffusion_eps_config.yaml', type=str)
    parser.add_argument('--train_config', default='configs/train_config.yaml',type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./trained_models')
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(args=[])

    # logger
    logger.configure(dir='./logs_eps',log_suffix='cell')

    # Device setting
    # device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    # logger.info(f"Device set to {device_str}.")
    # device = torch.device(device_str)
    dist_util.setup_dist()

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    train_config = load_yaml(args.train_config)
    total_config = {**model_config, **diffusion_config, **train_config}

    args = dict_to_args(total_config)
    logger.log("creating model and diffusion...")
    # Load model
    model = create_model(**model_config)
    model.to(dist_util.dev())


    diffusion = create_gaussian_diffusion(**diffusion_config)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        eval_interval=[500,500]
    ).run_loop()