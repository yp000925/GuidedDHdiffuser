"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,create_model,create_gaussian_diffusion,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config',default='configs/model_config.yaml', type=str)
    parser.add_argument('--diffusion_config',default='configs/diffusion_config.yaml', type=str)
    parser.add_argument('--train_config', default='configs/train_config.yaml',type=str)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='logs3/model010000.pt')
    parser.add_argument('--sample_size', type=int, default=5)
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(args=[])

    dist_util.setup_dist()
    logger.configure(dir='./test',log_suffix='cell')

    logger.log("creating model and diffusion...")

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)

    model = create_model(**model_config)
    model.to(dist_util.dev())
    diffusion = create_gaussian_diffusion(**diffusion_config)

    # load model
    logger.log("Loading...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    logger.log("Successfully loaded from",args.model_path)

    # sampling
    logger.log("sampling...")

    with th.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (args.sample_size, 3, model_config['image_size'], model_config['image_size']),
            progress=True)
        def clear_color(x):
            if th.is_complex(x):
                x = th.abs(x)
            x = x.detach().cpu().squeeze().numpy()
            x -= np.min(x)
            x /= np.max(x)
            return np.transpose(x, (1, 2, 0))
        for i in range(args.sample_size):
            fname = f"sample_{i}.png"
            plt.imsave(os.path.join(logger.get_dir(), fname), clear_color(sample[i,:,:,:]))
        logger.log(f"created {args.sample_size} samples")



if __name__ == "__main__":
    main()
