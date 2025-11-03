"""Save the ground-truth label maps of the test set as colour images (for qualitative figures).

Run from the repository root:
    PYTHONPATH=. python tools/visualize_ground_truth.py --config local_configs.NYUDepthv2.DFormerv2_S_text
"""
import argparse
import os
from importlib import import_module
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Save colour-coded ground-truth maps")
    parser.add_argument("--config", default="local_configs.NYUDepthv2.DFormerv2_S_text", help="config module")
    parser.add_argument("--save_path", default="vis_gt", help="output directory")
    parser.add_argument("--palette_path", default="utils/nyucmap.npy", help="colour palette (.npy)")
    args = parser.parse_args()

    config = getattr(import_module(args.config.replace("/", ".").replace(".py", "")), "C")
    config.pad = False  # no SUN RGB-D padding, as in infer.py
    os.makedirs(args.save_path, exist_ok=True)

    # Same test images as eval.py / infer.py, loaded without distributed training
    val_loader, _ = get_val_loader(SimpleNamespace(distributed=False), RGBXDataset, config, val_batch_size=1)
    palette = np.load(args.palette_path)

    for minibatch in tqdm(val_loader, desc="Ground truth"):
        # Labels are shifted by -1 when loaded, so unlabelled pixels (0) wrap around to 255
        gt = minibatch["label"][0].cpu().numpy().astype(np.uint8)
        name = os.path.splitext(os.path.basename(minibatch["fn"][0]))[0]
        plt.imsave(os.path.join(args.save_path, f"{name}_gt_color.png"), palette[gt])

    logger.info(f"Saved {len(val_loader.dataset)} ground-truth maps to {args.save_path}")


if __name__ == "__main__":
    main()
