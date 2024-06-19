import os
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from model import Generator

parser = ArgumentParser("Real Time Image Super Resolution")
parser.add_argument("--image_dir", default=None, required=True, type=str)
parser.add_argument("--output_dir", default=None, required=True, type=str)


def main():
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    config = OmegaConf.load("configs/config.yaml")
    model_path = "models/model.pt"
    model = Generator(config.generator)
    weights = torch.load(model_path, map_location="cpu")
    new_weights = {}
    for k, v in weights.items():
        new_weights[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(new_weights)
    model.to(device)
    model.eval()

    image_paths = sorted(
        [
            x
            for x in os.listdir(args.image_dir)
            if x.lower().endswith(".png")
            or x.lower().endswith(".jpg")
            or x.lower().endswith("jpeg")
        ]
    )
    print(f"Found {len(image_paths)} to super resolve, starting...")
    for image_path in tqdm(image_paths, total=len(image_paths), desc="Super Resolving"):
        lr_image = Image.open(os.path.join(args.image_dir, image_path)).convert("RGB")
        lr_image = np.array(lr_image)
        lr_image = (torch.from_numpy(lr_image) / 127.5) - 1.0
        lr_image = lr_image.permute(2, 0, 1).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            sr_image = model(lr_image).cpu()
            sr_image = (sr_image + 1.0) / 2.0
            sr_image = sr_image.permute(0, 2, 3, 1).squeeze()
            sr_image = (sr_image * 255).numpy().astype(np.uint8)
        Image.fromarray(sr_image).save(os.path.join(args.output_dir, os.path.basename(image_path)))


if __name__ == "__main__":
    main()

