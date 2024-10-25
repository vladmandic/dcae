import os
import sys
import cv2
from rich import print as rprint
import torch


def get_image(fn):
    if not os.path.isfile(fn):
        rprint(f'file not found: {fn}')
        sys.exit(1)
    image = cv2.imread(fn)
    if image is None:
        rprint(f'image cannot be opened: {fn}')
        sys.exit(1)
    w, h, _ = image.shape
    if w < 256 or h < 256:
        rprint(f'image too small: {w,h}')
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_tensor(image, scale, device, dtype):
    w, h, _ = image.shape
    image = cv2.resize(image, (scale * (h // scale), scale * (w // scale)))
    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
    tensor = tensor.to(device, dtype)
    return tensor


def mem(reset=True):
    peak = torch.cuda.memory_stats()['allocated_bytes.all.peak'] // 1024 // 1024
    if reset:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return peak
