import math
import random

from collections.abc import Mapping
from typing import Any, Dict, List

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from matplotlib.figure import Figure
from msgspec import Struct
from msgspec.structs import asdict
from torch import Tensor


def flatten(d: Mapping, key: str = "", sep: str = ".") -> dict:
    return {nk: vv for k, v in d.items() for nk, vv in (flatten(v, f"{key}{sep}{k}" if key else k, sep=sep).items() if isinstance(v, Mapping) else [(f"{key}{sep}{k}" if key else k, v)])}


def normalize(o: Struct) -> Dict[str, Any]:
    return {k: normalize(v) for k, v in asdict(o).items()} if isinstance(o, Struct) else o


def makegrid(x: Tensor, dpi: int = 100) -> Figure:
    # x: (B, C, H, W)
    x = einops.rearrange(x, "b c h w -> b h w c")
    B, H, W, _ = x.shape

    M = math.ceil(math.sqrt(B))
    N = math.ceil(B / M)

    # Native-size figure in inches
    figsize = (M * W / dpi, N * H / dpi)

    F, A = plt.subplots(
        N,
        M,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    A = A.flatten()

    for i, a in enumerate(A):
        if i < B:
            a.imshow(x[i])
        a.axis("off")

    # Prevent matplotlib from resizing the figure
    F.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return F


def read_prompts(path: str) -> List[str]:
    with open(path, "r") as promptfile:
        lines: List[str] = promptfile.readlines()
        lines: List[str] = [line.strip().split("||")[1] for line in lines]
        return lines


def num_params(model: nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 1e9:
        return f"{total_params / 1e9:.2f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.2f}M"
    else:
        return str(total_params)


def shuffle(captions: List[str]) -> List[str]:
    captions_shuffled: List[str] = []

    for caption in captions:
        tags: List[str] = caption.split(" ")
        random.shuffle(tags)
        caption: str = " ".join(tags)
        captions_shuffled.append(caption)

    return captions_shuffled


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
