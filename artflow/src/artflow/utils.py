from typing import Any, Dict, List, cast

import torch
import torchvision as TV
import torchvision.transforms.v2.functional as TF

from torch import Tensor


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images: List[Tensor] = []
    captions: List[str] = []
    scores: List[float] = []

    for item in batch:
        captions.append(item["caption"])
        scores.append(item["score"])
        images.append(item["image"])

    images: List[Tensor] = cast(List[Tensor], TV.io.decode_jpeg(images, device=f"cuda:{torch.cuda.current_device()}"))
    images: List[Tensor] = [TF.to_dtype(TF.resize(image, batch[0]["dim"]), scale=True) for image in images]

    return {
        "scores": torch.tensor(scores),
        "images": torch.stack(images),
        "captions": captions,
    }
