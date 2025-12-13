import math

import matplotlib.pyplot as plt
import torch
import torchvision as TV

from torch.utils.data import DataLoader
from tqdm import tqdm

from .args import Args
from .bucket import AspectRatioPartition, RandomBucketSampler
from .dataset import DanbooruDataset
from .utils import collate_fn


def train(args: Args) -> None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    partition = AspectRatioPartition(
        buckets=args.dataset.buckets.partitions,
        asp_dis=args.dataset.buckets.asp_dis,
        max_res=args.dataset.buckets.max_res,
        min_res=args.dataset.buckets.min_res,
    )
    dataset = DanbooruDataset(
        path=args.dataset.path,
        partition=partition,
    )
    sampler = RandomBucketSampler(
        sampling=args.dataset.buckets.sampling,
        num_samples=args.dataset.num_samples,
        batch_size=args.train.batch_size,
        data=dataset.data,
        seed=args.seed,
    )
    loader: DataLoader = DataLoader(
        num_workers=args.dataset.num_workers,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        dataset=dataset,
    )

    for batch in tqdm(loader):
        for i, caption in enumerate(batch["captions"]):
            print(f"{i}: {caption}")
        plt.imshow(TV.utils.make_grid(batch["images"].cpu(), nrow=math.floor(math.sqrt(batch["images"].size(0)))).permute((1, 2, 0)))
        plt.show()
