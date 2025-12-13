import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from .args import Args
from .bucket import AspectRatioPartition, RandomBucketSampler
from .dataset import DanbooruDataset


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
        dataset=dataset,
        pin_memory=True,
    )

    for batch in tqdm(loader):
        pass
