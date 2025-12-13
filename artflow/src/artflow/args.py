from typing import Annotated, List, Literal, Tuple

import tyro

from msgspec import Struct


class BucketArgs(Struct):
    """Arguments for partitioning the dataset inbuckets"""

    # Partition the data in buckets
    partitions: List[Tuple[int, int]]

    # Bucket sampling method
    sampling: Literal["F", "U"] = "U"

    # L1 aspect ratio distance threshold
    asp_dis: float = 0.1

    # Minimum size threshold in pixels
    min_res: float = 0

    # Maximum size threshold in pixels
    max_res: float = float("inf")


class DatasetArgs(Struct):
    """Arguments for selecting a dataset"""

    # Path to the dataset on disk
    path: str

    # Number of workers in parallel
    num_workers: int

    # Number of samples per-epoch
    num_samples: int

    # Partition the data in buckets
    buckets: BucketArgs


class TextEncoder(Struct):
    pass


class VAEArgs(Struct):
    pass


class ModelArgs(Struct):
    pass


class TrainArgs(Struct):
    """Arguments for training procedure"""

    # Number of samples in one batch
    batch_size: int


class Optimizer(Struct):
    pass


class TrackArgs(Struct):
    pass


class SampleArgs(Struct):
    pass


class Args(Struct, kw_only=True):
    """Arguments for training a diffusion model from scratch"""

    # Experiment seed
    seed: int = 42

    # Read dataset from disk
    dataset: DatasetArgs

    # Parameters used for training
    train: TrainArgs


Args = Annotated[Args, tyro.conf.arg(name="")]  # type: ignore
