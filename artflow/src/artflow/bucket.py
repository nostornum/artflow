from dataclasses import KW_ONLY, dataclass
from typing import Any, Dict, Iterator, List, Literal, Tuple

import PIL
import PIL.Image
import polars as pl
import torch
import torchvision.transforms.v2 as T

from PIL.Image import Image
from polars import DataFrame, LazyFrame
from torch import Generator, Tensor
from torch.utils.data import Sampler


@dataclass
class AspectRatioPartition:
    buckets: List[Tuple[int, int]]
    asp_dis: float
    min_res: float
    max_res: float

    def __call__(self, data: DataFrame) -> DataFrame:
        buckets: LazyFrame = (
            LazyFrame(
                {
                    "bh": [h for h, _ in self.buckets],
                    "bw": [w for _, w in self.buckets],
                }
            )
            .with_columns(ar=pl.col.bw.truediv(pl.col.bh))
            .with_row_index("bucket")
            .sort(by="ar")
        )

        query: LazyFrame = data.lazy()
        query: LazyFrame = query.filter(pl.min_horizontal("h", "w") >= self.min_res)
        query: LazyFrame = query.filter(pl.max_horizontal("h", "w") <= self.max_res)
        query: LazyFrame = query.with_columns(ar=pl.col.w.truediv(pl.col.h))
        query: LazyFrame = query.sort(by=pl.col.ar)
        query: LazyFrame = query.join_asof(buckets, on="ar", strategy="nearest", tolerance=self.asp_dis, coalesce=True)
        query: LazyFrame = query.filter(~pl.col.bucket.is_null()).drop("index", strict=False).with_row_index("index")
        return query.collect()


@dataclass
class RandomBucketSampler(Sampler[List[int]]):
    data: DataFrame
    _: KW_ONLY
    seed: int = 42
    batch_size: int = 1
    num_samples: int = 0
    sampling: Literal["U", "F"] = "F"

    def __post_init__(self) -> None:
        assert len(self.data) != 0, "input 'data' cannot be empty"
        assert self.data.get_column("bucket", default=None) is not None, "input 'data' must have 'bucket' column"
        self._rng: Generator = Generator().manual_seed(self.seed)
        self._bkt: DataFrame = self.data.lazy().filter(pl.col("bucket") != -1).group_by("bucket").agg(pl.len().alias("bucket_size"), pl.col("index").alias("bucket_content")).sort(by="bucket_size", descending=True).collect()
        self._b_idx: Dict[int, Tensor] = {bucket: torch.tensor(content) for bucket, content in self._bkt.select(["bucket", "bucket_content"]).iter_rows()}
        self._w_frq: Tensor = torch.tensor(pl.Series(self._bkt.select("bucket_size")).to_list(), dtype=torch.float32)
        self._w_uni: Tensor = torch.ones(self._w_frq.size(0))
        self._w_bkt: Tensor = self._w_frq if self.sampling == "F" else self._w_uni
        self._w_bkt: Tensor = self._w_bkt / self._w_bkt.sum()
        print(self._bkt)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_samples):
            i_bkt: Tensor = torch.multinomial(self._w_bkt, num_samples=1, generator=self._rng)
            b_idx: Tensor = self._b_idx[int(i_bkt)]
            s_idx: Tensor = torch.randint(0, b_idx.nelement(), [self.batch_size], generator=self._rng)
            s_idx: Tensor = b_idx[s_idx]
            yield s_idx.tolist()

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> Dict[str, Tensor]:
        return {"batch_sampler": self._rng.get_state()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._rng.set_state(state_dict["batch_sampler"])


@dataclass
class BucketTransform:
    def __post_init__(self) -> None:
        self._transforms: Dict[Tuple[int, int], T.Transform] = {}

    def __call__(self, x: Image, size: Tuple[int, int]) -> Tensor:
        return self._transforms[size](x) if size in self._transforms else self._transforms.setdefault(size, BucketTransform.create(size))(x)

    @staticmethod
    def create(size: Tuple[int, int]) -> T.Transform:
        return T.Compose(
            [
                T.Lambda(lambda img: BucketTransform.resize_to_cover(img, size[0], size[1])),
                T.RandomCrop(size),
                T.ToImage(),
                T.ToDtype(dtype=torch.float32, scale=True),
            ]
        )

    @staticmethod
    def resize_to_cover(img: Image, target_h: int, target_w: int) -> Image:
        w, h = img.size
        scale: float = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=PIL.Image.Resampling.LANCZOS)
