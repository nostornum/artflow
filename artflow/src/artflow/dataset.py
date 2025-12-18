import os

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import PIL
import PIL.Image
import polars as pl

from PIL.Image import Image
from polars import DataFrame, LazyFrame
from torch import Tensor
from torch.utils.data import Dataset


class DanbooruDataset(Dataset):
    "Danbooru 2024 Dataset (https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel)"

    def __init__(self, path: str, partition: Callable[[DataFrame], DataFrame], transform: Callable[[Image, Tuple[int, int]], Tensor]) -> None:
        super().__init__()
        self.path: Path = Path(path, "raw")
        self.data: DataFrame = self.__makedata()
        self.data: DataFrame = partition(self.data)
        self.transform: Callable[[Image, Tuple[int, int]], Tensor] = transform

    def __makedata(self) -> DataFrame:
        path_label: Path = self.path / "labels.parquet"
        path_arrow: Path = path_label.with_suffix(".arrow")

        if path_arrow.exists():
            return pl.read_ipc(path_arrow, memory_map=True)

        meta: LazyFrame = pl.scan_ndjson(self.path / "metadata_data.ndjson")
        data: LazyFrame = pl.scan_parquet(path_label)
        data: LazyFrame = data.join(meta, on="id")
        data: LazyFrame = data.with_columns(path=pl.format("{}{}{}{}", pl.col("tar").str.replace(".tar", ""), pl.lit(os.path.sep), pl.col("id"), pl.lit(".jpg")))
        data: LazyFrame = data.sort(by="id")
        data: LazyFrame = data.select(pl.col("path"), pl.col("image_height").alias("h"), pl.col("image_width").alias("w"), pl.col("tag_string").alias("caption"), pl.col("score"))
        data: LazyFrame = data.with_row_index("index")
        data.sink_ipc(path_arrow)
        return self.__makedata()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row: Dict[str, Any] = self.data.row(index, named=True)
        img_p: str = str(self.path / f"{row['path']}")
        img_r: Image = PIL.Image.open(img_p).convert("RGB")
        img_b: Tensor = self.transform(img_r, (row["bh"], row["bw"]))
        return {"caption": row["caption"], "image": img_b, "score": row["score"]}

    def __len__(self) -> int:
        return len(self.data)
