import os

from pathlib import Path
from typing import Any, Callable, Dict

import kornia_rs as K
import polars as pl
import torch

from numpy.typing import NDArray
from polars import DataFrame, LazyFrame
from torch import Tensor
from torch.utils.data import Dataset


class DanbooruDataset(Dataset):
    "Danbooru 2024 Dataset (https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel)"

    def __init__(self, path: str, partition: Callable[[DataFrame], DataFrame]) -> None:
        super().__init__()
        self.path: Path = Path(path, "raw")
        self.data: DataFrame = self.__makedata()
        self.data: DataFrame = partition(self.data)

    def __makedata(self) -> DataFrame:
        path_label: Path = self.path / "labels.parquet"
        path_arrow: Path = path_label.with_suffix(".arrow")

        if path_arrow.exists():
            return pl.read_ipc(path_arrow, memory_map=True)

        meta: LazyFrame = pl.scan_ndjson(self.path / "metadata_data.ndjson")
        data: LazyFrame = pl.scan_parquet(path_label)
        data = data.join(meta, on="id")
        data = data.with_columns(path=pl.format("{}{}{}{}", pl.col("tar").str.replace(".tar", ""), pl.lit(os.path.sep), pl.col("id"), pl.lit(".jpg")))
        data = data.sort(by="id")
        data = data.select(pl.col("path"), pl.col("image_height").alias("h"), pl.col("image_width").alias("w"), pl.col("tag_string").alias("caption"), pl.col("score"))
        data = data.with_row_index("index")
        data.sink_ipc(path_arrow)
        return self.__makedata()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row: Dict[str, Any] = self.data.row(index, named=True)
        image: NDArray = K.read_image_jpeg(str(self.path / f"{row['path']}"), "rgb")  # type: ignore
        image: NDArray = K.resize(image, (row["bh"], row["bw"]), interpolation="nearest")  # type: ignore
        image: Tensor = torch.from_dlpack(image).permute((2, 0, 1))  # type: ignore
        return {"caption": row["caption"], "image": image, "score": row["score"]}

    def __len__(self) -> int:
        return len(self.data)
