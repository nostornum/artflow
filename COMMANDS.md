uv run python -m artflow                              \
 --dataset.path ./data/danbooru                       \
 --dataset.num-samples 10000                          \
 --dataset.num-workers 8                              \
 --dataset.buckets.partitions 256 256 320 224 224 320 \
 --dataset.buckets.asp_dis 0.1                        \
 --dataset.buckets.max_res 4096                       \
 --dataset.buckets.min_res 128                        \
 --train.batch-size 32
