accelerate launch artflow/src/artflow                 \
 --seed 18                                            \
 --track.ckpt-every 500                               \
 --track.eval-every 500                               \
 --track.loss-every 10                                \
 --track.path logs                                    \
 --dataset.path data/danbooru                         \
 --dataset.num-workers 4                              \
 --dataset.buckets.partitions 320 224                 \
 --dataset.buckets.sampling F                         \
 --dataset.buckets.asp_dis 0.1                        \
 --dataset.buckets.max_res inf                        \
 --dataset.buckets.min_res 128                        \
 --sample.promptfile prompts.txt                      \
 --train.ckpt-resume checkpoint_78500.pth             \
 --train.ckpt-folder ckpt                             \
 --train.batch-size 42                                \
 --train.steps 1000000


uv run python -m artflow                              \
 --track.ckpt-every 1000                              \
 --track.eval-every 200                               \
 --track.run-id 7dysxkig                              \
 --track.loss-every 10                                \
 --track.path logs                                    \
 --dataset.path data/danbooru                         \
 --dataset.num-workers 16                             \
 --dataset.buckets.partitions 256 256 320 224 224 320 \
 --dataset.buckets.asp_dis 0.1                        \
 --dataset.buckets.max_res 8192                       \
 --dataset.buckets.min_res 128                        \
 --sample.promptfile prompts.txt                      \
 --train.ckpt-folder ckpt                             \
 --train.batch-size 32                                \
 --train.steps 10000
