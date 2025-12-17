from typing import Annotated, List, Literal, Tuple

import tyro

from msgspec import Struct


class BucketArgs(Struct, kw_only=True):
    """Arguments for partitioning the dataset inbuckets"""

    # Partition the data in buckets
    partitions: List[Tuple[int, int]]

    # Bucket sampling method
    sampling: Literal["F", "U"] = "F"

    # L1 aspect ratio distance threshold
    asp_dis: float = 0.1

    # Minimum size threshold in pixels
    min_res: float = 0

    # Maximum size threshold in pixels
    max_res: float = float("inf")


class DatasetArgs(Struct, kw_only=True):
    """Arguments for selecting a dataset"""

    # Path to the dataset on disk
    path: str

    # Number of workers in parallel
    num_workers: int

    # Partition the data in buckets
    buckets: BucketArgs


class LLMArgs(Struct, kw_only=True):
    """Arguments for the LLM model"""

    # Checkpoint from Huggingface
    ckpt: str = "google/gemma-3-270m"

    # Maximum text sequence length
    max_length: int = 128


class VAEArgs(Struct, kw_only=True):
    """Arguments for the VAE model"""

    # Checkpoint from Huggingface
    ckpt: str = "black-forest-labs/FLUX.2-dev"

    # Latent scaling factor
    scaling_factor: float = 1 / 1.6601


class ModelArgs(Struct, kw_only=True):
    """Arguments for the diffusion model"""

    # Input patch size
    patch_size: int = 2

    # Number of attention heads
    num_heads: int = 16

    # Number of text embedder layers
    num_embedder_layers: int = 2

    # Number of encoder layers
    num_encoder_layers: int = 24

    # Number of decoder layers
    num_decoder_layers: int = 3

    # Channel dimension of input
    dim_input: int = 32

    # Channel dimension of timestep
    dim_timestep: int = 256

    # Channel dimension of text embedder
    dim_txt_emb: int = 640

    # Channel dimension of encoder
    dim_hidden_enc: int = 768

    # Channel dimension of decoder
    dim_hidden_dec: int = 128


class OptimizerArgs(Struct, kw_only=True):
    """Arguments for Muon optimizer"""

    # Learning rate
    lr: float = 1e-3

    # Number of warmup steps
    warmup: int = 1000


class TrainArgs(Struct, kw_only=True):
    """Arguments for training procedure"""

    # Number of samples in one batch
    batch_size: int

    # Number of training steps
    steps: int = 1000

    # Probability to drop text condition
    cond_drop: float = 0.1

    # Gradient accumulation steps
    grad_accumulation_steps: int = 1

    # Perform gradient clipping
    clip_grad: float = 1.0

    # Path to the checkpoint directory
    ckpt_folder: str

    # Path to the resumed checkpoint
    ckpt_resume: str | None = None

    # Configure optimizer
    optim: OptimizerArgs

    # Weight for CFM loss
    w_cfm: float = 0.05


class TrackArgs(Struct, kw_only=True):
    """Arguments for tracking experiments"""

    # Location to store logs
    path: str

    # Name of the project
    project: str = "artflow"

    # Id of the run
    run_id: str | None = None

    # Sampling interval
    eval_every: int = 1

    # Checkpoint interval
    ckpt_every: int = 1

    # Metrics interval
    loss_every: int = 1


class SampleArgs(Struct, kw_only=True):
    """Arguments for generating new samples"""

    # Weight for classifier-free guidance
    w: float = 3.0

    # Batch size for sampling
    batch: int = 16

    # Number of sampling steps
    timesteps: int = 75

    # Path to the prompt file
    promptfile: str


class Args(Struct, kw_only=True):
    """Arguments for training a diffusion model from scratch"""

    # Experiment seed
    seed: int = 42

    # Experiment tracking arguments
    track: TrackArgs

    # Read dataset from disk
    dataset: DatasetArgs

    # Training settings
    train: TrainArgs

    # Sample configuration
    sample: SampleArgs

    # Model configuration
    model: ModelArgs

    # VAE cofniguration
    vae: VAEArgs

    # LLM configuration
    llm: LLMArgs


Args = Annotated[Args, tyro.conf.arg(name="")]  # type: ignore
