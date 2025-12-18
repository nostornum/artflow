import math
import os

from dataclasses import KW_ONLY, dataclass
from math import prod
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from msgspec.structs import asdict
from timm.optim.muon import Muon
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, GemmaTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup as CosineScheduler

import wandb

from .args import Args
from .bucket import AspectRatioPartition, BucketTransform, RandomBucketSampler
from .dataset import DanbooruDataset
from .model import DeCo
from .utils import flatten, makegrid, normalize, num_params, read_prompts, set_seed, shuffle


@dataclass
class Trainer:
    _: KW_ONLY
    accelerator: Accelerator
    vae: AutoencoderKLFlux2
    llm: Gemma3ForCausalLM
    tok: GemmaTokenizer
    sch: LRScheduler
    opt: Optimizer
    model: DeCo
    args: Args

    def __post_init__(self) -> None:
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.llm.requires_grad_(False)
        self.llm.eval()
        self.model.requires_grad_(True)
        self.model.train()

    def text_encode(self, caption: List[str]) -> Tensor:
        text_enc: BatchEncoding = self.tok(text=caption, max_length=self.args.llm.max_length, truncation=True, padding="max_length", padding_side="right", return_tensors="pt")
        text_ids: Dict[str, Tensor] = {k: v.to(self.accelerator.device) for k, v in text_enc.items()}["input_ids"]
        text_emb = self.llm(input_ids=text_ids, output_hidden_states=True)["hidden_states"]
        return text_emb[-1].float()

    def imgs_encode(self, image: Tensor) -> Tensor:
        latents = [cast(Any, self.vae.encode(x)).latent_dist.sample() for x in image.split(self.args.vae.batch)]
        return torch.cat(latents, dim=0) * self.args.vae.scaling_factor

    def imgs_decode(self, latent: Tensor) -> Tensor:
        latent = latent / self.args.vae.scaling_factor
        images = [cast(Any, self.vae.decode(cast(FloatTensor, x))).sample for x in latent.split(self.args.vae.batch)]
        return torch.cat(images, dim=0)

    def shift_time(self, x: Tensor, t: Tensor) -> Tensor:
        m: int = prod(x.shape[1:])
        n: int = 4096
        s: float = math.sqrt(n / m)
        return s * t / (1 + (s - 1) * t)

    def step(self, batch: Dict[str, Any], c_null: Tensor) -> Dict[str, Tensor]:
        self.opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Encode inputs
            B: int = batch["image"].size(0)
            x: Tensor = self.imgs_encode(batch["image"] * 2 - 1)
            c: Tensor = self.text_encode(shuffle(batch["caption"]))

            # Drop condition for CFG
            mask: Tensor = torch.rand(B, device=x.device) < self.args.train.cond_drop
            c[mask] = c_null

            # Sample timestep
            e: Tensor = torch.randn_like(x)
            t: Tensor = torch.rand([B, 1, 1, 1], device=self.accelerator.device)
            t: Tensor = self.shift_time(x, t)

            # Compute target
            x_g: Tensor = x
            x_t: Tensor = t * x + (1 - t) * e
            v_t: Tensor = (x_g - x_t) / (1 - t).clamp_min(0.05)

        with self.accelerator.accumulate(self.model):
            # Perform x-pred and compute v-loss
            x_p: Tensor = self.model.forward(x=x_t, t=t, c=c)
            v_p: Tensor = (x_p - x_t) / (1 - t).clamp_min(0.05)
            v_m: Tensor = F.mse_loss(v_p, v_t)

            # Compute v-cfm
            v_c: Tensor = -F.mse_loss(v_p, v_t.roll(1, 0))

            # Total Loss
            loss: Tensor = v_m + self.args.train.w_cfm * v_c

            # Backprop
            self.accelerator.backward(loss=loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.train.clip_grad)
            self.opt.step()
            self.sch.step()

        return {
            "loss": loss.detach(),
            "cfm": v_c.detach(),
            "mse": v_m.detach(),
            "emb": c.detach(),
            "eps": e.detach(),
        }

    @torch.no_grad()
    def sample(self, eps: Tensor, c_cond: Tensor, c_null: Tensor) -> Tensor:
        t: Tensor = torch.linspace(0, 1, steps=self.args.sample.timesteps + 1, device=self.accelerator.device)
        t: Tensor = self.shift_time(eps, t)
        w: float = self.args.sample.w
        B: int = eps.size(0)
        x: Tensor = eps

        for t_curr, t_next in zip(t, t[1:]):
            t_curr: Tensor = t_curr.expand(B, 1, 1, 1)
            x_con: Tensor = self.model.forward(x, t_curr, c_cond)
            x_unc: Tensor = self.model.forward(x, t_curr, c_null)
            v_con: Tensor = (x_con - x) / (1 - t_curr).clamp_min(0.05)
            v_unc: Tensor = (x_unc - x) / (1 - t_curr).clamp_min(0.05)
            v_cfg: Tensor = v_unc + w * (v_con - v_unc)
            x: Tensor = x + (t_next - t_curr) * v_cfg

        return (self.imgs_decode(x) * 0.5 + 0.5).clip(0, 1)

    @staticmethod
    def save(ckpt_dir: str, dataloader: DataLoader, accelerator: Accelerator, model: Module, optimizer: Optimizer, scheduler: LRScheduler, step: int) -> None:
        state_dict: Dict[str, Any] = {}
        state_dict["model"] = accelerator.unwrap_model(model).state_dict()
        state_dict["optimizer"] = optimizer.state_dict()
        state_dict["scheduler"] = scheduler.state_dict()
        state_dict["step"] = step
        accelerator.save(state_dict, os.path.join(ckpt_dir, f"checkpoint_{step}.pth"))

    @staticmethod
    def load(ckpt_path: str, dataloader: DataLoader, accelerator: Accelerator, model: Module, optimizer: Optimizer, scheduler: LRScheduler) -> int:
        state_dict: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        return state_dict["step"]


def train(args: Args) -> None:
    wandb.login()
    set_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    # Load dataset
    partition = AspectRatioPartition(
        buckets=args.dataset.buckets.partitions,
        asp_dis=args.dataset.buckets.asp_dis,
        max_res=args.dataset.buckets.max_res,
        min_res=args.dataset.buckets.min_res,
    )
    dataset = DanbooruDataset(
        transform=BucketTransform(),
        path=args.dataset.path,
        partition=partition,
    )
    sampler = RandomBucketSampler(
        sampling=args.dataset.buckets.sampling,
        batch_size=args.train.batch_size,
        num_samples=args.train.steps,
        data=dataset.data,
        seed=args.seed,
    )
    loader: DataLoader = DataLoader(
        num_workers=args.dataset.num_workers,
        batch_sampler=sampler,
        dataset=dataset,
        pin_memory=True,
    )

    # Load models
    vae: AutoencoderKLFlux2 = AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name_or_path=args.vae.ckpt,
        token=os.environ["HF_TOKEN"],
        subfolder="vae",
    )
    llm: Gemma3ForCausalLM = Gemma3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.llm.ckpt,
        token=os.environ["HF_TOKEN"],
    )
    tok: GemmaTokenizer = GemmaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.llm.ckpt,
        token=os.environ["HF_TOKEN"],
    )
    model: DeCo = DeCo(**asdict(args.model))

    # Model parameters count
    print("Params: ", num_params(model))

    # Setup accelerator
    opt: Optimizer = Muon(model.parameters(), args.train.optim.lr)
    sch: LRScheduler = CosineScheduler(opt, args.train.optim.warmup, args.train.steps)
    accelerator: Accelerator = Accelerator(mixed_precision="bf16", log_with="wandb", gradient_accumulation_steps=args.train.grad_accumulation_steps)
    loader, llm, vae, model, opt, sch = accelerator.prepare(loader, llm, vae, model, opt, sch)

    # Setup experiment
    run_args: Dict[str, Any] = dict(
        project=args.track.project,
        id=args.track.run_id,
        dir=args.track.path,
    )

    # Unwrap models
    vae = cast(AutoencoderKLFlux2, vae.module if accelerator.num_processes != 1 else vae)
    llm = cast(Gemma3ForCausalLM, llm.module if accelerator.num_processes != 1 else llm)
    model = cast(DeCo, model)

    # Resume from checkpoint
    step: int = Trainer.load(os.path.join(args.train.ckpt_folder, args.train.ckpt_resume), loader, accelerator, model, opt, sch) if args.train.ckpt_resume else 0

    # Initialize experiment
    accelerator.init_trackers(args.track.project, config=flatten(normalize((args))), init_kwargs=run_args)

    # Initialize trainer
    trainer = Trainer(
        accelerator=accelerator,
        model=model,
        args=args,
        vae=vae,
        llm=llm,
        tok=tok,
        opt=opt,
        sch=sch,
    )

    # Use prompt file for sampling
    with torch.no_grad():
        c_emb: Tensor = trainer.text_encode(read_prompts(args.sample.promptfile)[: args.sample.batch])
        c_unc: Tensor = trainer.text_encode([""])

    # Training
    for batch in tqdm(loader, disable=not accelerator.is_main_process):
        out: Dict[str, Tensor] = trainer.step(batch, c_unc)

        if step % args.track.ckpt_every == 0:
            Trainer.save(args.train.ckpt_folder, loader, accelerator, model, opt, sch, step)

        if step % args.track.loss_every == 0:
            accelerator.log({"train/loss": out["loss"].cpu().item()}, step)
            accelerator.log({"train/cfm": out["cfm"].cpu().item()}, step)
            accelerator.log({"train/l2": out["mse"].cpu().item()}, step)
            accelerator.log({"train/lr": sch.get_last_lr()[0]}, step)

        if step % args.track.eval_every == 0:
            model.eval()

            # Use subset
            x_rgb: Tensor = batch["image"][: args.sample.batch].cpu()
            x_eps: Tensor = out["eps"][: args.sample.batch]
            b_emb: Tensor = out["emb"][: args.sample.batch]

            # Sample using entries
            x_out: Tensor = trainer.sample(eps=torch.randn_like(x_eps), c_cond=b_emb, c_null=c_unc).cpu()
            accelerator.log({"eval/sample": (m := makegrid(x_out))}, step)
            plt.close(m)

            # Sample using prompts
            x_out: Tensor = trainer.sample(eps=torch.randn_like(x_eps), c_cond=c_emb, c_null=c_unc).cpu()
            accelerator.log({"eval/prompt": (m := makegrid(x_out))}, step)
            plt.close(m)

            # Send inputs
            accelerator.log({"eval/inputs": (m := makegrid(x_rgb))}, step)
            plt.close(m)

            model.train()

        step += 1
