import os
import io
import signal
import datetime
import glob
import itertools
from itertools import islice
import json
import math
import re
import subprocess
import sys

pid_data = {"process_pids": []}
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
from typing import Tuple, Optional
from collections import deque
from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime, sleep

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import psutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from utils import (
    HParams,
    plot_spectrogram_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
    load_config_from_json,
    mel_spec_similarity,
    flush_writer,
    block_tensorboard_flush_on_exit,
    si_sdr,
    wave_to_mel,
    small_model_naming,
    old_session_cleanup,
    print_init_setup,
    train_loader_safety,
    verify_spk_dim,
    early_stopper
)
from losses import (
    discriminator_loss,
    generator_loss,
    discriminator_tprls_loss,
    generator_tprls_loss,
    HingeAdversarialLoss,
    feature_loss,
    kl_loss,
    phase_loss,
    MRSTFTLoss,
)
from mel_processing import (
    spec_to_mel_torch,
    MultiScaleMelSpectrogramLoss
)
from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.lib.algorithm.lora import apply_lora, merge_lora
from rvc.train.utils import replace_keys_in_dict

# Parse command line arguments start region ===========================

model_name = sys.argv[1]
epoch_save_frequency = int(sys.argv[2])
total_epoch_count = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])
save_only_latest_net_models = strtobool(sys.argv[9])
save_weight_models = strtobool(sys.argv[10])
use_warmup = strtobool(sys.argv[11])
warmup_duration = int(sys.argv[12])
cleanup = strtobool(sys.argv[13])
vocoder = sys.argv[14]
architecture = sys.argv[15]
optimizer_choice = sys.argv[16]
adversarial_loss = sys.argv[17]
use_checkpointing = strtobool(sys.argv[18])
use_tf32 = bool(strtobool(sys.argv[19]))
use_benchmark = bool(strtobool(sys.argv[20]))
use_deterministic = bool(strtobool(sys.argv[21]))
spectral_loss = sys.argv[22]
lr_scheduler = sys.argv[23]
exp_decay_gamma = float(sys.argv[24])
use_kl_annealing = strtobool(sys.argv[25])
kl_annealing_cycle_duration = int(sys.argv[26])
vits2_mode = strtobool(sys.argv[27])
rolling_loss_steps = int(sys.argv[28])

grad_clip_scheduling = bool(strtobool(sys.argv[29]))
grad_clip_steps_duration = int(sys.argv[30])
grad_clip_value_g_cap, grad_clip_value_d_cap = (int(sys.argv[31]), int(sys.argv[32]))
grad_clip_value_g_release, grad_clip_value_d_release = (int(sys.argv[33]), int(sys.argv[34]))

use_custom_lr = strtobool(sys.argv[35])
custom_lr_g, custom_lr_d = (float(sys.argv[36]), float(sys.argv[37])) if use_custom_lr else (None, None)
assert not use_custom_lr or (custom_lr_g and custom_lr_d), "Invalid custom LR values."

use_lora = bool(strtobool(sys.argv[38]))
lora_rank = int(sys.argv[39])

# Parse command line arguments end region ===========================

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

# Load the config from json
config = load_config_from_json(config_save_path)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

# AMP precision / dtype init
if config.train.fp16_run: 
    train_dtype = torch.float16
else:
    train_dtype = torch.float32

# Torch backends config
torch.backends.cuda.matmul.allow_tf32 = use_tf32
torch.backends.cudnn.allow_tf32 = use_tf32
torch.backends.cudnn.benchmark = use_benchmark
torch.backends.cudnn.deterministic = use_deterministic

# Globals ( Do not alter these )
global_step = 0
warmup_completed = False
from_scratch = False
use_lr_scheduler = lr_scheduler != "none"

# Globals ( tweakable~ )
enable_persistent_workers = True

c_stft = 0.0 # Unused for now.

pretrain_preview = True
pretrain_preview_interval = 100 # Measured in steps.

override_pretrain_lr = False
force_from_scratch = False
new_pretrain_lr = 5e-5 # If you changed it, it needs to be re-adjusted to the most recent lr ~ anytime you resume!
strict_load = True # Whether to be strict in loading ckpts for resuming

clip_grad_norm_override = False
clip_grad_norm_override_value_g = 100
clip_grad_norm_override_value_d = 100

# EXPERIMENTAL
use_sid_swap = False
custom_sid = 1

use_2_sample_kl = False # Needs more testing, for now disabled. You can still set it to True if you wanna test it out.
free_bits = 0.1

##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


class NullDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        b = y.shape[0]
        grad_anchor = self._dummy * 0
        ones = torch.ones(b, device=y.device)  + grad_anchor
        zeros = torch.zeros(b, device=y.device) + grad_anchor
        fake_fmap = [grad_anchor.expand(b)]
        return [ones], [zeros], fake_fmap, fake_fmap

def univhd_project_gamma(net_d, vocoder, rank, global_step):
    if vocoder != "APEX-GAN":
        return
    disc = net_d.module if hasattr(net_d, "module") else net_d
    gamma_val = None
    with torch.no_grad():
        for d in disc.discriminators:
            if hasattr(d, "harmonic_filter"):
                # Clamp the value
                d.harmonic_filter.gamma.clamp_(min=1.0)
                # Store it for printing
                gamma_val = d.harmonic_filter.gamma.item()
    if rank == 0 and global_step % 100 == 0 and gamma_val is not None:
        print(f"[UnivHD] gamma: {gamma_val:.6f}")

class EarlyStopSignalHandler:
    def __init__(self):
        self.stop_triggered = False
        signal.signal(signal.SIGINT, self._handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._handler)

    def _handler(self, signum, frame):
        self.stop_triggered = True
        print(f"\n[TRAINING] Early Stopping signal received! Finishing current step and saving...")


def eval_infer(net_g, reference):
    net_g.eval()
    with torch.no_grad():
        if hasattr(net_g, "module"):
            o, *_ = net_g.module.infer(*reference)
        else:
            o, *_ = net_g.infer(*reference)
    net_g.train()
    return o

class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"

def setup_env_and_distr(rank, n_gpus, device, device_id, config):
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

def prepare_dataloaders(config, n_gpus, rank, batch_size):
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=enable_persistent_workers,
        prefetch_factor=8
    )
    train_loader_safety(train_loader)

    return train_loader

def get_g_model(config, sample_rate, vocoder, use_checkpointing):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
        vits2_mode = vits2_mode,
        use_2_sample_kl = use_2_sample_kl,
    )

def get_d_model(config, vocoder, use_checkpointing):
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        # MPD + MSD + MRD ( unified ) - RingFormer architecture v1 and v2
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **dict(config.mrd)
        )
    elif vocoder == "APEX-GAN":
        '''
        CoMBD + SBD + UnivHD + GLD - TRIALS
        An experimental ensemble of mine which ( in theory ) is supposed to cover all required domains
        [ Supposedly stable but effectiveness and actual stability of UnivHD + GLD is still uncertain. ]
        '''
        from rvc.lib.algorithm.discriminators.multi import HolisticMultiDomainDiscriminator
        return HolisticMultiDomainDiscriminator(
            sample_rate=config.data.sample_rate,
            segment_size_samples=config.train.segment_size,
            use_spectral_norm=config.model.use_spectral_norm,
        )

        '''
        CoMBD + SBD + UnivHD - TRIALS
        [ Supposedly stable but effectiveness and actual stability of UnivHD is still uncertain. ]
        '''
        # from rvc.lib.algorithm.discriminators.multi import CoMBD_SBD_UnivHD_Combined
        # # CoMBD + SBD + UnivHD ( unified )
        # return CoMBD_SBD_UnivHD_Combined(
            # sample_rate=config.data.sample_rate,
            # segment_size_samples=config.train.segment_size,
            # use_spectral_norm=config.model.use_spectral_norm,
        # )


        '''
        Dummy discriminator - only for debugging / isolated generator tests
        '''
        # return NullDiscriminator()


        '''
        CoMBD + SBD + MRD preset; Heavy, cannot test it reliably on 12 gig card.
        '''
        # from rvc.lib.algorithm.discriminators.multi import CoMBD_SBD_MRD_Combined
        # return CoMBD_SBD_MRD_Combined(
            # sample_rate=config.data.sample_rate,
            # segment_size_samples=config.train.segment_size,
            # use_spectral_norm=config.model.use_spectral_norm,
            # **dict(config.mrd)
        # )


    elif vocoder == "RefineGAN":
        # from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined_RefineGan
        # # Trimmed MPD + MSD + MRD ( unified )
        # return MPD_MSD_MRD_Combined_RefineGan(
            # config.model.use_spectral_norm,
            # use_checkpointing=use_checkpointing
        # )
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        # MPD + MSD + MRD ( unified )
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **dict(config.mrd)
        )
    else: # For NSF HiFi-GAN
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined
        # MPD + MSD ( unified ) - Original RVC Setup
        return MPD_MSD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing
        )

def get_optimizers(
    net_g,
    net_d,
    config,
    optimizer_choice,
    custom_lr_g,
    custom_lr_d,
    use_custom_lr,
    total_epoch_count,
    train_loader
):
    # Common args for optims
    common_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate_g,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.0,
    )
    common_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate_d,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.0,
    )
    radam_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate_g,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01,
        decoupled_weight_decay=True,
        foreach=True,
    )
    radam_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate_d,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01,
        decoupled_weight_decay=True,
        foreach=True,
    )
    adabelief_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate_g,
        betas=(0.8, 0.999),
        eps=1e-16,
        weight_decay=0,
        #weight_decouple=False,
        rectify=True,
        adamc=False,
        #lr_max=custom_lr_g if use_custom_lr else config.train.learning_rate_g, # only used when adamc is enabled
    )
    adabelief_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate_d,
        betas=(0.8, 0.999),
        eps=1e-16,
        weight_decay=0,
        #weight_decouple=False,
        rectify=True,
        adamc=False,
        #lr_max=custom_lr_g if use_custom_lr else config.train.learning_rate_d, # only used when adamc is enabled
    )
    # For exotic optimizers
    ranger_args = dict(
        num_epochs=total_epoch_count,
        num_batches_per_epoch=len(train_loader),
        use_madgrad=False,
        use_warmup=False,
        warmdown_active=False,
        use_cheb=False,
        lookahead_active=True,
        normloss_active=False,
        normloss_factor=1e-4,
        softplus=False,
        use_adaptive_gradient_clipping=True,
        agc_clipping_value=0.01,
        agc_eps=1e-3,
        using_gc=True,
        gc_conv_only=True,
        using_normgc=False,
    )

    if optimizer_choice == "AdamW":
        optim_g = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, fused=True)
        optim_d = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, fused=True)

    elif optimizer_choice == "RAdam":
        optim_g = torch.optim.RAdam(filter(lambda p: p.requires_grad, net_g.parameters()), **radam_args_g)
        optim_d = torch.optim.RAdam(filter(lambda p: p.requires_grad, net_d.parameters()), **radam_args_d)

    elif optimizer_choice == "DiffGrad":
        from rvc.train.custom_optimizers.diffgrad import diffgrad
        optim_g = diffgrad(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g)
        optim_d = diffgrad(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d)

    elif optimizer_choice == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21
        optim_g = Ranger21(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, **ranger_args)
        optim_d = Ranger21(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, **ranger_args)

    elif optimizer_choice == "AdaBelief":
        from rvc.train.custom_optimizers.adabelief import AdaBelief
        optim_g = AdaBelief(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g)
        optim_d = AdaBelief(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d)
    else:
        raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")
    return optim_g, optim_d

def setup_models_for_training(net_g, net_d, device, device_id, n_gpus):
    net_g = net_g.to(device_id) if device.type == "cuda" else net_g.to(device)
    net_d = net_d.to(device_id) if device.type == "cuda" else net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[device_id]) # find_unused_parameters=True)

    return net_g, net_d

def _apply_lora_to_generator(net_g, rank):
    model_g = net_g.module if hasattr(net_g, "module") else net_g
    apply_lora(model_g, r=lora_rank)
    for name, param in model_g.named_parameters():
        if "lora_" in name:
            pass
        elif name.endswith(".bias"):
            pass
        else:
            param.requires_grad = False
    for emb_name in ["emb_g", "enc_p.emb_pitch"]:
        emb = model_g
        for part in emb_name.split("."):
            emb = getattr(emb, part, None)
            if emb is None:
                break
        if emb is not None and hasattr(emb, "weight"):
            emb.weight.requires_grad = True

    if rank == 0:
        total_lora = sum(p.numel() for n, p in model_g.named_parameters() if "lora_" in n)
        total_trainable = sum(p.numel() for p in model_g.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model_g.parameters())
        # Count wrappers per component
        wrappers = {"enc_p": 0, "dec": 0, "enc_q": 0, "flow": 0, "other": 0}
        for n, m in model_g.named_modules():
            if hasattr(m, "lora_A"):
                key = n.split(".")[0]
                if key in wrappers:
                    wrappers[key] += 1
                else:
                    wrappers["other"] += 1
        print(f"[ ] LoRA r={lora_rank}: {total_lora:,} adapter params ({total_trainable:,} trainable / {total_all:,} total)")
        for comp in ["enc_p", "dec", "enc_q", "flow"]:
            if wrappers[comp]:
                print(f"      {comp}: {wrappers[comp]} layers")
        if wrappers["other"]:
            print(f"      other: {wrappers['other']} layers")
        print(f"      + emb_g.weight, enc_p.emb_pitch.weight (direct)")


def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, sample_rate, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing)
    net_d = get_d_model(config, vocoder, use_checkpointing)
    try:
        print("    ██████  Starting the training ...")

        # Get latest G and D based on the highest steps count in the filename
        def get_highest_checkpoint(prefix):
            pattern = re.compile(rf"^{prefix}(\d+)\.pth$")
            files = []
            for f in os.listdir(experiment_dir):
                match = pattern.match(f)
                if match:
                    files.append((int(match.group(1)), os.path.join(experiment_dir, f)))
            return sorted(files, key=lambda x: x[0], reverse=True)[0][1] if files else None

        # Confirm presence of checkpoints
        # If they exist, attempt to resume the training
        g_checkpoint_path = get_highest_checkpoint("G_")
        d_checkpoint_path = get_highest_checkpoint("D_")
        if g_checkpoint_path and d_checkpoint_path:

            # Apply LoRA before loading checkpoint so LoRA keys exist
            if use_lora and lora_rank > 0:
                _apply_lora_to_generator(net_g, rank)

            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

            # Init the optimizers
            optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

            # Load the model and optim states
            _, _, _, epoch_str, gradscaler_dict = load_checkpoint(g_checkpoint_path, net_g, optim_g, strict_load)
            _, _, _, epoch_str, _ = load_checkpoint(d_checkpoint_path, net_d, optim_d, strict_load)

            if override_pretrain_lr:
                new_lr_for_pretrain = new_pretrain_lr
                for param_group in optim_g.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                for param_group in optim_d.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                print(f"[OVERRIDE] Pretrain LR Override: {new_lr_for_pretrain}")

            #epoch_str += 1
            #global_step = (epoch_str - 1) * len(train_loader)

            global_step = int(os.path.basename(g_checkpoint_path).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            print(f"[RESUMING] (G) & (D) at global_step: {global_step} and epoch count: {epoch_str - 1}")

            # Re-create optimizers for LoRA (only track trainable params)
            if use_lora and lora_rank > 0:
                optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)
        else:
            raise FileNotFoundError("No checkpoints found.")

    except FileNotFoundError:
    # If no checkpoints are available, using the Pretrains directly
        epoch_str = 1
        global_step = 0
        gradscaler_dict = {}

        # Loading the pretrained Generator model
        if pretrainG not in ["", "None"]:
            if rank == 0:
                print(f"[ ] Loading pretrained (G) '{pretrainG}'")
            checkpoint = torch.load(pretrainG, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_g.load_state_dict(state_dict, strict=True)

            if use_sid_swap and custom_sid != 0:
                total_sids = net_g.emb_g.weight.size(0)

                if custom_sid >= total_sids:
                    print(f"[SID SWAP] {custom_sid} is out of bounds!")
                    print(f"[SID SWAP] Currently chosen pretrains only support SIDs from 0 to {total_sids - 1}.")
                    sys.exit("Invalid SID Selection. Please choose a lower custom_sid.")
                if rank == 0:
                    print(f"███ [SID SWAP] Swapping SID: 0 with SID: {custom_sid}")

                with torch.no_grad():
                    temp_sid_0 = net_g.emb_g.weight[0].clone()
                    net_g.emb_g.weight[0].copy_(net_g.emb_g.weight[custom_sid])
                    net_g.emb_g.weight[custom_sid].copy_(temp_sid_0)
                if rank == 0:
                    print(f"███ [SID SWAP] Swap successful. Model is ready for fine-tuning.")

        # Loading the pretrained Discriminator model
        if pretrainD not in ["", "None"]:
            if rank == 0:
                print(f"[ ] Loading pretrained (D) '{pretrainD}'")
            checkpoint = torch.load(pretrainD, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_d.load_state_dict(state_dict, strict=True)

        # Apply LoRA after loading pretrained weights (wraps convs with loaded weights preserved)
        if use_lora and lora_rank > 0:
            _apply_lora_to_generator(net_g, rank)

        # Load the models and optionally wrap with DDP
        net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict

def prepare_schedulers(optim_g, optim_d, use_warmup, warmup_duration, use_lr_scheduler, lr_scheduler, exp_decay_gamma, total_epoch_count, epoch_str, global_step, train_loader):
    warmup_scheduler_g, warmup_scheduler_d = None, None
    scheduler_g, scheduler_d = None, None

    num_batches_per_epoch = len(train_loader)

    if override_pretrain_lr:
        scheduler_resume_epoch = -1
        scheduler_resume_step = -1
    else:
        scheduler_resume_epoch = epoch_str - 1
        scheduler_resume_step = global_step - 1

    if use_warmup:
        warmup_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            optim_g, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_duration)
        )
        warmup_scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            optim_d, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_duration)
        )

    if not use_warmup:
        for param_group in optim_g.param_groups: # For Generator
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        for param_group in optim_d.param_groups: # For Discriminator
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler:
        if lr_scheduler == "exp decay epoch":
            # Exponential decay lr scheduler per epoch
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch)
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch)

        elif lr_scheduler == "exp decay step":
            # Exponential-decay style lr scheduler per step
            exp_decay_gamma_step = exp_decay_gamma ** (1.0 / num_batches_per_epoch)
            class DynamicStepLR(torch.optim.lr_scheduler.MultiplicativeLR):
                def __init__(self, optimizer, gamma, last_epoch=-1):
                    self.gamma = gamma
                    super().__init__(optimizer, lr_lambda=lambda step: self.gamma, last_epoch=last_epoch)
            scheduler_g = DynamicStepLR(optim_g, gamma=exp_decay_gamma_step, last_epoch=scheduler_resume_step)
            scheduler_d = DynamicStepLR(optim_d, gamma=exp_decay_gamma_step, last_epoch=scheduler_resume_step)

        elif lr_scheduler == "cosine annealing epoch":
            # Cosine annealing lr scheduler per epoch
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch)

    return warmup_scheduler_g, warmup_scheduler_d, scheduler_g, scheduler_d


def get_reference_sample(train_loader, device, config):
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),
    ])

    if use_custom_ref:
        print("[REFERENCE] Using custom reference input from 'logs\\reference\\'")

        phone = torch.FloatTensor(np.repeat(np.load(os.path.join(reference_path, "ref_feats.npy")), 2, axis=0)).unsqueeze(0).to(device)
        pitch = torch.LongTensor(np.load(os.path.join(reference_path, "ref_f0c.npy"))).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(np.load(os.path.join(reference_path, "ref_f0f.npy"))).unsqueeze(0).to(device)

        # Measure lengths
        lengths = [phone.shape[1], pitch.shape[1], pitchf.shape[1]]
        min_len = min(lengths)

        # Trim to min length
        phone = phone[:, :min_len, :]
        pitch = pitch[:, :min_len]
        pitchf = pitchf[:, :min_len]
        phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)
        sid = torch.LongTensor([0]).to(device)

    else:
        print("[REFERENCE] No custom reference found. Fetching from train_loader.")
        info = next(iter(train_loader))
        # Unpack everything from the loader
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info

        # Move only the first sample of the batch to device
        phone = phone[0:1].to(device)
        phone_lengths = phone_lengths[0:1].to(device)
        pitch = pitch[0:1].to(device)
        pitchf = pitchf[0:1].to(device)
        sid = sid[0:1].to(device)

        batch_indices = []
        for batch in train_loader.batch_sampler:
            batch_indices = batch
            break

        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            file_paths = train_loader.dataset.dataset.get_file_paths(batch_indices)
        else:
            file_paths = train_loader.dataset.get_file_paths(batch_indices)

        file_name = os.path.basename(file_paths[0])
        print(f"[REFERENCE] Origin of the ref: {file_name}")

    return (phone, phone_lengths, pitch, pitchf, sid, config.train.seed)





def main():
    """
    Main function to start the training process.
    """
    global gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    wavs = [wav for wav in glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*")) if wav.endswith((".wav", ".flac"))]
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print(f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz).")
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus) 
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("No GPU detected, fallback to CPU. This will take a very long time ...")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []

        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    total_epoch_count,
                    epoch_save_frequency,
                    save_weight_models,
                    save_only_latest_net_models,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()
            pid_data["process_pids"].append(subproc.pid)

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        old_session_cleanup(now_dir, model_name)
    start()

def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed, optimizer_choice, from_scratch

    stopper = EarlyStopSignalHandler()

    if 'warmup_completed' not in globals():
        warmup_completed = False

    # Initial print / session info for console
    print_init_setup(
        warmup_duration,
        rank,
        use_warmup,
        config,
        optimizer_choice,
        lr_scheduler,
        exp_decay_gamma,
        use_kl_annealing,
        kl_annealing_cycle_duration,
        spectral_loss,
        adversarial_loss,
        vits2_mode
    )

    # Initial setup
    setup_env_and_distr(
        rank,
        n_gpus,
        device,
        device_id,
        config
    )

    # Dataloading and loaders preparation
    train_loader = prepare_dataloaders(
        config,
        n_gpus,
        rank,
        batch_size
    )

    # Spk dim verif
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, pretrainG)
    config.model.spk_embed_dim = spk_dim

    # Spectral loss init
    fn_spectral_loss2 = None

    if spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
    elif spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
    elif spectral_loss == "Hybrid L1":
        fn_spectral_loss = torch.nn.L1Loss()
        fn_spectral_loss2 = MRSTFTLoss(sample_rate=sample_rate, fmin=5000.0, fmin_weight=0.1)
    elif spectral_loss == "Hybrid MS":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
        fn_spectral_loss2 = MRSTFTLoss(sample_rate=sample_rate, fmin=5000.0, fmin_weight=0.1)
    elif spectral_loss == "DEBUG":
        fn_spectral_loss = MRSTFTLoss(sample_rate=sample_rate, fmin=5000.0, fmin_weight=0.1)
    else:
        print("ERROR: Chosen spectral loss is undefined. Exiting.")
        sys.exit(1)

    # Hinge adversarial loss
    fn_hinge_loss = HingeAdversarialLoss() if adversarial_loss == "hinge" else None

    # Loading of models and optims
    net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict = load_models_and_optimizers(
        config,
        pretrainG,
        pretrainD,
        vocoder,
        use_checkpointing,
        sample_rate, 
        optimizer_choice,
        custom_lr_g,
        custom_lr_d,
        use_custom_lr, 
        total_epoch_count,
        train_loader,
        device,
        device_id,
        n_gpus,
        rank
    )

    # Tensorboard handling
    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            flush_secs=86400,
            purge_step=global_step + 1
        )
        block_tensorboard_flush_on_exit(writer_eval)

        if global_step != 0:
            print(f"[INIT] TensorBoard writer initialized. Purging logs after step: {global_step}")
        else:
            print(f"[INIT] TensorBoard writer initialized.")

    # from-scratch checker ( disables average loss )
    if (pretrainG in ["", "None"] and pretrainD in ["", "None"]) or force_from_scratch:
        from_scratch = True
        if rank == 0:
            print("[INIT] No pretrains used: Average loss disabled!")

    # Prepare the schedulers
    warmup_scheduler_g, warmup_scheduler_d, scheduler_g, scheduler_d = prepare_schedulers(
        optim_g,
        optim_d,
        use_warmup,
        warmup_duration,
        use_lr_scheduler, 
        lr_scheduler,
        exp_decay_gamma,
        total_epoch_count,
        epoch_str,
        global_step,
        train_loader
    )

    # Hann window for stft ( for RingFormer only. )
    hann_window = torch.hann_window(config.model.gen_istft_n_fft).to(device) if vocoder in ["RingFormer_v1", "RingFormer_v2"] else None

    # GradScaler for FP16 training
    gradscaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    if len(gradscaler_dict) > 0:
        gradscaler.load_state_dict(gradscaler_dict)
        print("[INIT] Loading gradscaler state dict - FP16")

    # Reference sample for live-infer
    reference = get_reference_sample(train_loader, device, config)

    # Cache for training with " cache " enabled
    cache = []

    for epoch in range(epoch_str, total_epoch_count + 1):
        should_stop = training_loop(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            train_loader,
            [writer_eval],
            cache,
            total_epoch_count,
            epoch_save_frequency,
            save_weight_models,
            save_only_latest_net_models,
            device,
            device_id,
            reference,
            fn_spectral_loss,
            n_gpus,
            gradscaler,
            fn_hinge_loss,
            fn_spectral_loss2,
            hann_window,
            stopper=stopper
        )

        if use_warmup and epoch <= warmup_duration:
            if warmup_scheduler_g:
                warmup_scheduler_g.step()
            if warmup_scheduler_d:
                warmup_scheduler_d.step()

            # Logging of finished warmup
            if epoch == warmup_duration:
                warmup_completed = True
                print(f"    ██████  Warmup completed at epochs: {warmup_duration}")
                print(f"    ██████  LR G: {optim_g.param_groups[0]['lr']}")
                print(f"    ██████  LR D: {optim_d.param_groups[0]['lr']}")
                # scheduler:
                if lr_scheduler == "exp decay epoch":
                    print(f"    ██████  Starting the per-epoch exponential lr decay with gamma of {exp_decay_gamma}")
                elif lr_scheduler == "cosine annealing epoch":
                    print("    ██████  Starting per-epoch cosine annealing scheduler " )

        if use_lr_scheduler and (not use_warmup or warmup_completed):
            # Once the warmup phase is completed, uses exponential lr decay
            if lr_scheduler in ["exp decay epoch", "cosine annealing epoch"]:
                scheduler_g.step()
                scheduler_d.step()

def training_loop(
    rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    train_loader,
    writers,
    cache,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    device,
    device_id,
    reference,
    fn_spectral_loss,
    n_gpus,
    gradscaler,
    fn_hinge_loss=None,
    fn_spectral_loss2=None,
    hann_window=None,
    stopper=None
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        config (object): Configuration object containing training parameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, net_d].
        train_loader: training dataloader.
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        device (torch.device): The device to use for training (CPU or GPU).
        reference (list): Contains reference sample. Either custom or from train loader.
        fn_spectral_loss: spectral loss;  can be l1, multi-scale or ms-stft.
        fn_spectral_loss2: 2nd spectral loss
        gradscaler: gradscaler for fp16
        hann_window: hann window used for RingFormer
    """
    global global_step, warmup_completed, use_lr_scheduler, lr_scheduler, use_warmup

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers if schedulers is not None else (None, None)

    train_loader = train_loader if train_loader is not None else None

    if writers is not None:
        writer = writers[0]

    fn_hinge_loss = fn_hinge_loss if fn_hinge_loss is not None else None
    
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    # Partial resume aligning
    current_epoch_start_step = (epoch - 1) * len(train_loader)
    start_batch_idx = global_step - current_epoch_start_step
    start_batch_idx = max(0, start_batch_idx)

    if start_batch_idx > 0:
        train_loader.batch_sampler.start_index = start_batch_idx

    remaining_batches = len(train_loader) - start_batch_idx
    data_iterator = islice(enumerate(train_loader), remaining_batches)

    epoch_recorder = EpochRecorder()

    if not from_scratch:
        # Tensors init for averaged losses:
        if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
            tensor_count = 7
        else:
            tensor_count = 6
        epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        num_batches_in_epoch = 0

    avg_rolling_cache = {
        "grad_norm_d": deque(maxlen=rolling_loss_steps),
        "grad_norm_g": deque(maxlen=rolling_loss_steps),
        "loss_disc": deque(maxlen=rolling_loss_steps),
        "loss_adv": deque(maxlen=rolling_loss_steps),
        "loss_gen_total": deque(maxlen=rolling_loss_steps),
        "loss_fm": deque(maxlen=rolling_loss_steps),
        "loss_spectral": deque(maxlen=rolling_loss_steps),
        "loss_kl": deque(maxlen=rolling_loss_steps),
    }
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        avg_rolling_cache["loss_sd"] = deque(maxlen=rolling_loss_steps)

    use_amp = config.train.fp16_run and device.type == "cuda"

    with tqdm(total=len(train_loader), leave=False, initial=start_batch_idx) as pbar:
        for batch_idx, info in data_iterator:

            # # Catch up with previously processed steps if resuming from partial ckpts
            # if batch_idx < start_batch_idx:
                # continue

            global_step += 1
            if not from_scratch:
                num_batches_in_epoch += 1

            # Clip scheduling
            if not clip_grad_norm_override:
                if grad_clip_scheduling and grad_clip_steps_duration > 0:
                    if global_step < grad_clip_steps_duration:
                        # Clip
                        grad_clip_value_g = grad_clip_value_g_cap if grad_clip_value_g_cap != 0 else float("inf")
                        grad_clip_value_d = grad_clip_value_d_cap if grad_clip_value_d_cap != 0 else float("inf")
                    else:
                        # Release ( or 2nd clip phase )
                        grad_clip_value_g = grad_clip_value_g_release if grad_clip_value_g_release != 0 else float("inf")
                        grad_clip_value_d = grad_clip_value_d_release if grad_clip_value_d_release != 0 else float("inf")
                else:
                    grad_clip_value_g = grad_clip_value_d = float("inf") # Default: No Clipping
            else:
                grad_clip_value_g = clip_grad_norm_override_value_g
                grad_clip_value_d = clip_grad_norm_override_value_d

            # Device handling
            if device.type == "cuda":
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]

            # Tuple unpacking
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid) = info

            # Generator forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)

                # Vocoder-dependent unpacking
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    y_hat, ids_slice, x_mask, z_mask, vae_parts, (mag, _) = model_output
                elif vocoder == "APEX-GAN":
                    # y_hat_list = list of [coarse, mid, full] intermediates.
                    y_hat_list, ids_slice, x_mask, z_mask, vae_parts = model_output
                    y_hat = y_hat_list[-1] # final full-res waveform
                else:
                    y_hat, ids_slice, x_mask, z_mask, vae_parts = model_output

                # latent samples + Gaussian params for ELBO
                z, z_p, z_p2, m_p, logs_p, m_q, logs_q = vae_parts

                # Slice the original waveform ( y ) to match the generated slice:
                y = commons.slice_segments(y, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)


            # RingFormer related
            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                reshaped_y = y.view(-1, y.size(-1))
                reshaped_y_hat = y_hat.view(-1, y_hat.size(-1))

                y_stft = torch.stft(reshaped_y, n_fft=config.model.gen_istft_n_fft, hop_length=config.model.gen_istft_hop_size, win_length=config.model.gen_istft_n_fft, window=hann_window, return_complex=True)
                y_hat_stft = torch.stft(reshaped_y_hat, n_fft=config.model.gen_istft_n_fft, hop_length=config.model.gen_istft_hop_size, win_length=config.model.gen_istft_n_fft, window=hann_window, return_complex=True)
                target_magnitude = torch.abs(y_stft)  # shape: [B, F, T]

                loss_magnitude = torch.nn.functional.l1_loss(mag, target_magnitude)
                loss_phase = phase_loss(y_stft, y_hat_stft)
                loss_sd = (loss_magnitude + loss_phase) * 0.7


            # Discriminator forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                if vocoder == "APEX-GAN":
                    y_hat_d = [o.detach() for o in y_hat_list]
                else:
                    y_hat_d = y_hat.detach()
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat_d)

            with autocast(device_type="cuda", enabled=False):
                # Compute discriminator loss:
                if adversarial_loss == "lsgan":
                    loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                elif adversarial_loss == "tprls":
                    loss_disc = discriminator_tprls_loss(y_d_hat_r, y_d_hat_g)
                elif adversarial_loss == "hinge":
                    loss_fake, loss_real = fn_hinge_loss(y_d_hat_g, y_d_hat_r)
                    loss_disc = loss_fake + loss_real


            # Discriminator backward and update:
            optim_d.zero_grad(set_to_none=True)
            if train_dtype == torch.float16:
                gradscaler.scale(loss_disc).backward() # Scale and backward of the loss
                gradscaler.unscale_(optim_d) # Unscale
                scale = gradscaler.get_scale() # To retrieve current gradscaler's scaling
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=grad_clip_value_d) # Grad clipping
                gradscaler.step(optim_d) # Optim step
                univhd_project_gamma(net_d, vocoder, rank, global_step) # univhd safety constraint
            else:
                loss_disc.backward() # Loss backward
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=grad_clip_value_d) # Grad clipping
                optim_d.step() # Optim step
                univhd_project_gamma(net_d, vocoder, rank, global_step) # univhd safety constraint

            # Run discriminator on generated output
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                if vocoder == "APEX-GAN":
                    _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat_list)
                else:
                    _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

            # Compute generator losses:
            with autocast(device_type="cuda", enabled=False):

                # Spectral loss
                if spectral_loss == "L1 Mel Loss":
                    y_mel = wave_to_mel(config, y, half=train_dtype)
                    y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                    loss_spectral = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                elif spectral_loss == "Multi-Scale Mel Loss":
                    loss_spectral = fn_spectral_loss(y, y_hat) * config.train.c_mel / 3.0 # * 15
                elif spectral_loss == "Hybrid L1":
                    # L1 Mel
                    y_mel = wave_to_mel(config, y, half=train_dtype)
                    y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                    loss_l1_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel # * 45
                    # MR-STFT
                    loss_mrstft = fn_spectral_loss2(y_hat.float(), y.float()) * c_stft # 80
                    # Loss
                    loss_spectral = loss_l1_mel + loss_mrstft * 0.50
                elif spectral_loss == "Hybrid MS":
                    # Multi-Scale el L1
                    loss_ms_mel = fn_spectral_loss(y, y_hat) * config.train.c_mel / 3.0 # * 15
                    # MR-STFT
                    loss_mrstft = fn_spectral_loss2(y_hat.float(), y.float()) * c_stft # 80
                    # Loss
                    loss_spectral = loss_ms_mel + loss_mrstft * 0.50

                # Feature Matching loss
                loss_fm = feature_loss(fmap_r, fmap_g) * 2.0

                # Generator loss
                if adversarial_loss == "lsgan":
                    loss_adv = generator_loss(y_d_hat_g)
                elif adversarial_loss == "tprls":
                    y_d_hat_r_detached = [i.detach() for i in y_d_hat_r]
                    loss_adv = generator_tprls_loss(y_d_hat_r_detached, y_d_hat_g)
                elif adversarial_loss == "hinge":
                    loss_adv = fn_hinge_loss(y_d_hat_g)


                # Kl annealing handler
                if use_kl_annealing:
                    annealing_cycle_steps = len(train_loader) * kl_annealing_cycle_duration
                    kl_beta = 0.5 * (1 - math.cos((global_step % annealing_cycle_steps) * (math.pi / annealing_cycle_steps)))
                else:
                    kl_beta = 1.0

                # KL ( Kullback–Leibler divergence ) loss
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask, z_p2, free_bits) * config.train.c_kl 


                # RingFormer related
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    loss_magnitude = torch.nn.functional.l1_loss(mag, target_magnitude)
                    loss_phase = phase_loss(y_stft, y_hat_stft)
                    loss_sd = (loss_magnitude + loss_phase) * 0.7


                # Total generator loss
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    loss_gen_total = loss_adv + loss_fm + loss_spectral + loss_kl * kl_beta + loss_sd
                elif vocoder == "RefineGAN":
                    loss_gen_total = loss_adv + loss_fm + loss_spectral + loss_kl * kl_beta
                elif vocoder == "apex_gan":
                    loss_gen_total = loss_adv + loss_fm + loss_spectral + loss_kl * kl_beta
                else:
                    loss_gen_total = loss_adv + loss_fm + loss_spectral + loss_kl * kl_beta

            # Generator backward and update:
            optim_g.zero_grad(set_to_none=True)
            if train_dtype == torch.float16:
                gradscaler.scale(loss_gen_total).backward() # Scale and backward of the loss
                gradscaler.unscale_(optim_g) # Unscale
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=grad_clip_value_g) # Grad clipping
                gradscaler.step(optim_g) # Optim step
                gradscaler.update() # Scaler update, to prepare the scaling for the next iteration
                skip_lr_sched = (scale > gradscaler.get_scale())
            else:
                loss_gen_total.backward() # Loss backward
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=grad_clip_value_g) # Grad clipping
                optim_g.step() # Optim step
                skip_lr_sched = False


            # Per step exp lr decay for generator
            if not skip_lr_sched: # We skip lr scheduler step if there were nans / infs due to gradscaler's scaling.
                if use_lr_scheduler and (not use_warmup or warmup_completed) and lr_scheduler == "exp decay step":
                    scheduler_d.step()
                    scheduler_g.step()

            if not from_scratch:
                # Loss accumulation for epoch-avg
                epoch_loss_tensor[0].add_(loss_disc.detach())
                epoch_loss_tensor[1].add_(loss_adv.detach())
                epoch_loss_tensor[2].add_(loss_gen_total.detach())
                epoch_loss_tensor[3].add_(loss_fm.detach())
                epoch_loss_tensor[4].add_(loss_spectral.detach())
                epoch_loss_tensor[5].add_(loss_kl.detach())

                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    epoch_loss_tensor[6].add_(loss_sd.detach())

            # Loss accumulation for rolling-avg
            # Grads:
            if torch.isfinite(grad_norm_d):
                avg_rolling_cache["grad_norm_d"].append(grad_norm_d)
            else:
                writer.add_scalar("Grad_Norm/D_Skipped", 1, global_step)

            if torch.isfinite(grad_norm_g):
                avg_rolling_cache["grad_norm_g"].append(grad_norm_g)
            else:
                writer.add_scalar("Grad_Norm/G_Skipped", 1, global_step)

            # Losses:
            avg_rolling_cache["loss_disc"].append(loss_disc.detach())
            avg_rolling_cache["loss_adv"].append(loss_adv.detach()) 
            avg_rolling_cache["loss_gen_total"].append(loss_gen_total.detach())
            avg_rolling_cache["loss_fm"].append(loss_fm.detach())
            avg_rolling_cache["loss_spectral"].append(loss_spectral.detach())
            avg_rolling_cache["loss_kl"].append(loss_kl.detach())
            if "loss_sd" in avg_rolling_cache:
                avg_rolling_cache["loss_sd"].append(loss_sd.detach())


            if rank == 0 and global_step % rolling_loss_steps == 0:
                scalar_dict_rolling = {}

                # Learning rate retrieval for rolling logging
                if from_scratch:
                    scalar_dict_rolling.update({
                        "learning_rate/lr_d": optim_d.param_groups[0]["lr"],
                        "learning_rate/lr_g": optim_g.param_groups[0]["lr"],
                    })

                # logging rolling averages
                for key, queue in avg_rolling_cache.items():
                    if len(queue) > 0:
                        # determine loss or grad category
                        category = "loss" if "loss" in key else "grad"
                        # dynamic labeling
                        label = f"{category}_avg_{rolling_loss_steps}/{key}_{rolling_loss_steps}"
                        # Calculate mean
                        val = torch.stack(list(queue)).mean().item() if torch.is_tensor(queue[0]) else sum(queue)/len(queue)
                        scalar_dict_rolling[label] = val

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_rolling)
                flush_writer(writer, rank)

            if from_scratch and pretrain_preview and rank == 0 and global_step % pretrain_preview_interval == 0:
                print(f"    ██████  Generating pretrain-preview at step: {global_step}...")
                o = eval_infer(net_g, reference)
                audio_dict = {f"gen/audio_pretrain_{global_step}s": o[0, :, :]}
                summarize(
                    writer=writer,
                    global_step=global_step,
                    audios=audio_dict,
                    audio_sample_rate=config.data.sample_rate,
                )
                flush_writer(writer, rank)
                torch.cuda.empty_cache()

            pbar.update(1)

            if early_stopper(
                stopper, rank, global_step, epoch, architecture, 
                [net_g, net_d], [optim_g, optim_d], config, 
                experiment_dir, gradscaler, save_weight_models,
                model_name, vocoder, vits2_mode, n_gpus
            ):
                return True

        # end of batch train
    # end of tqdm

    if n_gpus > 1 and device.type == 'cuda':
        dist.barrier()

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        # For fp16 we need to .half() the mel spec
        if train_dtype == torch.float16:
            mel = mel.half()

        # Used for tensorboard chart - slice/mel_org
        y_mel = commons.slice_segments(
            mel,
            ids_slice,
            config.train.segment_size // config.data.hop_length,
            dim=3,
        )

        # used for tensorboard chart - slice/mel_gen
        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)

        # Mel similarity metric:
        mel_similarity = mel_spec_similarity(y_hat_mel, y_mel)
        print(f'Mel Spectrogram Similarity: {mel_similarity:.2f}%')
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_similarity, global_step)

        # Learning rate retrieval for avg-epoch variation:
        lr_d = optim_d.param_groups[0]["lr"]
        lr_g = optim_g.param_groups[0]["lr"]

        # At each epoch completion
        if global_step % len(train_loader) == 0 and not from_scratch:

            # Calculate the avg epoch loss:
            avg_epoch_loss = epoch_loss_tensor / num_batches_in_epoch

            # metrics dict
            scalar_dict_avg = {
            "loss_avg/loss_disc": avg_epoch_loss[0].item(),
            "loss_avg/loss_adv": avg_epoch_loss[1].item(),
            "loss_avg/loss_gen_total": avg_epoch_loss[2].item(),
            "loss_avg/loss_fm": avg_epoch_loss[3].item(),
            "loss_avg/loss_spectral": avg_epoch_loss[4].item(),
            "loss_avg/loss_kl": avg_epoch_loss[5].item(),
            "learning_rate/lr_d": lr_d,
            "learning_rate/lr_g": lr_g,
            }
            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                scalar_dict_avg.update({"loss_avg/loss_sd": avg_epoch_loss[6].item()})

            summarize(writer=writer, global_step=global_step, scalars=scalar_dict_avg)
            flush_writer(writer, rank)
            num_batches_in_epoch = 0
            epoch_loss_tensor.zero_()

        # Determine the plot data type
        if train_dtype == torch.float16:
            plot_dtype = torch.float16
        else:
            plot_dtype = torch.float32

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().to(plot_dtype).numpy()),
        }

        # At each epoch save point:
        if epoch % epoch_save_frequency == 0:

            # Inferencing on reference sample
            o = eval_infer(net_g, reference)
            audio_dict = {f"gen/audio_{epoch}e_{global_step}s": o[0, :, :]} # Eval-infer samples
            # Logging
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
            flush_writer(writer, rank)
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )
            flush_writer(writer, rank)

    # Save checkpoint
    model_add = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % epoch_save_frequency == 0:
            g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
            d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")

            if save_only_latest_net_models:
                old_files = glob.glob(os.path.join(experiment_dir, "G_*.pth")) + glob.glob(os.path.join(experiment_dir, "D_*.pth"))
                for f in old_files:
                    try:
                        os.remove(f)
                    except:
                        pass

            # Save Generator checkpoint
            save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path, gradscaler)
            # Save Discriminator checkpoint
            save_checkpoint(net_d, optim_d, config.train.learning_rate_d, epoch, d_path, gradscaler)

            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_add.append(os.path.join(experiment_dir, weight_model_name))

        # Check completion
        if epoch >= total_epoch_count:
            print(f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_total.item(), 3)} loss gen.")
            # Final model
            weight_model_name = small_model_naming(model_name, epoch, global_step)
            model_add.append(os.path.join(experiment_dir, weight_model_name))
            done = True

        if model_add:
            # Merge LoRA into conv weights for clean extraction
            if use_lora and lora_rank > 0:
                if rank == 0:
                    print(f"    ██████  Merging LoRA weights for model extraction...")
                model_g = net_g.module if hasattr(net_g, "module") else net_g
                # Deep-copy full model so we never modify the live model
                import copy
                model_copy = copy.deepcopy(model_g)
                merge_lora(model_copy, remove_wrappers=True, restore_weight_norm=True)
                ckpt = model_copy.state_dict()
            else:
                ckpt = (net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict())

            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=config,
                        vocoder=vocoder,
                        architecture=architecture,
                        vits2_mode=vits2_mode,
                    )
        if done:
            # Clean-up process IDs from memory
            pid_data["process_pids"].clear()  # Clear the PID list when done

            if rank == 0:
                writer.flush()
                writer.close()

            os._exit(0) #2333333

        with torch.no_grad():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
