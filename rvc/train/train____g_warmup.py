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
    kl_loss,
    kl_loss_fb,
    phase_loss,
    MultiScaleSTFTLoss,
)
from mel_processing import (
    spec_to_mel_torch,
    MultiScaleMelSpectrogramLoss
)
from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
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
optimizer_choice_g = sys.argv[16]
optimizer_choice_d = sys.argv[17]
use_checkpointing = strtobool(sys.argv[18])
use_tf32 = bool(strtobool(sys.argv[19]))
use_benchmark = bool(strtobool(sys.argv[20]))
use_deterministic = bool(strtobool(sys.argv[21]))
spectral_loss = sys.argv[22]
lr_scheduler_g = sys.argv[23]
lr_scheduler_d = sys.argv[24]
exp_decay_gamma_g = float(sys.argv[25])
exp_decay_gamma_d = float(sys.argv[26])
use_kl_annealing = strtobool(sys.argv[27])
kl_annealing_cycle_duration = int(sys.argv[28])
rolling_loss_steps = int(sys.argv[29])

grad_clip_scheduling = bool(strtobool(sys.argv[30]))
grad_clip_steps_duration = int(sys.argv[31])
grad_clip_value_g_cap, grad_clip_value_d_cap = (int(sys.argv[32]), int(sys.argv[33]))
grad_clip_value_g_release, grad_clip_value_d_release = (int(sys.argv[34]), int(sys.argv[35]))

use_custom_lr = strtobool(sys.argv[36])
custom_lr_g, custom_lr_d = (float(sys.argv[37]), float(sys.argv[38])) if use_custom_lr else (None, None)
assert not use_custom_lr or (custom_lr_g and custom_lr_d), "Invalid custom LR values."

use_2_sample_kl = bool(strtobool(sys.argv[39]))
use_best_step = bool(strtobool(sys.argv[40]))
double_d_updates = bool(strtobool(sys.argv[41]))

save_g_steps = 100  # 0 = disabled; set to N to save G checkpoint every N steps (e.g. 100)


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
use_lr_scheduler_g = lr_scheduler_g != "none"

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

free_bits = 0.0 # Total KL floor in nats (split across dims internally). 0.0 = no floor.


# --- Decoder layer Freezing ---
freeze_dec_upsamplers = False       # Transposed conv upsamplers (source of spectral aliasing)
freeze_dec_noise_convs = False      # Harmonic source injection layers (per-upsample-level)
freeze_dec_resblocks = False        # Residual blocks
freeze_dec_conv_pre = False         # Pre-convolution layer
freeze_dec_conv_post = False        # Post-convolution layer
freeze_dec_cond = False             # Speaker conditioning layer
freeze_dec_source_module = False    # NSF source module (SineGenerator + Linear + Tanh)


# --- Decoder differential LR ---
# 0.1 = 10%        0.01 = 1%        0.001 = 0.1% of base LR

dec_upsamplers_lr_scale = None # 0.1       # Slow down upsamplers instead of freezing ## 0.001 originally
dec_noise_convs_lr_scale = None     # Slow down noise convs instead of freezing
dec_resblocks_lr_scale = None # 0.1

##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

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

def endless_loader(loader):
    while True:
        for batch in loader:
            yield batch

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
        use_2_sample_kl = use_2_sample_kl,
    )


def _make_optimizer(model, choice, lr, num_epochs=None, num_batches=None, param_groups=None):
    params = param_groups if param_groups is not None else filter(lambda p: p.requires_grad, model.parameters())

    if choice == "AdamW":
        return torch.optim.AdamW(params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01, fused=True)

    elif choice == "RAdam":
        return torch.optim.RAdam(params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01, decoupled_weight_decay=True)

    elif choice == "DiffGrad":
        from rvc.train.custom_optimizers.diffgrad import diffgrad
        return diffgrad(params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0)

    elif choice == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21
        ranger_kw = dict(
            num_epochs=num_epochs, num_batches_per_epoch=num_batches,
            use_madgrad=False, use_warmup=False, warmdown_active=False,
            use_cheb=False, lookahead_active=True, normloss_active=False,
            normloss_factor=1e-4, softplus=False,
            use_adaptive_gradient_clipping=True, agc_clipping_value=0.01,
            agc_eps=1e-3, using_gc=True, gc_conv_only=True, using_normgc=False,
        )
        return Ranger21(params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0, **ranger_kw)

    elif choice == "AdaBelief":
        from rvc.train.custom_optimizers.adabelief import AdaBelief
        return AdaBelief(params, lr=lr, betas=(0.8, 0.999), eps=1e-16, weight_decay=0, rectify=False)

    elif choice == "Sched-Free AdamW":
        from schedulefree import AdamWScheduleFree
        return AdamWScheduleFree(params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01, warmup_steps=warmup_duration if use_warmup else 0)

    elif choice == "Sched-Free RAdam":
        from schedulefree import RAdamScheduleFree
        return RAdamScheduleFree(params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0, r=0.0, weight_lr_power=2.0, foreach=False, silent_sgd_phase=False)
    else:
        raise ValueError(f"Unknown optimizer choice: {choice}")


def build_decoder_param_groups(net_g, base_lr):
    """Build optimizer param groups with differential LR for decoder components.
    Returns None if no LR scales are configured (use default optimizer path)."""
    model = net_g.module if hasattr(net_g, "module") else net_g
    dec = getattr(model, "dec", None)
    if dec is None:
        return None

    scale_map = {}
    if dec_upsamplers_lr_scale is not None:
        scale_map["ups."] = dec_upsamplers_lr_scale
    if dec_noise_convs_lr_scale is not None:
        scale_map["noise_convs."] = dec_noise_convs_lr_scale
    if dec_resblocks_lr_scale is not None:
        scale_map["resblocks."] = dec_resblocks_lr_scale

    if not scale_map:
        return None

    slow_groups = {}  # scale -> [params]
    fast_params = []

    for name, param in dec.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for prefix, scale in scale_map.items():
            if name.startswith(prefix):
                slow_groups.setdefault(scale, []).append(param)
                matched = True
                break
        if not matched:
            fast_params.append(param)

    for name, param in model.named_parameters():
        if not name.startswith("dec.") and param.requires_grad:
            fast_params.append(param)

    groups = []
    if fast_params:
        groups.append({"params": fast_params, "lr": base_lr})
    for scale, params in slow_groups.items():
        groups.append({"params": params, "lr": base_lr * scale})

    return groups if groups else None


def get_optimizers(
    net_g,
    config,
    optimizer_choice_g,
    custom_lr_g,
    use_custom_lr,
    total_epoch_count,
    train_loader
):
    lr_g = custom_lr_g if use_custom_lr else config.train.learning_rate_g
    num_batches = len(train_loader)

    g_param_groups = build_decoder_param_groups(net_g, lr_g)

    optim_g = _make_optimizer(net_g, optimizer_choice_g, lr_g, num_epochs=total_epoch_count, num_batches=num_batches, param_groups=g_param_groups)

    return optim_g


def apply_decoder_freezes(net_g, rank):
    """Apply layer freezing to the decoder / vocoder for fine-tuning."""
    model = net_g.module if hasattr(net_g, "module") else net_g
    dec = getattr(model, "dec", None)
    if dec is None:
        return

    freeze_map = [
        (freeze_dec_upsamplers,    "ups",          "Upsamplers"),
        (freeze_dec_noise_convs,   "noise_convs",  "Noise Convs"),
        (freeze_dec_resblocks,     "resblocks",     "ResBlocks"),
        (freeze_dec_conv_pre,      "conv_pre",      "Conv Pre"),
        (freeze_dec_conv_post,     "conv_post",     "Conv Post"),
        (freeze_dec_cond,          "cond",          "Speaker Cond"),
        (freeze_dec_source_module, "m_source",      "Source Module"),
    ]

    frozen_parts = []
    frozen_params = 0

    for should_freeze, attr_name, display_name in freeze_map:
        if should_freeze and hasattr(dec, attr_name):
            module = getattr(dec, attr_name)
            for param in module.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_parts.append(display_name)

    if rank == 0:
        if frozen_parts:
            print(f"[INIT] Decoder frozen: {', '.join(frozen_parts)} ({frozen_params:,} params)")
        else:
            print("[INIT] Decoder: no layers frozen")


def setup_models_for_training(net_g, device, device_id, n_gpus):
    net_g = net_g.to(device_id) if device.type == "cuda" else net_g.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)

    return net_g


def load_model_and_optimizer(config, pretrainG, vocoder, use_checkpointing, sample_rate, optimizer_choice_g, custom_lr_g, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing)
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

            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g = setup_models_for_training(net_g, device, device_id, n_gpus)

            # Apply decoder / vocoder layer freezes ( for fine-tuning )
            apply_decoder_freezes(net_g, rank)

            # Init the optimizers
            optim_g = get_optimizers(net_g, config, optimizer_choice_g, custom_lr_g, use_custom_lr, total_epoch_count, train_loader)

            # Load the model and optim states
            _, _, _, epoch_str, gradscaler_dict_g = load_checkpoint(g_checkpoint_path, net_g, optim_g, strict_load)

            if override_pretrain_lr:
                new_lr_for_pretrain = new_pretrain_lr
                for param_group in optim_g.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                print(f"[OVERRIDE] Pretrain LR Override: {new_lr_for_pretrain}")

            #epoch_str += 1
            #global_step = (epoch_str - 1) * len(train_loader)

            global_step = int(os.path.basename(g_checkpoint_path).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            print(f"[RESUMING] (G) & (D) at global_step: {global_step} and epoch count: {epoch_str - 1}")

        else:
            raise FileNotFoundError("No checkpoints found.")

    except FileNotFoundError:
    # If no checkpoints are available, using the Pretrains directly
        epoch_str = 1
        global_step = 0
        gradscaler_dict_g = {}

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


        # Load the models and optionally wrap with DDP
        net_g = setup_models_for_training(net_g, device, device_id, n_gpus)

        # Apply decoder / vocoder layer freezes ( for fine-tuning )
        apply_decoder_freezes(net_g, rank)

        # Init the optimizers
        optim_g = get_optimizers(net_g, config, optimizer_choice_g, custom_lr_g, use_custom_lr, total_epoch_count, train_loader)

    return net_g, optim_g, epoch_str, global_step, gradscaler_dict_g


def prepare_schedulers(
    optim_g, use_warmup, warmup_duration,
    use_lr_scheduler_g, lr_scheduler_g, exp_decay_gamma_g,
    total_epoch_count, epoch_str, global_step, train_loader
):
    warmup_scheduler_g = None
    scheduler_g = None

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

    if not use_warmup:
        for param_group in optim_g.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler_g:
        if lr_scheduler_g == "exp decay epoch":
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=exp_decay_gamma_g, last_epoch=scheduler_resume_epoch
            )
        elif lr_scheduler_g == "exp decay step":
            exp_decay_gamma_g_step = exp_decay_gamma_g ** (1.0 / num_batches_per_epoch)
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=exp_decay_gamma_g_step, last_epoch=scheduler_resume_step
            )
        elif lr_scheduler_g == "cosine annealing epoch":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_g, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch
            )

    return warmup_scheduler_g, scheduler_g


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
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed, optimizer_choice_g, from_scratch

    stopper = EarlyStopSignalHandler()

    if 'warmup_completed' not in globals():
        warmup_completed = False

    # Initial print / session info for console
    print_init_setup(
        warmup_duration,
        rank,
        use_warmup,
        config,
        optimizer_choice_g,
        lr_scheduler_g,
        exp_decay_gamma_g,
        use_kl_annealing,
        kl_annealing_cycle_duration,
        spectral_loss,
    )

    # Initial setup
    setup_env_and_distr(rank, n_gpus, device, device_id, config)

    # Dataloading and loaders preparation
    train_loader = prepare_dataloaders(config, n_gpus, rank, batch_size)

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
        fn_spectral_loss2 = MultiScaleSTFTLoss()
    else:
        print("ERROR: Chosen spectral loss is undefined. Exiting.")
        sys.exit(1)


    # Loading of model and optim
    net_g, optim_g, epoch_str, global_step, gradscaler_dict_g = load_model_and_optimizer(
        config,
        pretrainG,
        vocoder,
        use_checkpointing,
        sample_rate,
        optimizer_choice_g,
        custom_lr_g,
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
    if pretrainG in ["", "None"] or force_from_scratch:
        from_scratch = True
        if rank == 0:
            print("[INIT] No pretrains used: Average loss disabled!")

    # Prepare the schedulers
    warmup_scheduler_g, scheduler_g = prepare_schedulers(
        optim_g,
        use_warmup,
        warmup_duration,
        use_lr_scheduler_g, 
        lr_scheduler_g,
        exp_decay_gamma_g,
        total_epoch_count,
        epoch_str,
        global_step,
        train_loader
    )

    # Hann window for stft ( for RingFormer only. )
    hann_window = torch.hann_window(config.model.gen_istft_n_fft).to(device) if vocoder in ["RingFormer_v1", "RingFormer_v2"] else None

    # GradScaler for FP16 training
    gradscaler_g = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))

    if len(gradscaler_dict_g) > 0:
        gradscaler_g.load_state_dict(gradscaler_dict_g)
        print("[INIT] Loading G gradscaler state dicts")
    else:
        print("[INIT] G gradscaler state dict not found - Fresh initialization")

    # Reference sample for live-infer
    reference = get_reference_sample(train_loader, device, config)

    # Cache for training with " cache " enabled
    cache = []

    for epoch in range(epoch_str, total_epoch_count + 1):
        should_stop = training_loop(
            rank,
            epoch,
            config,
            net_g,
            optim_g,
            scheduler_g,
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
            gradscaler_g,
            fn_spectral_loss2,
            hann_window,
            stopper=stopper,
        )

        if use_warmup and epoch <= warmup_duration:
            if warmup_scheduler_g:
                warmup_scheduler_g.step()

            # Logging of finished warmup
            if epoch == warmup_duration:
                warmup_completed = True
                print(f"    ██████  Warmup completed at epochs: {warmup_duration}")
                print(f"    ██████  LR G: {optim_g.param_groups[0]['lr']}")

                if lr_scheduler_g == "exp decay epoch":
                    print(f"    ██████  Starting (G) per-epoch exponential lr decay with gamma of {exp_decay_gamma_g}")
                elif lr_scheduler_g == "cosine annealing epoch":
                    print("    ██████  Starting (G) per-epoch cosine annealing scheduler " )


        if use_lr_scheduler_g and (not use_warmup or warmup_completed):
            if lr_scheduler_g in ["exp decay epoch", "cosine annealing epoch"]:
                scheduler_g.step()

def training_loop(
    rank,
    epoch,
    config,
    net,
    optim,
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
    gradscaler_g,
    fn_spectral_loss2=None,
    hann_window=None,
    stopper=None,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        config (object): Configuration object containing training parameters.
        net: generator model net_g
        optim: optimizer optim_g
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
        gradscaler_g: gradscaler for fp16 - Used for Generator
        hann_window: hann window used for RingFormer
    """
    global global_step, warmup_completed, use_lr_scheduler_g, lr_scheduler_g, use_warmup, use_best_step

    net_g = net
    optim_g = optim
    scheduler_g = schedulers if schedulers is not None else None

    train_loader = train_loader if train_loader is not None else None
    train_loader.batch_sampler.set_epoch(epoch)

    if writers is not None:
        writer = writers[0]

    # Best in-epoch step tracking
    if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam") and use_best_step:
        use_best_step = False
        if rank == 0:
            print("[ ATTENTION ] Best in-epoch step disabled ~ Cannot be used alongside Schedule-Free Optimizers.")
    if use_best_step:
        best_loss_g = float('inf')
        best_state_dict_g = None
        live_sd_g = None

    net_g.train()

    if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
        optim_g.train()

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
            tensor_count = 3
        else:
            tensor_count = 2
        epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        num_batches_in_epoch = 0

    avg_rolling_cache = {
        "loss_spectral": deque(maxlen=rolling_loss_steps),
        "loss_kl": deque(maxlen=rolling_loss_steps),
        "grad_norm_g": deque(maxlen=rolling_loss_steps),
    }
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        avg_rolling_cache["loss_sd"] = deque(maxlen=rolling_loss_steps)

    kl_std_cache = deque(maxlen=rolling_loss_steps)
    last_kl_per_dim = None

    use_amp = config.train.fp16_run and device.type == "cuda"

    with tqdm(total=len(train_loader), leave=False, initial=start_batch_idx) as pbar:
        for batch_idx, info in data_iterator:

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


            # Batch unpacking ( Main )
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid) = info


            # Generator main forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)

                # Vocoder-dependent unpacking
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    y_hat, ids_slice, x_mask, z_mask, vae_parts, (mag, _) = model_output
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
                    # MS-STFT
                    loss_ms_stft = fn_spectral_loss2(y_hat.float(), y.float()) * 1.0
                    # Loss
                    loss_spectral = loss_l1_mel + loss_ms_stft


                # Kl annealing handler
                if use_kl_annealing:
                    annealing_cycle_steps = len(train_loader) * kl_annealing_cycle_duration
                    kl_beta = 0.5 * (1 - math.cos((global_step % annealing_cycle_steps) * (math.pi / annealing_cycle_steps)))
                else:
                    kl_beta = 1.0

                # KL ( Kullback–Leibler divergence ) loss
                loss_kl = kl_loss_fb(z_p, logs_q, m_p, logs_p, z_mask, z_p2, free_bits) * config.train.c_kl 
                #loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl


                # KL diagnostic: per-dim std (raw, without free_bits clamp)
                with torch.no_grad():
                    if z_p2 is not None:
                        raw_kl = (logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
                                  + logs_p - logs_q - 0.5 + 0.5 * ((z_p2 - m_p) ** 2) * torch.exp(-2 * logs_p)) * 0.5
                    else:
                        raw_kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
                    raw_kl_per_dim = (raw_kl * z_mask).sum(dim=(0, 2)) / z_mask.sum(dim=(0, 2)).clamp(min=1)
                    kl_std_cache.append(raw_kl_per_dim.std().item())
                    last_kl_per_dim = raw_kl_per_dim.detach()

                # RingFormer related
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    loss_magnitude = torch.nn.functional.l1_loss(mag, target_magnitude)
                    loss_phase = phase_loss(y_stft, y_hat_stft)
                    loss_sd = (loss_magnitude + loss_phase) * 0.7


                # Total generator loss
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    loss_gen_total = loss_spectral + loss_kl * kl_beta + loss_sd
                else:
                    loss_gen_total = loss_spectral + loss_kl * kl_beta

            # Generator backward and update:
            optim_g.zero_grad(set_to_none=True)
            if train_dtype == torch.float16:
                gradscaler_g.scale(loss_gen_total).backward() # Scale and backward of the loss
                gradscaler_g.unscale_(optim_g) # Unscale
                scale_g = gradscaler_g.get_scale() # To retrieve current gradscaler's scaling
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=grad_clip_value_g) # Grad clipping
                gradscaler_g.step(optim_g) # Optim step
                gradscaler_g.update() # Scaler update, to prepare the scaling for the next iteration
                skip_lr_sched_g = (scale_g > gradscaler_g.get_scale())
            else:
                loss_gen_total.backward() # Loss backward
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=grad_clip_value_g) # Grad clipping
                optim_g.step() # Optim step
                skip_lr_sched_g = False


            # Track best step in this epoch (FM + Spectral)
            if use_best_step:
                loss_val = (loss_spectral).detach()
                if loss_val < best_loss_g:
                    best_loss_g = loss_val
                    model_g = net_g.module if hasattr(net_g, "module") else net_g
                    best_state_dict_g = {k: v.detach().clone() for k, v in model_g.state_dict().items()}


            # Per step exp lr decay for generator
            if not skip_lr_sched_g: # We skip lr scheduler step if there were nans / infs due to gradscaler's scaling.
                if use_lr_scheduler_g and (not use_warmup or warmup_completed) and lr_scheduler_g == "exp decay step":
                    scheduler_g.step()



            if not from_scratch:
                # Loss accumulation for epoch-avg
                epoch_loss_tensor[0].add_(loss_spectral.detach())
                epoch_loss_tensor[1].add_(loss_kl.detach())

                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    epoch_loss_tensor[3].add_(loss_sd.detach())

            # Loss accumulation for rolling-avg

            # Losses:
            avg_rolling_cache["loss_spectral"].append(loss_spectral.detach())
            avg_rolling_cache["loss_kl"].append(loss_kl.detach())
            if "loss_sd" in avg_rolling_cache:
                avg_rolling_cache["loss_sd"].append(loss_sd.detach())

            # Gradients:
            if torch.isfinite(grad_norm_g):
                avg_rolling_cache["grad_norm_g"].append(grad_norm_g)
            else:
                writer.add_scalar("Grad_Norm_Diag/G_Skipped", 1, global_step)


            if rank == 0 and global_step % rolling_loss_steps == 0:
                scalar_dict_rolling = {}

                # Learning rate retrieval for rolling logging
                if from_scratch:
                    scalar_dict_rolling.update({
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

                # KL diagnostics (diag tab)
                if len(kl_std_cache) > 0:
                    diag_scalar = sum(kl_std_cache) / len(kl_std_cache)
                    summarize(writer=writer, global_step=global_step, scalars={"diag/kl_std": diag_scalar})
                    writer.add_histogram("diag/kl_per_dim_hist", last_kl_per_dim.cpu(), global_step)

                flush_writer(writer, rank)

            if from_scratch and pretrain_preview and rank == 0 and global_step % pretrain_preview_interval == 0:
                print(f"    ██████  Generating pretrain-preview at step: {global_step}...")
                if optimizer_choice_g in ("AdamWScheduleFree", "RAdamScheduleFree"):
                    optim_g.eval()
                o = eval_infer(net_g, reference)
                if optimizer_choice_g in ("AdamWScheduleFree", "RAdamScheduleFree"):
                    optim_g.train()
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

            # Step-based G checkpoint save
            if save_g_steps > 0 and rank == 0 and global_step % save_g_steps == 0:
                g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
                if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
                    optim_g.eval()
                save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path, gradscaler_g)
                if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
                    optim_g.train()

            if early_stopper(
                stopper, rank, global_step, epoch, architecture, 
                net_g, optim_g, config, 
                experiment_dir, gradscaler_g, save_weight_models,
                model_name, vocoder, n_gpus
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

        # Used for tensorboard mel charts
        y_mel = commons.slice_segments(mel, ids_slice, config.train.segment_size // config.data.hop_length, dim=3) # slice/mel_org
        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype) # slice/mel_gen

        # Learning rate retrieval for avg-epoch variation:
        lr_g = optim_g.param_groups[0]["lr"]

        # At each epoch completion
        if global_step % len(train_loader) == 0 and not from_scratch:

            # Calculate the avg epoch loss:
            avg_epoch_loss = epoch_loss_tensor / num_batches_in_epoch

            # metrics dict
            scalar_dict_avg = {
            "loss_avg/loss_spectral": avg_epoch_loss[0].item(),
            "loss_avg/loss_kl": avg_epoch_loss[1].item(),
            "learning_rate/lr_g": lr_g,
            }
            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                scalar_dict_avg.update({"loss_avg/loss_sd": avg_epoch_loss[3].item()})

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

            # Swap to best-step weights for eval_infer preview
            if use_best_step and best_state_dict_g is not None:
                model_g = net_g.module if hasattr(net_g, "module") else net_g
                live_sd_g = {k: v.detach().clone() for k, v in model_g.state_dict().items()}
                model_g.load_state_dict(best_state_dict_g)

            # Inferencing on reference sample


            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam") and not (use_best_step and best_state_dict_g is not None):
                optim_g.eval()
            o = eval_infer(net_g, reference)
            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam") and not (use_best_step and best_state_dict_g is not None):
                optim_g.train()
            audio_dict = {f"gen/audio_{epoch}e_{global_step}s": o[0, :, :]} # Eval-infer samples

            # Restore live weights immediately ~ checkpoint saving stays raw
            if use_best_step and live_sd_g is not None:
                model_g.load_state_dict(live_sd_g)
                live_sd_g = None

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

            # Switch to eval mode for Schedule-Free optim before saving (uses averaged params)
            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
                optim_g.eval()

            # Save Generator checkpoint (live weights; averaging was restored above)
            save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path, gradscaler_g)

            # Switch back to train mode after saving
            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
                optim_g.train()

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
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            ckpt = best_state_dict_g if (use_best_step and best_state_dict_g is not None) else model_g.state_dict()

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
