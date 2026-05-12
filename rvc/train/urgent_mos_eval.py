"""
urgent_mos_eval.py
==================
Thin integration layer that runs URGENT-MOS absolute and comparative
speech-quality evaluation inside the RVC training loop and writes the
results to TensorBoard.

Usage (from train.py):
    from urgent_mos_eval import log_urgent_mos

    log_urgent_mos(
        writer       = writer,
        global_step  = global_step,
        gen_audio    = o,              # [1, 1, T] float tensor from eval_infer
        gen_sr       = config.data.sample_rate,
        ref_wav_path = mos_comparison_file,  # path to a sliced_audios file
        device       = device,
    )
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URGENT-MOS model — loaded once, cached here for the lifetime of the process
# ---------------------------------------------------------------------------
_mos_model: Optional[torch.nn.Module] = None
_mos_device: Optional[str] = None

URGENT_MOS_HF_REPO = "urgent-challenge/urgent-mos-f1c1m5dcorpus"
TARGET_SR = 16_000   # URGENT-MOS always wants 16 kHz


def _get_model(device: torch.device) -> torch.nn.Module:
    """Lazy-load and cache the URGENT-MOS model."""
    global _mos_model, _mos_device

    device_str = str(device)
    if _mos_model is not None and _mos_device == device_str:
        return _mos_model

    # Ensure the bundled urgent_mos package is importable from the repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from urgent_mos.utils import load_model_from_checkpoint  # type: ignore

    print(f"[URGENT-MOS] Loading model from HuggingFace: {URGENT_MOS_HF_REPO}")
    _mos_model = load_model_from_checkpoint(URGENT_MOS_HF_REPO, device=device_str)
    _mos_model.eval()
    _mos_device = device_str
    print(f"[URGENT-MOS] Model ready on {device_str}")
    return _mos_model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _to_16k_mono(audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Convert any shape tensor to a 1-D float waveform at TARGET_SR=16000 Hz."""
    if audio.dim() == 3:          # [B, C, T]
        audio = audio[0]           # → [C, T]
    if audio.dim() == 2:
        audio = audio.mean(0)      # → [T]  (mono mix)
    audio = audio.float().cpu()
    if orig_sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, orig_sr, TARGET_SR)
    return audio


def _load_ref_audio(ref_wav_path: str) -> tuple[torch.Tensor, int]:
    """Load a reference wav and return (1D float tensor @ 16 kHz, TARGET_SR)."""
    wav, sr = torchaudio.load(ref_wav_path)
    audio = _to_16k_mono(wav, sr)
    return audio, TARGET_SR


# ---------------------------------------------------------------------------
# Core eval functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_absolute(
    model: torch.nn.Module,
    gen_audio_1d: torch.Tensor,   # 1-D float @ 16 kHz
) -> dict[str, float]:
    """Run absolute MOS prediction; returns {metric_name: float}."""
    model_device = next(model.parameters()).device
    audio_on_dev = gen_audio_1d.to(model_device)

    scores = model.predict_absolute_scores([audio_on_dev])  # list of 1-D tensors
    return {k: float(v[0]) for k, v in scores.items()}


@torch.inference_mode()
def _run_comparative(
    model: torch.nn.Module,
    ref_audio_1d: torch.Tensor,   # 1-D float @ 16 kHz  (the "A" sample)
    gen_audio_1d: torch.Tensor,   # 1-D float @ 16 kHz  (the "B" sample)
) -> dict[str, float]:
    """Run comparative CMOS prediction; positive ⇒ ref preferred over gen.
    Returns {metric_name: float}."""
    model_device = next(model.parameters()).device
    ref_on_dev = ref_audio_1d.to(model_device)
    gen_on_dev = gen_audio_1d.to(model_device)

    reg_scores, _ = model.predict_comparative_scores_from_audio_pairs(
        [ref_on_dev],   # A
        [gen_on_dev],   # B
    )
    return {k: float(v[0]) for k, v in reg_scores.items()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_urgent_mos(
    writer,
    global_step: int,
    gen_audio: torch.Tensor,       # raw output of eval_infer: [1, 1, T]
    gen_sr: int,                   # sample rate of gen_audio
    ref_wav_path: Optional[str],   # path to reference wav from sliced_audios
    device: torch.device,
) -> None:
    """
    Run URGENT-MOS in both absolute and comparative modes, then log to
    TensorBoard.

    Absolute  → Metric/UrgentMOS_Abs/<metric>
    Comparative → Metric/UrgentMOS_Comp/<metric>   (positive = ref preferred)

    Safe to call only from rank==0.
    """
    try:
        model = _get_model(device)
    except Exception as exc:
        logger.warning(f"[URGENT-MOS] Could not load model, skipping eval: {exc}")
        return

    # Prepare the generated audio at 16 kHz mono
    gen_1d = _to_16k_mono(gen_audio, gen_sr)

    # ── Absolute MOS ────────────────────────────────────────────────────────
    try:
        abs_scores = _run_absolute(model, gen_1d)
        for metric, score in abs_scores.items():
            tag = f"Metric/UrgentMOS_Abs/{metric}"
            writer.add_scalar(tag, score, global_step)
        print(f"[URGENT-MOS] Absolute → " +
              ", ".join(f"{k}: {v:.4f}" for k, v in abs_scores.items()))
    except Exception as exc:
        logger.warning(f"[URGENT-MOS] Absolute inference failed: {exc}")

    # ── Comparative CMOS ─────────────────────────────────────────────────────
    if ref_wav_path and os.path.isfile(ref_wav_path):
        try:
            ref_1d, _ = _load_ref_audio(ref_wav_path)
            comp_scores = _run_comparative(model, ref_1d, gen_1d)
            for metric, score in comp_scores.items():
                tag = f"Metric/UrgentMOS_Comp/{metric}"
                writer.add_scalar(tag, score, global_step)
            print(f"[URGENT-MOS] Comparative (ref={os.path.basename(ref_wav_path)}) → " +
                  ", ".join(f"{k}: {v:.4f}" for k, v in comp_scores.items()))
        except Exception as exc:
            logger.warning(f"[URGENT-MOS] Comparative inference failed: {exc}")
    else:
        if ref_wav_path:
            logger.warning(f"[URGENT-MOS] Reference file not found, skipping comparative: {ref_wav_path}")
        else:
            logger.warning("[URGENT-MOS] No reference file set, skipping comparative.")
