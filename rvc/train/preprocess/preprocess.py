import os
import sys
import time
from scipy import signal
from scipy.io import wavfile
import numpy as np
import concurrent.futures
from tqdm import tqdm
import json
from distutils.util import strtobool
import librosa
import multiprocessing
import shutil
import soundfile as sf
import io
from fractions import Fraction

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio, load_audio_ffmpeg
from rvc.train.preprocess.slicer import Slicer
from rvc.train.preprocess.smartcutter.inference import SmartCutterInterface

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"
FLAC_COMPRESSION_LEVEL = 5 / 8 # FLAC level 5, libsndfile uses 0.0 - 1.0 range for compression level

# Large-file streaming: files longer than this are read in blocks instead of all at once
LARGE_FILE_THRESHOLD_SECS = 3600   # 1 hour
STREAM_BLOCK_SECS         = 1800    # process 30 minutes per block
STREAM_OVERLAP_SECS       = 2.0    # overlap between consecutive blocks (avoids boundary cut-off)


def secs_to_samples(secs, sr):
    """Return an *exact* integer number of samples for `secs` seconds at `sr` Hz.
       Raises if the result is not an integer (prevents float drift)."""
    frac = Fraction(str(secs)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{secs}s × {sr}Hz is not an integer sample count")
    return frac.numerator

def save_audio(path: str, name: str, sample_rate: int, format: str, audio: np.ndarray):
    """
    Save audio to file.
    Args:
        path: Path to a directory where the audio file will be saved.
        name: Name of the audio file without the extension.
        sample_rate: Sample rate of the audio file.
        format: Format of the audio file (WAV or FLAC).
        audio: Audio data array.
    """
    if format.lower() == "flac":
        memory_file = io.BytesIO()
        sf.write(
            memory_file,
            audio,
            sample_rate,
            format="FLAC",
            subtype="PCM_24",
            compression_level=FLAC_COMPRESSION_LEVEL
        )
        memory_file.seek(0)
        with open(os.path.join(path, f"{name}.flac"), "wb") as f:
            f.write(memory_file.read())
    else:
        wavfile.write(
            os.path.join(path, f"{name}.wav"),
            sample_rate,
            audio.astype(np.float32),
        )

class PreProcess:
    def __init__(self, sr: int, exp_dir: str):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr)
        self.exp_dir = exp_dir

        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_duration(path: str):
        """Return file duration in seconds without loading audio, or None on failure."""
        try:
            return sf.info(path).duration
        except Exception:
            return None

    def _load_audio_with_fallback(self, path: str, loading_resampling: str) -> np.ndarray:
        """
        Load audio using the requested backend.
        If that fails, fall back to librosa.load automatically.
        """
        try:
            if loading_resampling == "librosa":
                return load_audio(path, self.sr)
            else:
                return load_audio_ffmpeg(path, self.sr)
        except Exception as primary_err:
            logger.warning(
                f"Primary loader ({'librosa/SoXr' if loading_resampling == 'librosa' else 'ffmpeg'}) "
                f"failed for '{path}': {primary_err}. Falling back to librosa.load …"
            )
            try:
                audio, _ = librosa.load(path, sr=self.sr, mono=True)
                return audio
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Both primary loader and librosa fallback failed for '{path}'. "
                    f"Librosa error: {fallback_err}"
                ) from fallback_err

    def _stream_audio_blocks(self, path: str):
        """
        Generator that yields overlapping float32 mono audio blocks at self.sr.
        Designed for very large files (> LARGE_FILE_THRESHOLD_SECS) so the whole
        file is never held in RAM at once.

        Yields: (block_audio, is_first_block)
        """
        try:
            info = sf.info(path)
        except Exception as e:
            raise RuntimeError(f"soundfile could not read metadata for '{path}': {e}") from e

        native_sr   = info.samplerate
        block_frames   = int(native_sr * STREAM_BLOCK_SECS)
        overlap_frames = int(native_sr * STREAM_OVERLAP_SECS)

        overlap_sr_samples = int(self.sr * STREAM_OVERLAP_SECS)

        prev_tail_native = None  # raw (native SR) tail carried between blocks
        is_first = True

        for raw_block in sf.blocks(path, blocksize=block_frames, dtype="float32", always_2d=False):
            # Convert to mono
            if raw_block.ndim > 1:
                raw_block = raw_block.mean(axis=1)

            # Prepend the tail of the previous block so the slicer sees context
            if prev_tail_native is not None:
                combined = np.concatenate([prev_tail_native, raw_block])
            else:
                combined = raw_block

            # Save tail for next iteration before we modify combined
            prev_tail_native = raw_block[-overlap_frames:].copy() if len(raw_block) >= overlap_frames else raw_block.copy()

            # Resample to target SR
            if native_sr != self.sr:
                block_resampled = librosa.resample(combined, orig_sr=native_sr, target_sr=self.sr, res_type=RES_TYPE)
            else:
                block_resampled = combined.copy()

            # Trim the prepended overlap from the *output* (except for the very first block)
            # so we don't double-process the overlap region.
            if not is_first:
                block_resampled = block_resampled[overlap_sr_samples:]

            yield block_resampled, is_first
            is_first = False

    def _process_audio_streaming(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
        dataset_format: str,
        normalization_mode: str,
    ) -> float:
        """
        Memory-efficient processing pipeline for files longer than LARGE_FILE_THRESHOLD_SECS.
        Reads the file in overlapping blocks and runs each block through the normal
        effects / slicing / saving pipeline without ever holding the full file in RAM.
        """
        if noise_reduction:
            logger.warning(
                "Noise reduction with streaming large files processes each block "
                "independently — the noise profile is estimated per block."
            )

        audio_length = 0.0
        idx1 = 0   # global segment counter across all blocks

        for block_audio, _is_first in self._stream_audio_blocks(path):
            block_duration = librosa.get_duration(y=block_audio, sr=self.sr)
            audio_length += block_duration

            # Effects
            if process_effects:
                block_audio = signal.lfilter(self.b_high, self.a_high, block_audio)
            if noise_reduction:
                import noisereduce as nr
                block_audio = nr.reduce_noise(
                    y=block_audio, sr=self.sr, prop_decrease=reduction_strength
                )

            # Normalization (pre modes only — post modes are applied globally after all workers finish)
            if normalization_mode == "pre_rms":
                eps = 1e-9
                target_rms = 10 ** (-18.0 / 20)
                headroom   = 10 ** (-0.5  / 20)
                silence_thresh = 10 ** (-40.0 / 20)
                mask = np.abs(block_audio) > silence_thresh
                if np.any(mask):
                    rms  = np.sqrt(np.mean(block_audio[mask] ** 2) + eps)
                    gain = target_rms / rms
                else:
                    gain = 1.0
                block_audio = block_audio * gain
                peak = np.abs(block_audio).max()
                if peak > headroom:
                    block_audio = block_audio / peak * headroom
                block_audio = block_audio.astype(np.float32)
            elif normalization_mode == "pre_peak":
                peak = np.max(np.abs(block_audio))
                if peak > 0:
                    block_audio = (block_audio / peak * MAX_AMPLITUDE).astype(np.float32)

            # Slicing
            if cut_preprocess == "Skip":
                self.process_audio_segment(block_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                idx1 += 1
            elif cut_preprocess == "Simple":
                # simple_cut manages its own internal slice_idx per call; use idx1 as the
                # outer index so filenames remain unique across blocks.
                self.simple_cut(block_audio, sid, idx1, chunk_len, overlap_len, loading_resampling, dataset_format)
                idx1 += 1
            elif cut_preprocess == "Automatic":
                for audio_segment in self.slicer.slice(block_audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                            idx1 += 1
                            break

        return audio_length

    # ------------------------------------------------------------------

    def process_audio_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        loading_resampling: str,
        dataset_format: str
    ):
        # Saving slices for GroundTruth ( 'sliced_audios' dir )
        save_audio(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}", self.sr, dataset_format, audio)

        # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
        if loading_resampling == "librosa":
            chunk_16k = librosa.resample(
                audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        else: # ffmpeg
            chunk_16k = load_audio_ffmpeg(
                audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
            )

        # Saving slices for 16khz ( 'sliced_audios_16k' dir )
        save_audio(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}", SAMPLE_RATE_16K, dataset_format, chunk_16k)


    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
        dataset_format: str
    ):
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        stride = chunk_len_smpl - secs_to_samples(overlap_len, self.sr)

        slice_idx = 0
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_len_smpl]

            # If the last slice's below 3 seconds, we're padding it to 3 secs.
            if len(chunk) < chunk_len_smpl:
                padding_needed = chunk_len_smpl - len(chunk)
                if len(chunk) > self.sr * 1.0: 
                    padding = np.zeros(padding_needed, dtype=np.float32)
                    chunk = np.concatenate((chunk, padding))
                    logger.info(f"Padded final slice {sid}_{idx0}_{slice_idx} with {padding_needed} samples.")
                else:
                    break

            # Saving slices
            save_audio(self.gt_wavs_dir, f"{sid}_{idx0}_{slice_idx}", self.sr, dataset_format, chunk)

            # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
            if loading_resampling == "librosa":
                chunk_16k = librosa.resample(
                    chunk, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
                )
            else: # ffmpeg
                chunk_16k = load_audio_ffmpeg(
                    chunk, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
                )
            # Saving slices for 16khz ( 'sliced_audios_16k' dir )
            save_audio(self.wavs16k_dir, f"{sid}_{idx0}_{slice_idx}", SAMPLE_RATE_16K, dataset_format, chunk_16k)

            slice_idx += 1
            i += stride

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
        dataset_format: str,
        normalization_mode: str = "none",
    ):
        audio_length = 0
        try:
            # -----------------------------------------------------------------
            # Check duration first (no audio loaded yet) to choose load strategy
            # -----------------------------------------------------------------
            file_duration = self._get_file_duration(path)
            is_large_file = file_duration is not None and file_duration > LARGE_FILE_THRESHOLD_SECS

            if is_large_file:
                logger.info(
                    f"Large file detected ({format_duration(file_duration)}): '{path}'. "
                    f"Using memory-efficient streaming mode ({STREAM_BLOCK_SECS // 60}-min blocks)."
                )
                audio_length = self._process_audio_streaming(
                    path, idx0, sid, cut_preprocess, process_effects, noise_reduction,
                    reduction_strength, chunk_len, overlap_len, loading_resampling,
                    dataset_format, normalization_mode,
                )
                return audio_length

            # -----------------------------------------------------------------
            # Normal (non-streaming) path for files under the threshold
            # -----------------------------------------------------------------

            # Load audio — fall back to librosa if the primary loader fails
            audio = self._load_audio_with_fallback(path, loading_resampling)

            # Getting the length
            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            # Processing, Filtering, Noise reduction
            if process_effects:
                audio = signal.lfilter(self.b_high, self.a_high, audio)
            if noise_reduction:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=reduction_strength)

            # Slicing approach
            if cut_preprocess == "Skip":
                self.process_audio_segment(audio, sid, idx0, 0, loading_resampling, dataset_format)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, loading_resampling, dataset_format)
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                            idx1 += 1
                            break
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            raise e
        return audio_length

def _process_audio_worker(args):
    (
        path,
        idx0,
        sid,
        sr,
        exp_dir,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        loading_resampling,
        dataset_format,
        normalization_mode,
    ) = args
    pp = PreProcess(sr, exp_dir)
    return pp.process_audio(
        path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        loading_resampling,
        dataset_format,
        normalization_mode,
    )

def _apply_post_norm(audio: np.ndarray, sr: int, mode: str) -> np.ndarray:
    """
    Dispatch to the correct norm function.
    All three operate on float32/float64 arrays and return float32.
    """
    if mode == "post_rms":
        # RMS normalization - Consistent loudness targeting -18 dBFS RMS
        eps = 1e-9
        target_rms = 10 ** (-18.0 / 20)
        headroom = 10 ** (-0.5  / 20)
        silence_thresh = 10 ** (-40.0 / 20)
        mask = np.abs(audio) > silence_thresh
        if np.any(mask):
            rms = np.sqrt(np.mean(audio[mask] ** 2) + eps)
            gain = target_rms / rms
        else:
            gain = 1.0
        audio2 = audio * gain
        peak = np.abs(audio2).max()
        if peak > headroom:
            audio2 = audio2 / peak * headroom
        return audio2.astype(np.float32)

    elif mode == "post_peak_rvc":
        # Classic RVC normalization - soft peak blend (MAX_AMPLITUDE * ALPHA)
        a_max = np.abs(audio).max()
        if a_max <= 0:
            return audio.astype(np.float32)
        return ((audio / a_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio).astype(np.float32)

    elif mode == "post_peak":
        # Simple peak normalization to 0.95
        peak = np.max(np.abs(audio))
        if peak > 0:
            return (audio / peak * 0.95).astype(np.float32)
        return audio.astype(np.float32)

    return audio.astype(np.float32)


def _process_and_save_worker(args):
    """Shared post-norm worker"""
    file_name, gt_wavs_dir, wavs16k_dir, mode = args
    try:
        stem, ext = file_name.split(".")[0], file_name.split(".")[1]

        gt_audio, gt_sr = sf.read(os.path.join(gt_wavs_dir, file_name))
        save_audio(gt_wavs_dir, stem, gt_sr, ext, _apply_post_norm(gt_audio, gt_sr, mode))

        k16_audio, k16_sr = sf.read(os.path.join(wavs16k_dir, file_name))
        save_audio(wavs16k_dir, stem, k16_sr, ext, _apply_post_norm(k16_audio, k16_sr, mode))
    except Exception as e:
        logger.error(f"Error normalizing {file_name} ({mode}): {e}")
        raise e

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def save_dataset_duration(file_path, dataset_duration):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
    }
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def cleanup_dirs(exp_dir):
    gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
    wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
    logger.info("Cleaning up partially processed audio directories if they exist...")
    if os.path.exists(gt_wavs_dir):
        shutil.rmtree(gt_wavs_dir)
        logger.info(f"Deleted directory: {gt_wavs_dir}")
    if os.path.exists(wavs16k_dir):
        shutil.rmtree(wavs16k_dir)
        logger.info(f"Deleted directory: {wavs16k_dir}")


def run_smart_cutter_stage(input_root, exp_dir, sr):
    """
    Stage 1: Sequential GPU processing.
    Reads from input_root, writes to 'smart_cut_temp' inside exp_dir.
    Returns the path to the NEW input root (the temp folder).
    """
    ckpt_dir = os.path.join(now_directory, r"rvc/models/smartcutter")
    output_root = os.path.join(exp_dir, "smart_cut_temp")
    os.makedirs(output_root, exist_ok=True)

    print("[SmartCutter] Starting .. this may take a bit ... ")
    print(f"[SmartCutter] Original Input: {input_root}")
    print(f"[SmartCutter] Temp Output: {output_root}")

    # Initialize Interface
    engine = SmartCutterInterface(sr, ckpt_dir)
    engine.load_model()

    files_to_process = []
    # Scan files
    for root, _, filenames in os.walk(input_root):
        for f in filenames:
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                full_path = os.path.join(root, f)

                # Replicate folder structure in temp dir
                rel_path = os.path.relpath(full_path, input_root)
                out_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                files_to_process.append((full_path, out_path))

    # Process Loop
    with tqdm(total=len(files_to_process), desc="SmartCutting") as pbar:
        for in_p, out_p in files_to_process:
            engine.process_file(in_p, out_p)
            pbar.update(1)

    # Cleanup GPU
    engine.unload()
    print("SmartCutter Stage Complete. Proceeding to Slicing...")

    return output_root


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    loading_resampling: str,
    use_smart_cutter: bool,
    dataset_format: str
):
    start_time = time.time()
    print(f"Normalization mode: {normalization_mode}")
    sc_engine = None
    if use_smart_cutter:
        try:
            ckpt_dir = os.path.join(now_directory, r"rvc/models/smartcutter")
            sc_engine = SmartCutterInterface(sr, ckpt_dir)
            sc_engine.load_model()
            logger.info("SmartCutter model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SmartCutter: {e}")
            return

    speaker_map = {}

    root_files = [f for f in os.listdir(input_root) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"))]
    if root_files:
        speaker_map[input_root] = [os.path.join(input_root, f) for f in root_files]

    for root, dirs, filenames in os.walk(input_root):
        if root == input_root:
            continue

        audio_files = [os.path.join(root, f) for f in filenames if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"))]
        if audio_files:
            speaker_map[root] = audio_files

    speaker_count = len(speaker_map)

    if speaker_count > 1:
        detected_sids = set()
        for folder_path in speaker_map.keys():
            if folder_path == input_root:
                detected_sids.add(0)
            else:
                try:
                    folder_name = os.path.basename(folder_path)
                    sid = int(folder_name.split('_')[0])
                    detected_sids.add(sid)
                except (ValueError, IndexError):
                    logger.error(f"FATAL: Folder '{folder_name}' is invalid for multi-speaker. "
                                 f"Folders must start with an integer (e.g., '0_name').")
                    sys.exit(1)

        expected_sids = set(range(speaker_count))
        if detected_sids != expected_sids:
            missing = sorted(list(expected_sids - detected_sids))
            logger.error(f"FATAL: Speaker IDs are not contiguous or missing 0. "
                         f"Detected: {sorted(list(detected_sids))}. Missing: {missing}")
            sys.exit(1)
        else:
            logger.info("Contiguity check passed.")

    logger.info(f"Found {speaker_count} speakers to process.")
    cleanup_dirs(exp_dir)


    total_audio_length = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        for speaker_dir, audio_paths in tqdm(speaker_map.items(), desc="Processing Speakers"):

            temp_speaker_dir = None
            current_batch_paths = []
            try:
                if speaker_dir == input_root:
                    sid = 0
                else:
                    folder_name = os.path.basename(speaker_dir)
                    sid_str = folder_name.split('_')[0] 
                    sid = int(sid_str)
            except (ValueError, IndexError):
                logger.warning(f"Folder '{os.path.basename(speaker_dir)}' does not start with a valid integer ID. Using SID 0.")
                sid = 0

            if use_smart_cutter and sc_engine:
                temp_speaker_dir = os.path.join(exp_dir, "smart_cut_temp", str(sid))
                os.makedirs(temp_speaker_dir, exist_ok=True)

                for file_path in audio_paths:
                    filename = os.path.basename(file_path)
                    out_path = os.path.join(temp_speaker_dir, filename)

                    sc_engine.process_file(file_path, out_path)
                    current_batch_paths.append(out_path)
            else:
                current_batch_paths = audio_paths

            arg_list = [
                (
                    f_path,
                    idx,
                    sid,
                    sr,
                    exp_dir,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    chunk_len,
                    overlap_len,
                    loading_resampling,
                    dataset_format,
                    normalization_mode,
                )
                for idx, f_path in enumerate(current_batch_paths)
            ]

            for result in pool.imap_unordered(_process_audio_worker, arg_list):
                if result:
                    total_audio_length += result

            if temp_speaker_dir and os.path.exists(temp_speaker_dir):
                shutil.rmtree(temp_speaker_dir)

    main_temp_dir = os.path.join(exp_dir, "smart_cut_temp")
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)
        
    if use_smart_cutter and sc_engine:
        sc_engine.unload()

    save_dataset_duration(os.path.join(exp_dir, "model_info.json"), total_audio_length)

    POST_NORM_MODES = {
        "post_rms":      "RMS Normalization",
        "post_peak_rvc": "Peak Normalization (RVC)",
        "post_peak":     "Peak Normalization",
    }

    if normalization_mode in POST_NORM_MODES:
        logger.info(f"Post Normalization: {POST_NORM_MODES[normalization_mode]}. Initiating...")
        gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")

        audio_files = sorted(f for f in os.listdir(gt_wavs_dir) if f.endswith((".wav", ".flac")))
        arg_list = [(f, gt_wavs_dir, wavs16k_dir, normalization_mode) for f in audio_files]

        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(_process_and_save_worker, arg_list),
                total=len(audio_files),
                desc=POST_NORM_MODES[normalization_mode]
            ))

    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds "
                f"on {format_duration(total_audio_length)} of audio.")

if __name__ == "__main__":
    if len(sys.argv) < 15:
        print("Usage: python preprocess.py <experiment_directory> <input_root> <sample_rate> <num_processes or 'none'> <cut_preprocess> <process_effects> <noise_reduction> <reduction_strength> <chunk_len> <overlap_len> <normalization_mode> <loading_resampling> <use_smart_cutter> <dataset_format>")
        sys.exit(1)
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]

    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)

    cut_preprocess = str(sys.argv[5])
    process_effects = bool(strtobool(sys.argv[6]))
    noise_reduction = bool(strtobool(sys.argv[7]))
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    loading_resampling = str(sys.argv[12])
    use_smart_cutter = bool(strtobool(sys.argv[13]))
    dataset_format = str(sys.argv[14])

    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        loading_resampling,
        use_smart_cutter,
        dataset_format
    )
