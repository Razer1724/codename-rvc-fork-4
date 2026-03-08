"""
F0 Curve Editor Component
=========================
Provides:
  - extract_f0_for_editor()   – runs the chosen f0 method on an audio file,
                                returns a JSON string for the canvas editor.
                                Results are cached in assets/f0editor_userdata/base_f0/
                                keyed by audio content hash + method, so the
                                same song is never re-extracted twice even if
                                its filename changes.
  - save_edited_f0_to_temp()  – converts editor JSON back to a CSV temp-file
                                in assets/f0editor_userdata/session/ (purged on
                                startup). Wrapped in F0TempFile so pipeline.py's
                                `hasattr(f0_file,"name")` check passes.
  - list_f0_presets()         – returns sorted list of saved preset names.
  - save_f0_preset(name, freqs_json) – save a freq array as a named preset CSV.
  - load_f0_preset(name)      – load a preset and return its freq list as JSON.
  - build_f0_editor_html()    – returns the full self-contained HTML/CSS/JS
                                string to drop into a gr.HTML component.
  - cleanup_old_temp_f0()     – prunes base_f0 cache files older than 24 h.

Directory layout
----------------
assets/f0editor_userdata/
  base_f0/   persistent per-song CSV caches (hash+method keyed). Never auto-deleted
             except by cleanup_old_temp_f0(). Stores times+freqs only;
             spectrogram is recomputed from audio on each load.
  session/   temporary edit CSVs for the pipeline. Wiped on every process start.
  f0_presets/ user-saved curve presets. One CSV per preset (raw freq values,
             one per line). Resampled to current song length on import.

New in this version
-------------------
* Preset system  – Export current curve as a named preset; Import stretches
  any saved preset to match the current song length.
* Spectrogram overlay  – 🎨 Spec toggle button, opacity slider.
* Rectangular selection – select tool draws a real freq×time rect.
* Y-axis zoom          – Alt+scroll.
* Tool shortcuts       – keys 1–8, hover-scoped to the editor panel.
"""

import os
import sys
import json
import time
import zlib
import struct
import base64
import hashlib
import shutil
import tempfile
import traceback
import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)

WINDOW       = 160
SAMPLE_RATE  = 16000
TIME_STEP_MS = WINDOW / SAMPLE_RATE * 1000   # 10 ms
F0_MIN       = 50
F0_MAX       = 1100

# ── Directory layout ──────────────────────────────────────────────────────────
# assets/f0editor_userdata/
#   base_f0/   — persistent per-song CSV caches (hash + method keyed).
#   session/   — temporary edit CSVs for the pipeline (purged on startup).
#   f0_presets/— user-saved curve presets (one CSV of raw freqs per preset).
# ─────────────────────────────────────────────────────────────────────────────
F0EDITOR_DIR    = os.path.join(now_dir, "assets", "f0editor_userdata")
F0_BASE_DIR     = os.path.join(F0EDITOR_DIR, "base_f0")
F0_SESSION_DIR  = os.path.join(F0EDITOR_DIR, "session")
F0_PRESETS_DIR  = os.path.join(F0EDITOR_DIR, "f0_presets")

# Hash chunk: first 1 MB is enough to uniquely identify audio content while
# keeping startup fast even for large files.
_HASH_CHUNK = 1 * 1024 * 1024  # 1 MB


def _audio_hash(audio_path: str) -> str:
    """SHA-256 of the first 1 MB of the file — fast, name-insensitive."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as fh:
        h.update(fh.read(_HASH_CHUNK))
    return h.hexdigest()[:16]   # 16 hex chars = 64-bit prefix, collision-safe


def _base_cache_path(audio_path: str, f0_method: str) -> str:
    """Return the .csv path in base/ for this audio+method combination."""
    key = f"{_audio_hash(audio_path)}_{f0_method}.csv"
    return os.path.join(F0_BASE_DIR, key)


def _purge_session_dir() -> None:
    """Delete all session CSV files. Called once at import time."""
    if os.path.isdir(F0_SESSION_DIR):
        shutil.rmtree(F0_SESSION_DIR, ignore_errors=True)
    os.makedirs(F0_SESSION_DIR, exist_ok=True)
    # Ensure the other persistent dirs exist
    os.makedirs(F0_BASE_DIR, exist_ok=True)
    os.makedirs(F0_PRESETS_DIR, exist_ok=True)


# Purge stale session files the moment this module loads (= Gradio startup).
_purge_session_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Thin wrapper so pipeline.py's  hasattr(f0_file, "name")  check passes
# ──────────────────────────────────────────────────────────────────────────────
class F0TempFile:
    """Wraps a temp CSV path so pipeline.py treats it like a file object."""
    def __init__(self, path: str):
        self.name = path


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plasma_colormap(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map a float32 array v (values 0..1) to R, G, B uint8 arrays using a
    plasma-inspired dark palette: black → indigo → violet → orange → yellow.
    Pure numpy – no matplotlib dependency.
    """
    keys = [0.0,  0.12, 0.30, 0.50, 0.70, 0.88, 1.0]
    r_   = [0,    15,   80,  180,  245,  254,  252]
    g_   = [0,     5,   10,   50,  130,  188,  255]
    b_   = [0,   100,  140,   90,   20,   44,  164]
    r = np.interp(v, keys, r_).astype(np.uint8)
    g = np.interp(v, keys, g_).astype(np.uint8)
    b = np.interp(v, keys, b_).astype(np.uint8)
    return r, g, b


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    """
    Encode an (H, W, 3) uint8 numpy array as a PNG using only stdlib (zlib,
    struct).  No PIL/Pillow required.
    """
    H, W = rgb.shape[:2]

    # Build raw scanlines: each row is prefixed with filter-type byte 0x00
    raw = bytearray()
    for y in range(H):
        raw += b'\x00'
        raw += rgb[y].tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

    png = (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(bytes(raw), 3))
        + chunk(b'IEND', b'')
    )
    return png


# ──────────────────────────────────────────────────────────────────────────────
# F0 extraction  (now also produces a spectrogram)
# ──────────────────────────────────────────────────────────────────────────────
def extract_f0_for_editor(audio_path: str, f0_method: str = "rmvpe") -> str | None:
    """
    Load *audio_path*, run the requested f0 extractor, and return a JSON string:

        {
          "times":           [<float seconds>, ...],
          "freqs":           [<float Hz; 0=unvoiced>, ...],
          "time_step_ms":    10.0,
          "duration":        <float seconds>,

          # spectrogram (present when computation succeeded)
          "spec_b64":        "<base64-encoded RGB PNG>",
          "spec_n_bins":     513,
          "spec_nyquist":    8000,
          "spec_frame_step": 1    # >1 when audio was time-downsampled
        }

    The PNG is W=spec_cols wide, H=spec_n_bins tall.
    Row 0 = Nyquist (high freq), last row = DC (0 Hz)  — matches canvas Y.
    Column k corresponds to f0 frame  k * spec_frame_step.

    Returns None on failure.
    """
    if not audio_path or not os.path.exists(audio_path):
        print(f"[F0Editor] Audio file not found: {audio_path}")
        return None

    # ── Base cache lookup ─────────────────────────────────────────────────────
    # The base cache stores only times+freqs as a plain CSV (one "t,f" line per
    # frame).  On a hit we skip the expensive neural extractor and just
    # recompute the spectrogram (cheap STFT) before returning full JSON.
    cached_times = None
    cached_freqs = None
    try:
        os.makedirs(F0_BASE_DIR, exist_ok=True)
        cache_path = _base_cache_path(audio_path, f0_method)
        if os.path.exists(cache_path):
            rows = []
            with open(cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    t_s, f_s = line.split(",", 1)
                    rows.append((float(t_s), float(f_s)))
            cached_times = [r[0] for r in rows]
            cached_freqs = [r[1] for r in rows]
    except Exception as cache_exc:
        cached_times = cached_freqs = None

    try:
        import librosa
        import torch
        from rvc.lib.predictors.f0 import RMVPE, CREPE, FCPE
        from rvc.configs.config import Config

        cfg    = Config()
        device = cfg.device

        # ── load & normalise (always needed — for spec even on cache hit) ──────
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        peak     = np.abs(audio).max() / 0.95
        if peak > 1:
            audio /= peak

        audio_pad = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
        p_len     = audio_pad.shape[0] // WINDOW

        # ── extract f0 (skipped on cache hit) ────────────────────────────────
        if cached_times is not None:
            times = cached_times
            freqs = cached_freqs
        else:
            if f0_method == "rmvpe":
                model = RMVPE(device=device, sample_rate=SAMPLE_RATE, hop_size=WINDOW)
                f0    = model.get_f0(audio_pad, filter_radius=0.03)
                del model
            elif f0_method in ("crepe", "crepe-tiny"):
                ver   = "full" if f0_method == "crepe" else "tiny"
                model = CREPE(device=device, sample_rate=SAMPLE_RATE, hop_size=WINDOW)
                f0    = model.get_f0(audio_pad, F0_MIN, F0_MAX, p_len, ver)
                del model
            elif f0_method == "fcpe":
                model = FCPE(device=device, sample_rate=SAMPLE_RATE, hop_size=WINDOW)
                f0    = model.get_f0(audio_pad, p_len, filter_radius=0.006,
                                     test_time_augmentation=True)
                del model
            else:
                print(f"[F0Editor] Unknown f0_method '{f0_method}', falling back to rmvpe")
                model = RMVPE(device=device, sample_rate=SAMPLE_RATE, hop_size=WINDOW)
                f0    = model.get_f0(audio_pad, filter_radius=0.03)
                del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            f0    = np.asarray(f0, dtype=np.float32)[:p_len]
            times = [round(i * TIME_STEP_MS / 1000.0, 6) for i in range(len(f0))]
            freqs = f0.tolist()

            # ── Persist to base cache as CSV ──────────────────────────────────
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    for t, fr in zip(times, freqs):
                        fh.write(f"{t},{fr}\n")
                print(f"[F0Editor] Cached to: {os.path.basename(cache_path)}")
            except Exception as write_exc:
                print(f"[F0Editor] Cache write failed (non-fatal): {write_exc}")

        result = {
            "times":        times,
            "freqs":        freqs,
            "time_step_ms": TIME_STEP_MS,
            "duration":     round(len(times) * TIME_STEP_MS / 1000.0, 4),
        }

        # ── spectrogram — always computed fresh from audio ────────────────────
        # The base cache stores only f0 data, so spec is regenerated each time.
        # It's cheap (plain STFT) compared to the neural f0 extractor.
        try:
            n_fft = 1024   # 513 bins, ~15.6 Hz/bin @ 16 kHz

            D     = np.abs(librosa.stft(audio_pad, n_fft=n_fft,
                                        hop_length=WINDOW, center=True))
            D_db  = librosa.amplitude_to_db(D, ref=np.max)   # (n_bins, n_frames)

            # Trim columns to match f0 frame count
            D_db     = D_db[:, :len(freqs)]
            n_bins, n_frames = D_db.shape

            # Percentile-clip then normalise to [0, 1]
            lo_p = float(np.percentile(D_db, 5))
            hi_p = float(np.percentile(D_db, 99))
            norm = np.clip((D_db - lo_p) / (hi_p - lo_p + 1e-8),
                           0.0, 1.0).astype(np.float32)

            # Flip so row 0 = Nyquist (top = high freq = top of canvas)
            norm = norm[::-1, :].copy()

            # Apply colormap
            v          = norm.ravel()
            r_, g_, b_ = _plasma_colormap(v)
            rgb        = np.stack([r_, g_, b_], axis=-1).reshape(n_bins, n_frames, 3)

            png_bytes = _encode_rgb_png(rgb)
            spec_b64  = base64.b64encode(png_bytes).decode('ascii')

            result["spec_b64"]     = spec_b64
            result["spec_n_bins"]  = n_bins
            result["spec_nyquist"] = SAMPLE_RATE // 2   # 8000 Hz

        except Exception as spec_exc:
            print(f"[F0Editor] Spectrogram skipped: {spec_exc}")

        return json.dumps(result)

    except Exception as exc:
        print(f"[F0Editor] Error during f0 extraction: {exc}")
        print(traceback.format_exc())
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Save edited f0 back to a temp CSV the pipeline can consume
# ──────────────────────────────────────────────────────────────────────────────
def save_edited_f0_to_temp(json_string: str) -> F0TempFile | None:
    """
    Parse the JSON produced by the JS editor and write it as a CSV file that
    pipeline.py can read:  each line is  <time_seconds>,<freq_hz>

    Written to the session/ subdir which is purged on next startup.
    Returns an F0TempFile wrapper on success, None on failure.
    """
    if not json_string or json_string.strip() in ("", "null"):
        return None
    try:
        os.makedirs(F0_SESSION_DIR, exist_ok=True)
        data  = json.loads(json_string)
        times = data["times"]
        freqs = data["freqs"]

        fd, path = tempfile.mkstemp(suffix=".csv", dir=F0_SESSION_DIR)
        with os.fdopen(fd, "w") as f:
            for t, fr in zip(times, freqs):
                f.write(f"{t},{fr}\n")

        return F0TempFile(path)
    except Exception as exc:
        print(f"[F0Editor] Error saving edited f0: {exc}")
        print(traceback.format_exc())
        return None


# ──────────────────────────────────────────────────────────────────────────────
# House-keeping
# ──────────────────────────────────────────────────────────────────────────────
def cleanup_old_temp_f0(max_age_hours: int = 24) -> None:
    """
    Legacy name kept so inference.py import doesn't break.
    Prunes old files from base_f0/ cache only (session/ is wiped on startup).
    """
    if not os.path.isdir(F0_BASE_DIR):
        return
    cutoff = time.time() - max_age_hours * 3600
    for fname in os.listdir(F0_BASE_DIR):
        fpath = os.path.join(F0_BASE_DIR, fname)
        try:
            if os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Preset API  (called from inference.py event handlers)
# ──────────────────────────────────────────────────────────────────────────────
def _preset_path(name: str) -> str:
    """Sanitise preset name and return its full .csv path."""
    safe = "".join(c for c in name if c.isalnum() or c in " _-()").strip()
    if not safe:
        raise ValueError(f"Invalid preset name: {name!r}")
    return os.path.join(F0_PRESETS_DIR, safe + ".csv")


def list_f0_presets() -> list[str]:
    """Return sorted list of saved preset names (without extension)."""
    if not os.path.isdir(F0_PRESETS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(F0_PRESETS_DIR)
        if f.endswith(".csv")
    )


def save_f0_preset(name: str, freqs_json: str) -> str:
    """
    Save a freq array (JSON list of floats) as a named preset CSV.
    Each line is a single freq value in Hz (0 = unvoiced).
    Returns a status string.
    """
    try:
        freqs = json.loads(freqs_json)
        path  = _preset_path(name)
        os.makedirs(F0_PRESETS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for f in freqs:
                fh.write(f"{f}\n")
        print(f"[F0Editor] Preset saved: {os.path.basename(path)}")
        return json.dumps({"ok": True, "names": list_f0_presets()})
    except Exception as exc:
        print(f"[F0Editor] Preset save error: {exc}")
        return json.dumps({"ok": False, "error": str(exc)})


def load_f0_preset(name: str) -> str:
    """
    Load a named preset CSV and return its freq list as a JSON string.
    The JS side resamples the list to match the current song's frame count.
    Returns {"ok": True, "freqs": [...]} or {"ok": False, "error": "..."}.
    """
    try:
        path  = _preset_path(name)
        freqs = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    freqs.append(float(line))
        return json.dumps({"ok": True, "freqs": freqs})
    except Exception as exc:
        print(f"[F0Editor] Preset load error: {exc}")
        return json.dumps({"ok": False, "error": str(exc)})


# ──────────────────────────────────────────────────────────────────────────────
# HTML skeleton
# ──────────────────────────────────────────────────────────────────────────────
def build_f0_editor_html() -> str:
    """HTML + CSS skeleton only – no <script> tags."""
    return r"""
<style>
/* ── Container ────────────────────────────────────────────────────────────── */
#f0ed-root {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  background: #1a1a2e;
  border: 1px solid #2d2d4e;
  border-radius: 10px;
  overflow: hidden;
  color: #e0e0f0;
  user-select: none;
  position: relative;
}

/* ── Info modal ───────────────────────────────────────────────────────────── */
#f0ed-modal {
  display: none;
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(6,6,20,0.90);
  z-index: 9999;
  align-items: flex-start;
  justify-content: center;
  padding-top: 40px;
  box-sizing: border-box;
  overflow-y: auto;
}
#f0ed-modal.open { display: flex; }
#f0ed-modal-box {
  background: #151530;
  border: 1px solid #4a4a9e;
  border-radius: 8px;
  padding: 20px 24px;
  max-width: 640px;
  width: 92%;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  font-size: 12px;
  line-height: 1.7;
  color: #b8b8e0;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
#f0ed-modal-box h3 { color: #9999ff; margin: 0 0 12px; font-size: 14px; letter-spacing:.03em; }
#f0ed-modal-box h4 { color: #7777cc; margin: 14px 0 5px; font-size: 12px; text-transform:uppercase; letter-spacing:.06em; }
#f0ed-modal-box table { width: 100%; border-collapse: collapse; margin-bottom: 4px; }
#f0ed-modal-box td { padding: 3px 8px; vertical-align: top; border-bottom: 1px solid #22224a; }
#f0ed-modal-box td:first-child { color: #aaaaff; white-space: nowrap; font-family: monospace; min-width: 130px; }
#f0ed-modal-box .close-row { text-align: right; margin-top: 16px; }
#f0ed-modal-box .close-row button {
  background: #303060; border: 1px solid #6666cc; color: #aaaaff;
  padding: 5px 16px; border-radius: 5px; cursor: pointer; font-size: 12px;
}
#f0ed-modal-box .close-row button:hover { background: #4444aa; color: #fff; }
#f0ed-modal-box kbd {
  display: inline-block; background: #1e1e40; border: 1px solid #4a4a7e;
  border-radius: 3px; padding: 0 4px; font-family: monospace; font-size: 10px;
  color: #9999dd; line-height: 1.6;
}

/* ── Toolbar ──────────────────────────────────────────────────────────────── */
#f0ed-toolbar {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 10px;
  background: #12122a;
  border-bottom: 1px solid #2d2d4e;
  flex-wrap: wrap;
}
#f0ed-toolbar button {
  background: #252548;
  border: 1px solid #3a3a6e;
  color: #b0b0e0;
  padding: 4px 10px;
  border-radius: 5px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  white-space: nowrap;
}
#f0ed-toolbar button:hover  { background: #33336a; color: #fff; }
#f0ed-toolbar button.active { background: #5555cc; color: #fff; border-color: #7777ff; }
#f0ed-toolbar button.active-green { background: #226644; color: #aaffcc; border-color: #44cc88; }
#f0ed-toolbar .sep {
  width: 1px; height: 20px;
  background: #3a3a6e; margin: 0 4px; flex-shrink: 0;
}
#f0ed-toolbar label { font-size: 11px; color: #8888bb; white-space: nowrap; }
#f0ed-toolbar input[type=range] {
  width: 70px; accent-color: #7777ff; cursor: pointer;
}
#f0ed-toolbar select {
  background: #252548; border: 1px solid #3a3a6e; color: #b0b0e0;
  border-radius: 4px; font-size: 11px; padding: 2px 4px; cursor: pointer;
}

/* ── Persistent second toolbar row ───────────────────────────────────────── */
#f0ed-toolbar2 {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  background: #10102a;
  border-bottom: 1px solid #2d2d4e;
  flex-wrap: wrap;
}
#f0ed-toolbar2 button {
  background: #252548;
  border: 1px solid #3a3a6e;
  color: #b0b0e0;
  padding: 3px 9px;
  border-radius: 5px;
  font-size: 11px;
  cursor: pointer;
  white-space: nowrap;
}
#f0ed-toolbar2 button:hover  { background: #33336a; color: #fff; }
#f0ed-toolbar2 button.active-green { background: #226644; color: #aaffcc; border-color: #44cc88; }

/* ── Sub-toolbar (tool-specific settings) ────────────────────────────────── */
#f0ed-subtoolbar {
  display: none;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  background: #0e0e20;
  border-bottom: 1px solid #2d2d4e;
  font-size: 11px;
  color: #8888bb;
  flex-wrap: wrap;
}
#f0ed-subtoolbar.visible { display: flex; }
#f0ed-subtoolbar input[type=range] { width: 80px; accent-color: #7777ff; }
.f0ed-slider-wrap { display:inline-flex; align-items:center; gap:1px; vertical-align:middle; }
.f0ed-slider-wrap button {
  background:#1e1e40; border:1px solid #3a3a6e; color:#9090cc;
  width:15px; height:15px; border-radius:2px; font-size:11px; font-weight:bold;
  cursor:pointer; padding:0; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
}
.f0ed-slider-wrap button:hover { background:#33336a; color:#fff; }
/* Select tool buttons — smaller than toolbar, with group separators */
#f0ed-sub-select button {
  background: #252548; border: 1px solid #3a3a6e; color: #b0b0e0;
  padding: 3px 8px; border-radius: 4px; font-size: 11px;
  cursor: pointer; white-space: nowrap;
}
#f0ed-sub-select button:hover { background: #33336a; color: #fff; }
#f0ed-sub-select .sel-sep { width:1px; height:14px; background:#3a3a6e; margin:0 3px; display:inline-block; vertical-align:middle; }

#f0ed-subtoolbar input[type=number] {
  width: 52px; background: #252548; border: 1px solid #3a3a6e;
  color: #d0d0f0; border-radius: 4px; padding: 2px 4px; font-size: 11px;
}

/* ── Canvas wrapper ───────────────────────────────────────────────────────── */
#f0ed-canvas-wrap {
  position: relative;
  width: 100%;
  height: 580px;
  overflow: hidden;
  cursor: crosshair;
  background: #0e0e1e;
}
#f0ed-canvas-bg, #f0ed-canvas-main, #f0ed-canvas-overlay {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
}
#f0ed-canvas-bg      { z-index: 1; }
#f0ed-canvas-main    { z-index: 2; }
#f0ed-canvas-overlay { z-index: 3; }

/* ── Status bar ───────────────────────────────────────────────────────────── */
#f0ed-statusbar {
  display: flex;
  justify-content: space-between;
  padding: 4px 10px;
  background: #12122a;
  border-top: 1px solid #2d2d4e;
  font-size: 11px;
  color: #6666aa;
}
#f0ed-statusbar span { color: #9999cc; }

/* ── Preset bar ───────────────────────────────────────────────────────────── */
#f0ed-presetbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: #0e0e20;
  border-top: 1px solid #2a2a4a;
  font-size: 11px;
  color: #7777aa;
  flex-wrap: wrap;
}
#f0ed-presetbar label { color: #8888bb; white-space: nowrap; }
#f0ed-preset-select {
  background: #1a1a35; color: #c0c0e8;
  border: 1px solid #3a3a6e; border-radius: 4px;
  padding: 3px 6px; font-size: 11px;
  min-width: 140px; max-width: 220px;
  cursor: pointer;
}
#f0ed-preset-select:focus { outline: 1px solid #6666cc; }
#f0ed-preset-name {
  background: #1a1a35; color: #c0c0e8;
  border: 1px solid #3a3a6e; border-radius: 4px;
  padding: 3px 7px; font-size: 11px;
  width: 140px;
}
#f0ed-preset-name:focus { outline: 1px solid #6666cc; }
#f0ed-presetbar button {
  background: #22224a; color: #aaaadd;
  border: 1px solid #4a4a8a; border-radius: 4px;
  padding: 3px 9px; font-size: 11px;
  cursor: pointer; white-space: nowrap;
}
#f0ed-presetbar button:hover { background: #33336a; color: #fff; }
#f0ed-preset-sep { width: 1px; height: 16px; background: #2d2d4e; margin: 0 2px; }
#f0ed-preset-sep2 { width: 1px; height: 16px; background: #2d2d4e; margin: 0 2px; }
#f0ed-preset-status { color: #666699; font-size: 10px; margin-left: 4px; }
</style>

<div id="f0ed-root">

  <!-- ── Main toolbar ──────────────────────────────────────────────────── -->
  <div id="f0ed-toolbar">
    <!-- Drawing tools -->
    <button id="f0ed-tool-pen"      class="active" title="Freehand draw">✏️ Pen</button>
    <button id="f0ed-tool-smoothpen"               title="Smooth freehand draw">〰️ Smooth Pen</button>
    <button id="f0ed-tool-line"                    title="Drag to draw a straight line between two points">⟋ Line</button>
    <button id="f0ed-tool-envelope"                title="Double-click the pitch curve to place square handles; drag to reshape in real-time">◇ Envelope</button>
    <button id="f0ed-tool-select"                  title="Select a region to transpose">⬚ Select</button>
    <button id="f0ed-tool-smooth"                  title="Smooth-brush – paint to smoothen">≈ Smooth Brush</button>
    <button id="f0ed-tool-vibrato"                 title="Add vibrato to a region">〜 Vibrato</button>
    <button id="f0ed-tool-eraser"                  title="Erase (set to unvoiced / 0 Hz)">⬜ Eraser</button>
    <div class="sep"></div>
    <!-- History -->
    <button id="f0ed-undo"        title="Undo (Ctrl+Z)">↩ Undo</button>
    <button id="f0ed-redo"        title="Redo (Ctrl+Y)">↪ Redo</button>
    <button id="f0ed-reset"       title="Reset to original extracted curve">🔄 Reset Curve</button>
    <button id="f0ed-reset-view"  title="Reset zoom and pan to fit the full clip">↩ Reset View</button>
    <div class="sep"></div>
    <!-- Spectrogram overlay -->
    <button id="f0ed-spec-toggle" title="Toggle spectrogram overlay (shows harmonics)">🎨 Spec</button>
    <label title="Spectrogram opacity">
      Opacity: <input type="range" id="f0ed-spec-opacity" min="5" max="100" step="5" value="55" style="width:60px">
    </label>
    <button id="f0ed-ghost-toggle" class="active-green" title="Toggle original F0 ghost curve">👻 F0 Orig</button>
    <label title="Original F0 ghost opacity">
      Opacity: <input type="range" id="f0ed-ghost-opacity" min="5" max="100" step="5" value="100" style="width:60px">
    </label>
    <button id="f0ed-info" title="Show controls reference">ℹ Info</button>
  </div>

  <!-- ── Persistent second row ─────────────────────────────────────────── -->
  <div id="f0ed-toolbar2">
    <button id="f0ed-enforce" title="When active: Line and Envelope will only write onto already-voiced (non-silent) frames — drawing over silent gaps is blocked">⊢ Enforce Pitch Boundaries</button>
  </div>

  <!-- ── Tool-specific sub-toolbar ─────────────────────────────────────── -->
  <div id="f0ed-subtoolbar">
    <span id="f0ed-sub-smoothpen" style="display:none">
      Smoothing: <span class="f0ed-slider-wrap"><button class="f0ed-sb" data-t="f0ed-smooth-amount" data-d="-1">−</button><input type="range" id="f0ed-smooth-amount" min="0" max="0.95" step="0.05" value="0.5"><button class="f0ed-sb" data-t="f0ed-smooth-amount" data-d="1">+</button></span>
      <span id="f0ed-smooth-val">0.5</span> &nbsp;<small style="color:#666">(higher = smoother / more lag)</small>
    </span>
    <span id="f0ed-sub-vibrato" style="display:none">
      Rate (Hz): <span class="f0ed-slider-wrap"><button class="f0ed-sb" data-t="f0ed-vibrato-rate" data-d="-1">−</button><input type="range" id="f0ed-vibrato-rate" min="1" max="12" step="0.5" value="5"><button class="f0ed-sb" data-t="f0ed-vibrato-rate" data-d="1">+</button></span>
      <span id="f0ed-vrate-val">5</span>
      &nbsp; Depth (cents): <span class="f0ed-slider-wrap"><button class="f0ed-sb" data-t="f0ed-vibrato-depth" data-d="-1">−</button><input type="range" id="f0ed-vibrato-depth" min="5" max="100" step="5" value="25"><button class="f0ed-sb" data-t="f0ed-vibrato-depth" data-d="1">+</button></span>
      <span id="f0ed-vdepth-val">25</span>
    </span>
    <span id="f0ed-sub-select" style="display:none; align-items:center; gap:2px;">
      <button id="f0ed-sel-up-semi">+1 semi</button>
      <button id="f0ed-sel-down-semi">−1 semi</button>
      <span class="sel-sep"></span>
      <button id="f0ed-sel-up-oct">+1 oct</button>
      <button id="f0ed-sel-down-oct">−1 oct</button>
      <span class="sel-sep"></span>
      <button id="f0ed-sel-clear">Clear</button>
      <span class="sel-sep"></span>
      <button id="f0ed-sel-preview" title="Preview selected region as sine wave following the pitch curve">▶ Preview</button>
      <span id="f0ed-sel-preview-status" style="color:#888;font-size:10px;margin-left:2px;"></span>
    </span>
    <span id="f0ed-sub-envelope" style="display:none; align-items:center; gap:8px;">
      Smooth: <span class="f0ed-slider-wrap"><button class="f0ed-sb" data-t="f0ed-env-smooth" data-d="-1">−</button><input type="range" id="f0ed-env-smooth" min="0" max="100" step="1" value="0" style="width:80px"><button class="f0ed-sb" data-t="f0ed-env-smooth" data-d="1">+</button></span>
      <span id="f0ed-env-smooth-val">0</span>
      <small style="color:#555">(0 = sharp corners, 100 = fully curved spline through nodes)</small>
      &nbsp;
      <button id="f0ed-env-clear" title="Remove all envelope nodes and restore curve">✕ Clear Envelopes</button>
    </span>
    <span id="f0ed-sub-smooth" style="display:none">
      Brush radius: <span class="f0ed-slider-wrap"><button class="f0ed-sb" data-t="f0ed-brushradius" data-d="-1">−</button><input type="range" id="f0ed-brushradius" min="1" max="30" step="1" value="6"><button class="f0ed-sb" data-t="f0ed-brushradius" data-d="1">+</button></span>
      <span id="f0ed-brush-val">6</span>
    </span>
  </div>

  <!-- ── Canvas ─────────────────────────────────────────────────────────── -->
  <div id="f0ed-canvas-wrap">
    <canvas id="f0ed-canvas-bg"></canvas>
    <canvas id="f0ed-canvas-main"></canvas>
    <canvas id="f0ed-canvas-overlay"></canvas>
  </div>

  <!-- ── Status bar ─────────────────────────────────────────────────────── -->
  <div id="f0ed-statusbar">
    <div id="f0ed-status-left">Waiting for F0 data…</div>
    <div id="f0ed-status-right">Time: <span id="f0ed-cur-time">–</span> s &nbsp;|&nbsp; Freq: <span id="f0ed-cur-freq">–</span> Hz &nbsp;|&nbsp; Note: <span id="f0ed-cur-note">–</span></div>
  </div>

  <!-- ── Preset bar ─────────────────────────────────────────────────────── -->
  <div id="f0ed-presetbar">
    <label>Saved:</label>
    <select id="f0ed-preset-select" title="Select a saved preset">
      <option value="">— no presets saved —</option>
    </select>
    <button id="f0ed-preset-load" title="Apply selected preset to the current curve (resampled to song length)">▶ Load</button>
    <div id="f0ed-preset-sep"></div>
    <button id="f0ed-preset-import" title="Import a .csv file from disk into the editor (also saves it to presets)">📂 Import File</button>
    <input type="file" id="f0ed-preset-file-input" accept=".csv" style="display:none" />
    <div id="f0ed-preset-sep2"></div>
    <input type="text" id="f0ed-preset-name" placeholder="Preset name…" title="Name for the new preset" maxlength="64" />
    <button id="f0ed-preset-save" title="Save current curve as a named preset">💾 Save Preset</button>
    <span id="f0ed-preset-status"></span>
  </div>

  <!-- ── Info modal ─────────────────────────────────────────────────────── -->
  <div id="f0ed-modal">
    <div id="f0ed-modal-box">
      <h3>F0 Curve Editor — Controls Reference</h3>

      <h4>Drawing Tools  <small style="color:#666">(or press 1–8)</small></h4>
      <table>
        <tr><td>✏️ Pen &nbsp;<kbd>1</kbd></td><td>Freehand draw pitch directly onto the canvas. Interpolates between samples so fast strokes leave no gaps.</td></tr>
        <tr><td>〰️ Smooth Pen &nbsp;<kbd>2</kbd></td><td>Freehand draw with exponential smoothing — adjust the lag with the Smoothing slider in the sub-toolbar.</td></tr>
        <tr><td>⟋ Line &nbsp;<kbd>3</kbd></td><td>Drag to draw a straight interpolated line between two points. Respects <em>Enforce Pitch Boundaries</em>.</td></tr>
        <tr><td>◇ Envelope &nbsp;<kbd>4</kbd></td><td>Double-click on the pitch curve to place square control nodes — the curve immediately snaps to the interpolated shape. Drag nodes to reshape in real-time. Nodes persist when you switch tools. Use Ctrl+Z to undo the last node; ✕ Clear Envelopes removes all nodes and restores the curve. The Smooth slider morphs the interpolation from sharp linear corners (0) to a fully curved Catmull-Rom spline (100).</td></tr>
        <tr><td>⬚ Select &nbsp;<kbd>5</kbd></td><td>Drag to draw a rectangular selection (time × frequency). Only frames whose pitch falls inside the box are transposed or previewed. Use the sub-toolbar buttons to shift by semitones / octaves.</td></tr>
        <tr><td>≈ Smooth Brush &nbsp;<kbd>6</kbd></td><td>Paint over any region to smooth it locally with a Gaussian kernel. Adjust the brush radius in the sub-toolbar.</td></tr>
        <tr><td>〜 Vibrato &nbsp;<kbd>7</kbd></td><td>Paint rightward over a region to add vibrato. Adjust rate (Hz) and depth (cents) in the sub-toolbar. Each stroke starts at phase 0 so separate strokes never interfere.</td></tr>
        <tr><td>⬜ Eraser &nbsp;<kbd>8</kbd></td><td>Set frames to unvoiced (0 Hz / silent). Always works regardless of Enforce Pitch Boundaries.</td></tr>
      </table>

      <h4>Toolbar Extras</h4>
      <table>
        <tr><td>⊢ Enforce Pitch Boundaries</td><td>When green: the Line and Envelope tools skip over silent/unvoiced frames and never extend outside originally-voiced regions.</td></tr>
        <tr><td>↩ Undo / ↪ Redo</td><td>Step through the full edit history. Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z).</td></tr>
        <tr><td>🔄 Reset Curve</td><td>Restore the curve to the originally extracted F0, wipe envelope nodes, and clear the selection.</td></tr>
        <tr><td>🎨 Spec + Opacity</td><td>Toggle the STFT spectrogram background overlay and adjust its transparency. Columns align 1-to-1 with F0 frames.</td></tr>
        <tr><td>👻 F0 Orig + Opacity</td><td>Toggle the ghost overlay showing the original unedited F0 curve in green, and adjust its transparency.</td></tr>
      </table>

      <h4>Mouse Navigation</h4>
      <table>
        <tr><td>Scroll wheel</td><td>Zoom time axis around the cursor position.</td></tr>
        <tr><td>Alt + Scroll</td><td>Zoom frequency (Y) axis around the cursor frequency.</td></tr>
        <tr><td>Shift + Scroll</td><td>Pan left / right along the time axis.</td></tr>
        <tr><td>Middle-drag</td><td>Pan both time and frequency axes simultaneously.</td></tr>
        <tr><td>Right-drag</td><td>Pan the frequency (Y) axis only.</td></tr>
        <tr><td>Double-click</td><td>Place an envelope node (Envelope tool only).</td></tr>
      </table>

      <h4>Keyboard Shortcuts</h4>
      <table>
        <tr><td>1 – 8</td><td>Switch tool (only fires when cursor is over the editor): 1 Pen · 2 Smooth Pen · 3 Line · 4 Envelope · 5 Select · 6 Smooth Brush · 7 Vibrato · 8 Eraser</td></tr>
        <tr><td>Ctrl+Z</td><td>Undo (also removes the last placed envelope node one at a time)</td></tr>
        <tr><td>Ctrl+Y / Ctrl+Shift+Z</td><td>Redo</td></tr>
      </table>

      <div class="close-row"><button id="f0ed-modal-close">Close</button></div>
    </div>
  </div>

</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# JavaScript
# ──────────────────────────────────────────────────────────────────────────────
def build_f0_editor_js() -> str:
    """JS function body for gr.HTML(js_on_load=...).
    Gradio 6 executes this via:
      new Function("element","trigger","props","server","upload", body)
    js_on_load fires once when the panel mounts (i.e. after the data
    is already written into f0_data_for_editor).
    """
    return r""""use strict";

// ════════════════════════════════════════════════════════════════════════════
//  Constants
// ════════════════════════════════════════════════════════════════════════════
const F0_MAX    = 1200;   // Hz ceiling used throughout the editor
const TOOL_KEYS = ['pen','smoothpen','line','envelope','select','smooth','vibrato','eraser'];

// ════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════
const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

function hzToMidi(hz) {
  if (hz <= 0) return null;
  return 69 + 12 * Math.log2(hz / 440);
}
function midiToHz(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }
function hzToNoteName(hz) {
  if (hz <= 0) return '–';
  const m   = Math.round(hzToMidi(hz));
  const oct = Math.floor(m / 12) - 1;
  return NOTE_NAMES[((m % 12) + 12) % 12] + oct;
}
function hzToMel(hz) { return 2595 * Math.log10(1 + Math.max(hz, 1) / 700); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Gaussian blur 1-D
function gaussSmooth(arr, sigma) {
  const r   = Math.ceil(sigma * 2);
  const len = arr.length;
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    let sum = 0, wsum = 0;
    for (let d = -r; d <= r; d++) {
      const j = i + d;
      if (j < 0 || j >= len) continue;
      if (arr[j] === 0) continue;
      const w = Math.exp(-(d*d)/(2*sigma*sigma));
      sum  += arr[j] * w;
      wsum += w;
    }
    out[i] = wsum > 0 ? sum / wsum : arr[i];
  }
  return out;
}

// Chaikin smoothing pass (for smooth-pen)
function chaikinPass(pts) {
  if (pts.length < 3) return pts;
  const out = [pts[0]];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i], p1 = pts[i + 1];
    out.push({ x: 0.75*p0.x + 0.25*p1.x, y: 0.75*p0.y + 0.25*p1.y });
    out.push({ x: 0.25*p0.x + 0.75*p1.x, y: 0.25*p0.y + 0.75*p1.y });
  }
  out.push(pts[pts.length - 1]);
  return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  Editor class
// ════════════════════════════════════════════════════════════════════════════
class F0Editor {
  constructor() {
    this.bgCanvas   = document.getElementById('f0ed-canvas-bg');
    this.mainCanvas = document.getElementById('f0ed-canvas-main');
    this.ovCanvas   = document.getElementById('f0ed-canvas-overlay');
    this.bgCtx      = this.bgCanvas.getContext('2d');
    this.mainCtx    = this.mainCanvas.getContext('2d');
    this.ovCtx      = this.ovCanvas.getContext('2d');
    this.wrap       = document.getElementById('f0ed-canvas-wrap');

    // F0 data
    this.originalF0  = [];
    this.currentF0   = [];
    this.timeStepMs  = 10;

    // History
    this.history  = [];
    this.histIdx  = -1;
    this.MAX_HIST = 60;

    // X view (sample indices)
    this.viewStart = 0;
    this.viewEnd   = 0;

    // Y view (Hz)
    this.yMin        = 60;
    this.yMax        = 600;

    // Spectrogram
    this.specImage      = null;    // HTMLImageElement
    this.specNBins      = 0;
    this.specNyquist    = 8000;    // Hz at row 0 of image
    this.specVisible    = false;
    this.specOpacity    = 0.55;

    // Ghost (original) F0 overlay
    this.ghostVisible   = true;
    this.ghostOpacity   = 1.0;

    // Tool state
    this.activeTool  = 'pen';
    this.isDrawing   = false;
    this.lastSample  = null;
    this.lastFreq    = null;
    this.smoothPts   = [];
    this.selStart    = null;
    this.selEnd      = null;
    this.selFreqLo   = null;   // Hz — bottom of rectangular selection
    this.selFreqHi   = null;   // Hz — top of rectangular selection
    this.lineStart      = null;
    this.smoothHead     = null;

    // Envelope nodes — persistent across tool switches
    // Each node: { s, f, segStart, segEnd }
    this.envelopeNodes  = [];
    this.envDragging    = null;   // index of node currently being dragged
    this.envDragBase    = null;   // kept for legacy; replaced by preEnvelopeF0
    // Snapshot of currentF0 taken when the FIRST envelope node is placed.
    // All envelope re-applications start from this so they are always idempotent
    // and never compound on top of each other.
    this.preEnvelopeF0  = null;

    // Global draw option
    this.enforceBoundaries = false;

    // X panning (middle-button)
    this.isPanning    = false;
    this.panStartX    = 0;
    this.panStartY    = 0;
    this.panStartVS   = 0;
    this.panStartVE   = 0;

    // Y panning (right-button drag)
    this.isYPanning   = false;
    this.yPanStartY   = 0;
    this.yPanStartMin = 0;
    this.yPanStartMax = 0;

    // Gradio sync debounce timer
    this._syncTimer  = null;
    // Track whether the cursor is over the editor — used to scope 1-8 shortcuts
    this.isHovered   = false;

    this._bindUI();
    this._bindCanvas();
    this._resizeObserver();
  }

  // ── Data ────────────────────────────────────────────────────────────────
  loadData(jsonStr) {
    try {
      const d = JSON.parse(jsonStr);
      this.timeStepMs = d.time_step_ms || 10;
      this.originalF0 = Float32Array.from(d.freqs);
      this.currentF0  = Float32Array.from(d.freqs);
      this.viewStart  = 0;
      this.viewEnd    = this.currentF0.length;
      this.history    = [];
      this.histIdx    = -1;
      this._pushHistory();
      this._autoFitY();

      // Load spectrogram if present
      this.specImage = null;
      if (d.spec_b64) {
        this.specNBins    = d.spec_n_bins  || 513;
        this.specNyquist  = d.spec_nyquist || 8000;
        const img = new Image();
        img.onload = () => { this.specImage = img; this._renderAll(); };
        img.src = 'data:image/png;base64,' + d.spec_b64;
        if (!this.specVisible) {
          this.specVisible = true;
          const btn = document.getElementById('f0ed-spec-toggle');
          if (btn) btn.classList.add('active-green');
        }
      }

      this._renderAll();
      document.getElementById('f0ed-status-left').textContent =
        `Loaded ${d.freqs.length} frames (${(d.duration||0).toFixed(2)} s) · ${this.timeStepMs} ms hop` +
        (d.spec_b64 ? ' · spec ✓' : ' · no spec');
    } catch(e) {
      console.error('[F0Editor] loadData error:', e);
    }
  }

  toJSON() {
    const n  = this.currentF0.length;
    const ts = this.timeStepMs / 1000;
    const times = [], freqs = [];
    for (let i = 0; i < n; i++) {
      times.push(parseFloat((i * ts).toFixed(6)));
      freqs.push(this.currentF0[i]);
    }
    return JSON.stringify({ times, freqs, time_step_ms: this.timeStepMs, duration: n * ts });
  }

  // ── History ──────────────────────────────────────────────────────────────
  // Each entry: { f0, nodes, preEnv } — full snapshot of all mutable state
  _pushHistory() {
    this.history = this.history.slice(0, this.histIdx + 1);
    this.history.push({
      f0:     Float32Array.from(this.currentF0),
      nodes:  this.envelopeNodes.map(n => ({ ...n })),
      preEnv: this.preEnvelopeF0 ? Float32Array.from(this.preEnvelopeF0) : null,
    });
    if (this.history.length > this.MAX_HIST) this.history.shift();
    this.histIdx = this.history.length - 1;
    this._syncToGradio();
  }
  _restoreHistory(h) {
    this.currentF0     = Float32Array.from(h.f0);
    this.envelopeNodes = h.nodes.map(n => ({ ...n }));
    this.preEnvelopeF0 = h.preEnv ? Float32Array.from(h.preEnv) : null;
  }
  undo() {
    if (this.histIdx <= 0) return;
    this.histIdx--;
    this._restoreHistory(this.history[this.histIdx]);
    this._renderAll(); this._syncToGradio();
  }
  redo() {
    if (this.histIdx >= this.history.length - 1) return;
    this.histIdx++;
    this._restoreHistory(this.history[this.histIdx]);
    this._renderAll(); this._syncToGradio();
  }
  reset() {
    this.currentF0     = Float32Array.from(this.originalF0);
    this.selStart      = this.selEnd = null;
    this.selFreqLo     = this.selFreqHi = null;
    // Wipe envelope state so the reset is truly clean
    this.envelopeNodes = [];
    this.envDragging   = null;
    this.envDragBase   = null;
    this.preEnvelopeF0 = null;
    this._pushHistory();
    this._renderAll();
  }

  resetView() {
    this.viewStart = 0;
    this.viewEnd   = this.currentF0.length;
    this._autoFitY();
    this._renderAll();
  }

  // ── Coord helpers ────────────────────────────────────────────────────────
  _sampleToX(s) {
    const w = this.mainCanvas.width;
    return (s - this.viewStart) / (this.viewEnd - this.viewStart) * w;
  }
  _xToSample(x) {
    const w = this.mainCanvas.width;
    const s = this.viewStart + (x / w) * (this.viewEnd - this.viewStart);
    return Math.round(clamp(s, 0, this.currentF0.length - 1));
  }
  _freqToY(hz) {
    const h = this.mainCanvas.height;
    return h - (hz - this.yMin) / (this.yMax - this.yMin) * h;
  }
  _yToFreq(y) {
    const h = this.mainCanvas.height;
    return clamp(this.yMin + (1 - y / h) * (this.yMax - this.yMin), 0, F0_MAX);
  }
  _autoFitY() {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < this.currentF0.length; i++) {
      const f = this.currentF0[i];
      if (f <= 0) continue;
      if (f < lo) lo = f;
      if (f > hi) hi = f;
    }
    if (lo === Infinity) return;
    const pad = Math.max((hi - lo) * 0.25, 50);
    this.yMin = Math.max(20,   lo - pad);
    this.yMax = Math.min(1200, hi + pad);
  }

  // ── Y-axis helpers ───────────────────────────────────────────────────────
  _shiftY(deltaHz) {
    const range = this.yMax - this.yMin;
    this.yMin = Math.max(20,   this.yMin + deltaHz);
    this.yMax = this.yMin + range;
    if (this.yMax > 8000) { this.yMax = 8000; this.yMin = Math.max(20, this.yMax - range); }
    this._renderAll();
  }
  _zoomY(factor, pivotHz) {
    const range = this.yMax - this.yMin;
    if (pivotHz === undefined) pivotHz = (this.yMin + this.yMax) / 2;
    const newRange = clamp(range * factor, 30, 7980);
    const ratio = (pivotHz - this.yMin) / range;
    this.yMin = Math.max(20,   pivotHz - ratio * newRange);
    this.yMax = Math.min(8000, this.yMin + newRange);
    this._renderAll();
  }

  // ── Rendering ────────────────────────────────────────────────────────────
  _resizeCanvas() {
    const rect = this.wrap.getBoundingClientRect();
    const W = Math.floor(rect.width)  || 800;
    const H = Math.floor(rect.height) || 320;
    [this.bgCanvas, this.mainCanvas, this.ovCanvas].forEach(c => {
      if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    });
  }

  _renderAll() {
    this._resizeCanvas();
    this._drawGrid();
    this._drawCurve();
    this._drawOverlay();
  }

  // ── Spectrogram ──────────────────────────────────────────────────────────
  _drawSpec() {
    if (!this.specImage || !this.specVisible || !this.specNBins) return;
    const ctx   = this.bgCtx;
    const W     = this.bgCanvas.width, H = this.bgCanvas.height;
    const nBins = this.specNBins;
    const nyq   = this.specNyquist;

    // ── X: exact floats, one image column per f0 frame ───────────────────
    const srcX = this.viewStart;
    const srcW = this.viewEnd - this.viewStart;

    // ── Y: linear Hz → image row ─────────────────────────────────────────
    // Row 0 = nyquist (high), row nBins-1 = 0 Hz (low)
    const hzToRow = hz => (1 - hz / nyq) * (nBins - 1);
    const srcY    = clamp(hzToRow(this.yMax),             0, nBins - 1);
    const srcYb   = clamp(hzToRow(Math.max(this.yMin, 0)), 0, nBins - 1);
    const srcH    = Math.max(0.5, srcYb - srcY);

    ctx.save();
    ctx.globalAlpha              = this.specOpacity;
    ctx.globalCompositeOperation = 'screen';
    ctx.imageSmoothingEnabled    = true;
    ctx.imageSmoothingQuality    = 'high';
    ctx.drawImage(this.specImage, srcX, srcY, srcW, srcH, 0, 0, W, H);
    ctx.restore();
  }

  _drawGrid() {
    const ctx = this.bgCtx;
    const W   = this.bgCanvas.width, H = this.bgCanvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0b0b1a';
    ctx.fillRect(0, 0, W, H);

    // Draw spectrogram first (underneath grid lines)
    this._drawSpec();

    const NOTE_COLORS = { C: '#3a3a6e', 'F#': '#2a2a50' };
    const drawHLine = (hz, label, color) => {
      const y = this._freqToY(hz);
      if (y < 0 || y > H) return;
      ctx.strokeStyle = color || '#2a2a4e';
      ctx.lineWidth   = 1;
      ctx.setLineDash([4, 6]);
      ctx.beginPath(); ctx.moveTo(48, y); ctx.lineTo(W, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#5555aa';
      ctx.font      = '10px monospace';
      ctx.fillText(label, 2, y + 3);
    };

    const range = this.yMax - this.yMin;
    const step  = range < 100 ? 10 : range < 300 ? 25 : range < 600 ? 50 : 100;
    for (let f = Math.ceil(this.yMin/step)*step; f <= this.yMax; f += step) {
      drawHLine(f, f.toFixed(0) + ' Hz', '#252548');
    }

    // Vertical time lines
    if (this.currentF0.length > 0) {
      const sampPerSec = 1000 / this.timeStepMs;
      const secStep    = Math.max(1, Math.round((this.viewEnd - this.viewStart) / sampPerSec / 10));
      const startSec   = Math.ceil(this.viewStart / sampPerSec / secStep) * secStep;
      ctx.strokeStyle  = '#1c1c38';
      ctx.setLineDash([2, 6]);
      ctx.lineWidth    = 1;
      for (let s = startSec; s * sampPerSec < this.viewEnd; s += secStep) {
        const x = this._sampleToX(s * sampPerSec);
        if (x < 48 || x > W) continue;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - 14); ctx.stroke();
        ctx.fillStyle = '#4444aa';
        ctx.font      = '10px monospace';
        ctx.fillText(s + 's', x + 2, H - 4);
      }
      ctx.setLineDash([]);
    }
  }

  _drawCurve() {
    const ctx = this.mainCtx;
    const W   = this.mainCanvas.width, H = this.mainCanvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!this.currentF0.length) return;

    const f0 = this.currentF0;
    const vS = this.viewStart, vE = this.viewEnd;

    // Unvoiced dashed baseline
    ctx.strokeStyle = '#3a3a5e';
    ctx.lineWidth   = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    for (let i = vS; i < vE; i++) {
      if (f0[i] !== 0) continue;
      const x = this._sampleToX(i);
      if (i === 0 || f0[i-1] !== 0) ctx.moveTo(x, H * 0.93);
      else                           ctx.lineTo(x, H * 0.93);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Ghost original F0 curve (drawn first, underneath the edited curve)
    if (this.ghostVisible && this.originalF0 && this.originalF0.length) {
      const orig = this.originalF0;
      ctx.save();
      ctx.globalAlpha = this.ghostOpacity;
      ctx.strokeStyle = '#33cc66';
      ctx.lineWidth   = 1.5;
      ctx.shadowColor = '#00ff88';
      ctx.shadowBlur  = 4;
      ctx.setLineDash([]);
      ctx.beginPath();
      let ghostSeg = false;
      for (let i = vS; i < vE; i++) {
        if (orig[i] <= 0) { ghostSeg = false; continue; }
        const x = this._sampleToX(i);
        const y = this._freqToY(orig[i]);
        if (!ghostSeg) { ctx.moveTo(x, y); ghostSeg = true; }
        else             ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Voiced curve
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0,   '#aa88ff');
    grad.addColorStop(0.5, '#7766ee');
    grad.addColorStop(1,   '#4444bb');
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = '#6644cc';
    ctx.shadowBlur  = 6;
    ctx.beginPath();
    let inSeg = false;
    for (let i = vS; i < vE; i++) {
      if (f0[i] <= 0) { inSeg = false; continue; }
      const x = this._sampleToX(i);
      const y = this._freqToY(f0[i]);
      if (!inSeg) { ctx.moveTo(x, y); inSeg = true; }
      else          ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Voiced/unvoiced boundary dots
    ctx.fillStyle = '#cc99ff';
    for (let i = vS + 1; i < vE - 1; i++) {
      const wasV = f0[i-1] > 0, isV = f0[i] > 0;
      if (wasV !== isV) {
        const si = isV ? i : i - 1;
        if (f0[si] > 0) {
          const x = this._sampleToX(si), y = this._freqToY(f0[si]);
          ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  }

  _drawOverlay() {
    const ctx = this.ovCtx;
    const W   = this.ovCanvas.width, H = this.ovCanvas.height;
    ctx.clearRect(0, 0, W, H);

    // Selection highlight — rectangular, respects both time and freq bounds
    if (this.activeTool === 'select' && this.selStart !== null && this.selEnd !== null
        && this.selFreqLo !== null && this.selFreqHi !== null) {
      const x1 = this._sampleToX(Math.min(this.selStart, this.selEnd));
      const x2 = this._sampleToX(Math.max(this.selStart, this.selEnd));
      const y1 = this._freqToY(this.selFreqHi);   // high freq = low Y
      const y2 = this._freqToY(this.selFreqLo);   // low  freq = high Y
      ctx.fillStyle   = 'rgba(120,120,255,0.18)';
      ctx.strokeStyle = 'rgba(180,180,255,0.6)';
      ctx.lineWidth   = 1;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    // Line-tool drag preview
    if (this.activeTool === 'line' && this.isDrawing && this.lineStart) {
      const x = this._sampleToX(this.lineStart.s);
      const y = this._freqToY(this.lineStart.f);
      ctx.strokeStyle = 'rgba(255,220,100,0.7)';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(this._lastMouseX || x, this._lastMouseY || y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Envelope nodes — always drawn regardless of active tool
    this._drawEnvelopeNodes(ctx, W, H);
  }

  // ── Envelope node rendering ───────────────────────────────────────────────
  _drawEnvelopeNodes(ctx, W, H) {
    const SZ = 11; // square side in px

    // Group nodes by segment, draw interpolated curve overlay per group
    const groups = this._envGroupsBySegment();
    for (const nodes of groups) {
      if (nodes.length < 2) continue;
      const sorted = [...nodes].sort((a, b) => a.s - b.s);
      const lo = sorted[0].s, hi = sorted[sorted.length - 1].s;
      const sliderV  = parseFloat((document.getElementById('f0ed-env-smooth') || {}).value) || 0;
      const tension  = (sliderV / 10) * 0.5;
      ctx.strokeStyle = 'rgba(255,210,60,0.78)';
      ctx.lineWidth   = 2;
      ctx.shadowColor = 'rgba(255,160,0,0.40)';
      ctx.shadowBlur  = 4;
      ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      for (let i = lo; i <= hi; i++) {
        if (this.currentF0[i] <= 0) { started = false; continue; }
        const f  = this._nodesInterp(sorted, i, tension);
        const px = this._sampleToX(i), py = this._freqToY(f);
        if (!started) { ctx.moveTo(px, py); started = true; }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Draw every node as a white square
    for (let i = 0; i < this.envelopeNodes.length; i++) {
      const n   = this.envelopeNodes[i];
      const px  = this._sampleToX(n.s);
      const py  = this._freqToY(n.f);
      if (px < -SZ || px > W + SZ) continue;
      const hot = this.envDragging === i;
      ctx.fillStyle   = hot ? '#ffee44' : '#ffffff';
      ctx.strokeStyle = hot ? '#ff9900' : 'rgba(80,80,200,0.9)';
      ctx.lineWidth   = hot ? 2 : 1.5;
      ctx.fillRect(px - SZ / 2, py - SZ / 2, SZ, SZ);
      ctx.strokeRect(px - SZ / 2, py - SZ / 2, SZ, SZ);
    }

    // Status hint when Envelope tool is active
    if (this.activeTool === 'envelope') {
      ctx.fillStyle = 'rgba(200,200,255,0.55)';
      ctx.font      = '11px monospace';
      ctx.fillText(
        this.envelopeNodes.length === 0
          ? 'Dbl-click on the pitch curve to place envelope nodes'
          : 'Dbl-click: add node  |  Drag: move (stays on same voiced region)  |  Ctrl+Z: undo last node',
        8, H - 6
      );
    }
  }

  // ── Envelope helpers ──────────────────────────────────────────────────────

  // Find voiced segment containing sample s; returns {lo, hi} or null
  _getSegmentBounds(s) {
    const f0 = this.currentF0;
    if (s < 0 || s >= f0.length || f0[s] <= 0) return null;
    let lo = s, hi = s;
    while (lo > 0 && f0[lo - 1] > 0) lo--;
    while (hi < f0.length - 1 && f0[hi + 1] > 0) hi++;
    return { lo, hi };
  }

  // Group envelopeNodes by which voiced segment they belong to
  // Returns array of arrays (one per distinct segment)
  _envGroupsBySegment() {
    const map = {};
    for (const n of this.envelopeNodes) {
      const key = `${n.segStart}:${n.segEnd}`;
      if (!map[key]) map[key] = [];
      map[key].push(n);
    }
    return Object.values(map);
  }

  // Catmull-Rom spline interpolation through nodes sorted by s.
  // tension=0   → piecewise linear (sharp corners at every node)
  // tension=0.5 → standard Catmull-Rom (smooth curves passing through every node)
  //
  // The curve ALWAYS passes exactly through each node regardless of tension,
  // so no step/jump can occur at node boundaries.
  _nodesInterp(sorted, s, tension = 0) {
    if (sorted.length === 1) return sorted[0].f;
    if (s <= sorted[0].s) return sorted[0].f;
    if (s >= sorted[sorted.length - 1].s) return sorted[sorted.length - 1].f;

    for (let k = 0; k < sorted.length - 1; k++) {
      const p1 = sorted[k], p2 = sorted[k + 1];
      if (s < p1.s || s > p2.s) continue;

      const span = p2.s - p1.s;
      if (span <= 0) return p1.f;
      const t = (s - p1.s) / span;

      if (tension === 0) {
        return p1.f + (p2.f - p1.f) * t;
      }

      // Ghost points for end segments (mirror the adjacent node)
      const p0 = k > 0                 ? sorted[k - 1] : { s: p1.s - (p2.s - p1.s), f: p1.f - (p2.f - p1.f) };
      const p3 = k < sorted.length - 2 ? sorted[k + 2] : { s: p2.s + (p2.s - p1.s), f: p2.f + (p2.f - p1.f) };

      // Catmull-Rom tangents, chord-length normalised for non-uniform node spacing
      const m1 = tension * (p2.f - p0.f) / (p1.s - p0.s + span) * span;
      const m2 = tension * (p3.f - p1.f) / (span + p3.s - p2.s) * span;

      // Cubic Hermite basis
      const t2 = t * t, t3 = t2 * t;
      return (2*t3 - 3*t2 + 1) * p1.f
           + (t3  - 2*t2  + t) * m1
           + (-2*t3 + 3*t2)    * p2.f
           + (t3  - t2)        * m2;
    }
    return sorted[sorted.length - 1].f;
  }

  // Apply all envelope nodes onto currentF0.
  // ALWAYS starts from preEnvelopeF0 so repeated calls are idempotent.
  // The smooth slider controls Catmull-Rom tension (0=linear, fully curved at max).
  // The curve always passes exactly through every node — no blending needed.
  //
  // Restore strategy: we restore the FULL voiced segment (segStart..segEnd),
  // not just the current node span [lo..hi].  This prevents a "dirty window"
  // bug where dragging a node leftward and then back right leaves ghost spline
  // values in the frames the node swept through but no longer covers.
  // Frames outside any segment (e.g. a transposition on a different region)
  // are still never touched — segStart/segEnd are clamped to each voiced run.
  _applyEnvelopeNodes() {
    if (!this.preEnvelopeF0) return;
    const src     = this.preEnvelopeF0;
    const sliderV = parseFloat(document.getElementById('f0ed-env-smooth').value) || 0;
    // Map slider 0..100 -> tension 0..0.5, squared for gentle low-end response
    const t01     = (sliderV / 100);
    const tension = t01 * t01 * 0.5;

    const groups = this._envGroupsBySegment();
    for (const nodes of groups) {
      if (nodes.length < 2) continue;
      const sorted  = [...nodes].sort((a, b) => a.s - b.s);
      const lo      = sorted[0].s;
      const hi      = sorted[sorted.length - 1].s;
      // All nodes in a group share the same segStart/segEnd (keyed that way).
      const segLo   = nodes[0].segStart;
      const segHi   = nodes[0].segEnd;

      // Step 1: restore the whole segment so no frame can carry a stale spline
      // value from a previous drag position.
      for (let i = segLo; i <= segHi; i++) this.currentF0[i] = src[i];

      // Step 2: write the spline only over the current node span [lo, hi].
      for (let i = lo; i <= hi; i++) {
        if (src[i] <= 0) continue;
        this.currentF0[i] = Math.max(20, this._nodesInterp(sorted, i, tension));
      }
    }
  }

  // Hit-test: returns index of node whose square contains (x,y), or -1
  _findNearestNode(x, y) {
    const HIT = 7; // half SZ + 1
    for (let i = 0; i < this.envelopeNodes.length; i++) {
      const px = this._sampleToX(this.envelopeNodes[i].s);
      const py = this._freqToY(this.envelopeNodes[i].f);
      if (Math.abs(x - px) <= HIT && Math.abs(y - py) <= HIT) return i;
    }
    return -1;
  }

  // Double-click: place a new envelope node snapped to the curve
  _onDblClick(e) {
    if (this.activeTool !== 'envelope') return;
    e.preventDefault();
    const { x } = this._getCanvasPos(e);
    const s     = this._xToSample(x);
    const f     = this.currentF0[s];
    if (f <= 0) return;   // no nodes on silent frames

    // Avoid duplicate (within 3 samples)
    if (this.envelopeNodes.some(n => Math.abs(n.s - s) < 3)) return;

    const seg = this._getSegmentBounds(s);
    if (!seg) return;

    // On first node: snapshot the curve BEFORE any envelope is applied.
    // All future _applyEnvelopeNodes() calls start from this clean base.
    if (this.envelopeNodes.length === 0) {
      this.preEnvelopeF0 = Float32Array.from(this.currentF0);
    }

    this.envelopeNodes.push({ s, f, segStart: seg.lo, segEnd: seg.hi });
    // Apply immediately on placement so the curve snaps to the interpolated
    // shape as you build out your control points — standard envelope behaviour.
    if (this.envelopeNodes.length >= 2) {
      this._applyEnvelopeNodes();
    }
    this._pushHistory();
    this._renderAll();
  }

  // ── Tool application ─────────────────────────────────────────────────────
  // Helper: returns false when enforceBoundaries is on and the frame was
  // originally unvoiced — prevents any tool extending into silent regions.
  _canWrite(i) {
    return !this.enforceBoundaries || this.originalF0[i] > 0;
  }

  _applyPen(sample, freq, prevSample, prevFreq) {
    if (prevSample === null || prevSample === sample) {
      if (this._canWrite(sample)) this.currentF0[sample] = freq;
      return;
    }
    const lo = Math.min(prevSample, sample), hi = Math.max(prevSample, sample);
    for (let i = lo; i <= hi; i++) {
      if (!this._canWrite(i)) continue;
      const t = hi === lo ? 0 : (i - lo) / (hi - lo);
      const f = prevSample < sample
        ? prevFreq + (freq - prevFreq) * t
        : freq     + (prevFreq - freq) * t;
      this.currentF0[i] = Math.max(0, f);
    }
  }
  _applyEraser(sample, prevSample) {
    const lo = prevSample !== null ? Math.min(prevSample, sample) : sample;
    const hi = prevSample !== null ? Math.max(prevSample, sample) : sample;
    // Eraser always works — silencing voiced frames is always intentional.
    for (let i = lo; i <= hi; i++) this.currentF0[i] = 0;
  }
  _applySmoothBrush(centerSample, radius) {
    const sigma = radius / 2;
    const lo    = Math.max(0, centerSample - radius * 2);
    const hi    = Math.min(this.currentF0.length - 1, centerSample + radius * 2);
    const patch = new Float32Array(hi - lo + 1);
    for (let i = lo; i <= hi; i++) patch[i - lo] = this.currentF0[i];
    const smoothed = gaussSmooth(patch, sigma);
    for (let i = lo; i <= hi; i++) {
      if (!this._canWrite(i)) continue;
      this.currentF0[i] = smoothed[i - lo];
    }
  }
  _applyVibratoToRegion(lo, hi, rateBps, depthCents) {
    const ts = this.timeStepMs / 1000;
    for (let i = lo; i <= hi; i++) {
      if (this.currentF0[i] <= 0) continue;
      if (!this._canWrite(i)) continue;
      // Phase is relative to the stroke start (lo) so each painted region
      // always begins at phase=0, keeping separate strokes independent.
      const t     = (i - lo) * ts;
      const phase = 2 * Math.PI * rateBps * t;
      const mult  = Math.pow(2, (depthCents / 100) * Math.sin(phase) / 12);
      this.currentF0[i] *= mult;
    }
  }
  // ── Pitch preview via Web Audio API ─────────────────────────────────────
  _previewPitch() {
    if (this.selStart === null || this.selEnd === null) {
      document.getElementById('f0ed-sel-preview-status').textContent = 'no selection';
      return;
    }
    const lo  = Math.min(this.selStart, this.selEnd);
    const hi  = Math.max(this.selStart, this.selEnd);
    const f0  = this.currentF0;
    const hop = this.timeStepMs / 1000;   // seconds per f0 frame

    // Render at 44100 Hz. Each f0 frame covers hopSamples audio samples.
    const SR         = 44100;
    const hopSamples = Math.round(hop * SR);
    const nFrames    = hi - lo + 1;
    const totalSamp  = nFrames * hopSamples;
    const buf        = new Float32Array(totalSamp);
    const twoPi      = 2 * Math.PI;

    const fLo = this.selFreqLo ?? 0;
    const fHi = this.selFreqHi ?? Infinity;

    // Render raw sine — GainNode handles all fades at audio-thread precision
    let phase = 0;
    for (let fi = 0; fi < nFrames; fi++) {
      const freq = f0[lo + fi];
      // Silence frames outside the freq band (treat as unvoiced for preview)
      if (freq <= 0 || freq < fLo || freq > fHi) { phase = 0; continue; }
      const phaseInc = twoPi * freq / SR;
      const base     = fi * hopSamples;
      for (let s = 0; s < hopSamples; s++) {
        buf[base + s] = Math.sin(phase) * 0.6;
        phase += phaseInc;
        if (phase > twoPi) phase -= twoPi;
      }
    }

    // Stop any current preview, then play using a GainNode for click-free
    // fades — gain automation is sample-accurate in the audio thread, unlike
    // buffer-level amplitude which gets block-quantized and clicks.
    try {
      if (this._previewCtx) { try { this._previewCtx.close(); } catch(_){} }
      this._previewCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
      const ctx      = this._previewCtx;
      const duration = totalSamp / SR;
      const fadeT    = Math.min(0.040, duration * 0.15); // 40 ms or 15% whichever shorter

      const abuf = ctx.createBuffer(1, totalSamp, SR);
      abuf.copyToChannel(buf, 0);

      const src  = ctx.createBufferSource();
      src.buffer = abuf;

      // GainNode: ramp 0→1 at start, 1→0 at end — sub-sample accurate
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(1, ctx.currentTime + fadeT);
      gain.gain.setValueAtTime(1, ctx.currentTime + duration - fadeT);
      gain.gain.linearRampToValueAtTime(0, ctx.currentTime + duration);

      src.connect(gain);
      gain.connect(ctx.destination);

      const status = document.getElementById('f0ed-sel-preview-status');
      status.textContent = '▶ playing…';
      src.onended = () => { status.textContent = ''; };
      src.start(ctx.currentTime);
    } catch(err) {
      console.error('[F0Editor] preview error:', err);
      document.getElementById('f0ed-sel-preview-status').textContent = 'audio error';
    }
  }

  transposeSelection(semitones) {
    if (this.selStart === null || this.selEnd === null) return;
    const lo   = Math.min(this.selStart, this.selEnd);
    const hi   = Math.max(this.selStart, this.selEnd);
    const mult = Math.pow(2, semitones / 12);
    // Freq bounds: if rect selection has them, only transpose frames inside the band.
    const fLo  = this.selFreqLo ?? 0;
    const fHi  = this.selFreqHi ?? Infinity;
    for (let i = lo; i <= hi; i++) {
      const f = this.currentF0[i];
      if (f > 0 && f >= fLo && f <= fHi)
        this.currentF0[i] = clamp(f * mult, 20, 1200);
    }
    this._pushHistory();
    this._renderAll();
  }

  // ── Mouse events ─────────────────────────────────────────────────────────
  _getCanvasPos(e) {
    const rect = this.ovCanvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _onMouseDown(e) {
    if (!this.currentF0.length) return;
    e.preventDefault();

    // Right-click → Y-axis pan
    if (e.button === 2) {
      this.isYPanning   = true;
      this.yPanStartY   = e.clientY;
      this.yPanStartMin = this.yMin;
      this.yPanStartMax = this.yMax;
      return;
    }

    // Middle-click → pan both X and Y
    if (e.button === 1) {
      this.isPanning    = true;
      this.panStartX    = e.clientX;
      this.panStartY    = e.clientY;
      this.panStartVS   = this.viewStart;
      this.panStartVE   = this.viewEnd;
      this.yPanStartMin = this.yMin;
      this.yPanStartMax = this.yMax;
      return;
    }

    const { x, y } = this._getCanvasPos(e);
    const sample    = this._xToSample(x);
    const freq      = this._yToFreq(y);

    // Envelope tool: only action is starting a node drag
    if (this.activeTool === 'envelope') {
      const idx = this._findNearestNode(x, y);
      if (idx >= 0) {
        this.envDragging = idx;
        // envDragBase is no longer needed; preEnvelopeF0 is the authoritative base
      }
      return;
    }

    this.isDrawing  = true;
    this.lastSample = sample;
    this.lastFreq   = freq;
    this.smoothPts  = [];
    this.smoothHead = null;

    if (this.activeTool === 'line') {
      this.lineStart = { s: sample, f: freq };
    } else if (this.activeTool === 'select') {
      this.selStart  = sample;
      this.selEnd    = sample;
      this.selFreqLo = freq;
      this.selFreqHi = freq;
    } else if (this.activeTool === 'vibrato') {
      // Snapshot current curve at stroke start. Every mousemove re-applies
      // vibrato from this clean base → no accumulation within one stroke.
      // vibratoMax tracks the rightmost sample reached; going left never
      // expands the region so backwards movement has no effect.
      this.vibratoBase = Float32Array.from(this.currentF0);
      this.vibratoMax  = sample;
      this.vibratoMin  = sample;
    }
  }

  _onMouseMove(e) {
    if (!this.currentF0.length) return;
    const { x, y } = this._getCanvasPos(e);
    this._lastMouseX = x; this._lastMouseY = y;

    // Status bar update
    const sample  = this._xToSample(x);
    const curFreq = this.currentF0[sample] || 0;
    document.getElementById('f0ed-cur-time').textContent =
      (sample * this.timeStepMs / 1000).toFixed(3);
    document.getElementById('f0ed-cur-freq').textContent =
      curFreq > 0 ? curFreq.toFixed(1) : '(unvoiced)';
    document.getElementById('f0ed-cur-note').textContent =
      curFreq > 0 ? hzToNoteName(curFreq) : '–';

    // Y pan (right-drag) — inverted: drag down → see lower frequencies
    if (this.isYPanning) {
      const dy      = e.clientY - this.yPanStartY;
      const range   = this.yPanStartMax - this.yPanStartMin;
      const hzPerPx = range / this.mainCanvas.height;
      const delta   = dy * hzPerPx;
      let newMin = this.yPanStartMin + delta;
      let newMax = this.yPanStartMax + delta;
      if (newMin < 20)   { newMin = 20;   newMax = 20 + range; }
      if (newMax > 8000) { newMax = 8000; newMin = 8000 - range; }
      this.yMin = newMin;
      this.yMax = newMax;
      this._renderAll();
      return;
    }

    // Middle-drag → pan both X and Y (inverted: drag feels like pushing the content)
    if (this.isPanning) {
      // X pan: drag right → see content to the right (later in time)
      const dx      = e.clientX - this.panStartX;
      const ww      = this.ovCanvas.width;
      const viewLen = this.panStartVE - this.panStartVS;
      const shift   = Math.round(dx / ww * viewLen);
      const newVS   = clamp(this.panStartVS - shift, 0, this.currentF0.length - viewLen);
      this.viewStart = newVS;
      this.viewEnd   = newVS + viewLen;

      // Y pan: drag down → view moves down (see lower frequencies)
      const dy       = e.clientY - this.panStartY;
      const range    = this.yPanStartMax - this.yPanStartMin;
      const hzPerPx  = range / this.mainCanvas.height;
      const deltaHz  = dy * hzPerPx;
      let newMin = this.yPanStartMin + deltaHz;
      let newMax = this.yPanStartMax + deltaHz;
      if (newMin < 20)   { newMin = 20;   newMax = 20 + range; }
      if (newMax > 8000) { newMax = 8000; newMin = 8000 - range; }
      this.yMin = newMin;
      this.yMax = newMax;

      this._renderAll(); return;
    }

    // Envelope node drag — X+Y, clamped to that node's voiced segment
    if (this.envDragging !== null) {
      const n      = this.envelopeNodes[this.envDragging];
      const rawS   = this._xToSample(x);
      const rawF   = this._yToFreq(y);
      // Clamp sample to the segment this node was placed on
      n.s = clamp(rawS, n.segStart, n.segEnd);
      n.f = Math.max(20, rawF);
      // Restore from pre-envelope snapshot, then re-apply all nodes cleanly
      this._applyEnvelopeNodes();
      this._renderAll(); return;
    }

    if (!this.isDrawing) { this._drawOverlay(); return; }

    const freq = this._yToFreq(y);
    const tool = this.activeTool;

    if (tool === 'pen') {
      this._applyPen(sample, freq, this.lastSample, this.lastFreq);
    } else if (tool === 'smoothpen') {
      const alpha = 1.0 - (parseFloat(document.getElementById('f0ed-smooth-amount').value) || 0.5);
      if (!this.smoothHead) {
        this.smoothHead = { x: sample, y: freq };
      } else {
        this.smoothHead.x += alpha * (sample - this.smoothHead.x);
        this.smoothHead.y += alpha * (freq   - this.smoothHead.y);
      }
      const sx = Math.round(this.smoothHead.x);
      const sy = this.smoothHead.y;
      this._applyPen(sx, sy, this.lastSample, this.lastFreq);
    } else if (tool === 'eraser') {
      this._applyEraser(sample, this.lastSample);
    } else if (tool === 'smooth') {
      const r = parseInt(document.getElementById('f0ed-brushradius').value) || 6;
      this._applySmoothBrush(sample, r);
    } else if (tool === 'select') {
      this.selEnd    = sample;
      this.selFreqLo = Math.min(this.selFreqLo ?? freq, freq);
      this.selFreqHi = Math.max(this.selFreqHi ?? freq, freq);
    } else if (tool === 'vibrato') {
      if (this.vibratoBase) {
        // Expand rightward only — moving left never shrinks or re-amplifies
        if (sample > this.vibratoMax) this.vibratoMax = sample;
        // Re-apply from clean snapshot each frame so result is always identical
        // regardless of mouse speed, jitter, or retracing
        const lo   = this.vibratoMin;
        const hi   = this.vibratoMax;
        const rate = parseFloat(document.getElementById('f0ed-vibrato-rate').value)  || 5;
        const dep  = parseFloat(document.getElementById('f0ed-vibrato-depth').value) || 25;
        for (let i = lo; i <= hi; i++) this.currentF0[i] = this.vibratoBase[i];
        this._applyVibratoToRegion(lo, hi, rate, dep);
      }
    }

    if (tool === 'smoothpen' && this.smoothHead) {
      this.lastSample = Math.round(this.smoothHead.x);
      this.lastFreq   = this.smoothHead.y;
    } else {
      this.lastSample = sample;
      this.lastFreq   = freq;
    }
    this._renderAll();
  }

  _onMouseUp(e) {
    if (this.isYPanning) { this.isYPanning = false; return; }
    if (this.isPanning)  { this.isPanning  = false; return; }

    // Finish envelope node drag
    if (this.envDragging !== null) {
      this.envDragging = null;
      this._pushHistory();
      this._renderAll();
      return;
    }

    if (!this.isDrawing) return;
    this.isDrawing = false;

    const { x, y } = this._getCanvasPos(e);
    const sample   = this._xToSample(x);
    const freq     = this._yToFreq(y);

    if (this.activeTool === 'vibrato') {
      this.vibratoBase = null;
      this.vibratoMax  = undefined;
      this.vibratoMin  = undefined;
    } else if (this.activeTool === 'smoothpen') {
      this.smoothHead = null;
    } else if (this.activeTool === 'line' && this.lineStart) {
      const s0 = this.lineStart.s, f0 = this.lineStart.f;
      const lo = Math.min(s0, sample), hi = Math.max(s0, sample);
      const span = sample - s0;
      for (let i = lo; i <= hi; i++) {
        if (!this._canWrite(i)) continue;
        const t = span !== 0 ? (i - s0) / span : 0;
        this.currentF0[i] = Math.max(0, f0 + (freq - f0) * t);
      }
      this.lineStart = null;
    }
    this._pushHistory();
    this._renderAll();
  }

  _onWheel(e) {
    e.preventDefault();
    const { x, y } = this._getCanvasPos(e);

    if (e.altKey) {
      // Alt+scroll → zoom Y axis around cursor frequency
      const pivotHz = this._yToFreq(y);
      const factor  = e.deltaY > 0 ? 1.25 : 0.8;
      this._zoomY(factor, pivotHz);
      return;
    }

    if (e.shiftKey) {
      // Shift+scroll → pan X axis (horizontal scroll)
      const vLen  = this.viewEnd - this.viewStart;
      const shift = Math.round(vLen * 0.15 * (e.deltaY > 0 ? 1 : -1));
      const newVS = clamp(this.viewStart + shift, 0, this.currentF0.length - vLen);
      this.viewStart = newVS;
      this.viewEnd   = newVS + vLen;
      this._renderAll();
      return;
    }

    // Default: zoom X axis around cursor position
    const pivot  = this._xToSample(x);
    const vLen   = this.viewEnd - this.viewStart;
    const factor = e.deltaY > 0 ? 1.25 : 0.8;
    let newLen   = clamp(Math.round(vLen * factor), 10, this.currentF0.length);
    const ratio  = (pivot - this.viewStart) / vLen;
    let newStart = Math.round(pivot - ratio * newLen);
    newStart     = clamp(newStart, 0, this.currentF0.length - newLen);
    this.viewStart = newStart;
    this.viewEnd   = newStart + newLen;
    this._renderAll();
  }

  // ── UI binding ───────────────────────────────────────────────────────────
  _activateTool(name) {
    this.activeTool = name;
    document.querySelectorAll('#f0ed-toolbar button[id^="f0ed-tool-"]').forEach(b => {
      b.classList.toggle('active', b.id === 'f0ed-tool-' + name);
    });
    const sub    = document.getElementById('f0ed-subtoolbar');
    const subMap = {
      smoothpen: 'f0ed-sub-smoothpen',
      vibrato:   'f0ed-sub-vibrato',
      select:    'f0ed-sub-select',
      smooth:    'f0ed-sub-smooth',
      envelope:  'f0ed-sub-envelope',
    };
    Object.values(subMap).forEach(id => {
      document.getElementById(id).style.display = 'none';
    });
    if (subMap[name]) {
      document.getElementById(subMap[name]).style.display = '';
      sub.classList.add('visible');
    } else {
      sub.classList.remove('visible');
    }
    // Keep selection highlight visible when switching away from select tool
    this._drawOverlay();
  }

  _bindUI() {
    // Drawing tools
    ['pen','smoothpen','line','envelope','select','smooth','vibrato','eraser'].forEach(t => {
      const el = document.getElementById('f0ed-tool-' + t);
      if (el) el.addEventListener('click', () => this._activateTool(t));
    });

    document.getElementById('f0ed-undo').addEventListener('click',       () => this.undo());
    document.getElementById('f0ed-redo').addEventListener('click',       () => this.redo());
    document.getElementById('f0ed-reset').addEventListener('click',      () => this.reset());
    document.getElementById('f0ed-reset-view').addEventListener('click', () => this.resetView());
    // Spec toggle
    document.getElementById('f0ed-spec-toggle').addEventListener('click', () => {
      this.specVisible = !this.specVisible;
      document.getElementById('f0ed-spec-toggle')
        .classList.toggle('active-green', this.specVisible);
      this._renderAll();
    });

    // Spec opacity
    document.getElementById('f0ed-spec-opacity').addEventListener('input', e => {
      this.specOpacity = parseInt(e.target.value) / 100;
      this._renderAll();
    });

    // Ghost F0 toggle
    document.getElementById('f0ed-ghost-toggle').addEventListener('click', () => {
      this.ghostVisible = !this.ghostVisible;
      document.getElementById('f0ed-ghost-toggle')
        .classList.toggle('active-green', this.ghostVisible);
      this._renderAll();
    });

    // Ghost F0 opacity
    document.getElementById('f0ed-ghost-opacity').addEventListener('input', e => {
      this.ghostOpacity = parseInt(e.target.value) / 100;
      this._renderAll();
    });

    // Enforce pitch boundaries toggle
    document.getElementById('f0ed-enforce').addEventListener('click', () => {
      this.enforceBoundaries = !this.enforceBoundaries;
      document.getElementById('f0ed-enforce')
        .classList.toggle('active-green', this.enforceBoundaries);
    });

    // Info modal
    document.getElementById('f0ed-info').addEventListener('click', () => {
      document.getElementById('f0ed-modal').classList.add('open');
    });
    document.getElementById('f0ed-modal-close').addEventListener('click', () => {
      document.getElementById('f0ed-modal').classList.remove('open');
    });
    document.getElementById('f0ed-modal').addEventListener('click', e => {
      if (e.target === document.getElementById('f0ed-modal'))
        document.getElementById('f0ed-modal').classList.remove('open');
    });

    // Envelope sub-toolbar
    // Smooth slider: re-apply nodes live when changed
    const envSmooth = document.getElementById('f0ed-env-smooth');
    const envSmVal  = document.getElementById('f0ed-env-smooth-val');
    if (envSmooth && envSmVal) {
      envSmooth.addEventListener('input', () => {
        envSmVal.textContent = envSmooth.value;
        if (this.envelopeNodes.length >= 2 && this.preEnvelopeF0) {
          this._applyEnvelopeNodes();
          this._renderAll();
        }
      });
      // Push to undo history only when the user releases the slider,
      // so scrubbing doesn't flood the history stack.
      envSmooth.addEventListener('change', () => {
        if (this.envelopeNodes.length >= 2 && this.preEnvelopeF0) {
          this._pushHistory();
        }
      });
    }
    document.getElementById('f0ed-env-clear').addEventListener('click', () => {
      // Restore the full segment (segStart..segEnd) for each group so any frames
      // the node swept through during dragging are also cleaned up, not just the
      // final [lo, hi] span.  Everything outside all segments is left as-is.
      if (this.preEnvelopeF0) {
        const groups = this._envGroupsBySegment();
        for (const nodes of groups) {
          if (nodes.length < 1) continue;
          const segLo = nodes[0].segStart;
          const segHi = nodes[0].segEnd;
          for (let i = segLo; i <= segHi; i++) {
            this.currentF0[i] = this.preEnvelopeF0[i];
          }
        }
      }
      this.envelopeNodes = [];
      this.envDragging   = null;
      this.preEnvelopeF0 = null;
      this._pushHistory();
      this._renderAll();
    });

    // Slider live labels (other tools)
    const sliderLabel = (sliderId, labelId) => {
      const s = document.getElementById(sliderId), l = document.getElementById(labelId);
      if (s && l) s.addEventListener('input', () => { l.textContent = s.value; });
    };
    sliderLabel('f0ed-smooth-amount', 'f0ed-smooth-val');
    sliderLabel('f0ed-vibrato-rate',  'f0ed-vrate-val');
    sliderLabel('f0ed-vibrato-depth', 'f0ed-vdepth-val');
    sliderLabel('f0ed-brushradius',   'f0ed-brush-val');

    // ± step buttons on subtoolbar sliders
    document.querySelectorAll('.f0ed-sb').forEach(btn => {
      btn.addEventListener('click', () => {
        const sl  = document.getElementById(btn.dataset.t);
        if (!sl) return;
        const dir  = parseFloat(btn.dataset.d);
        const step = parseFloat(sl.step) || 1;
        sl.value   = Math.min(parseFloat(sl.max),
                     Math.max(parseFloat(sl.min), parseFloat(sl.value) + dir * step));
        sl.dispatchEvent(new Event('input',  { bubbles: true }));
        sl.dispatchEvent(new Event('change', { bubbles: true }));
      });
    });

    // Selection transpose buttons
    document.getElementById('f0ed-sel-up-semi').addEventListener('click',   () => this.transposeSelection(1));
    document.getElementById('f0ed-sel-down-semi').addEventListener('click', () => this.transposeSelection(-1));
    document.getElementById('f0ed-sel-up-oct').addEventListener('click',    () => this.transposeSelection(12));
    document.getElementById('f0ed-sel-down-oct').addEventListener('click',  () => this.transposeSelection(-12));
    document.getElementById('f0ed-sel-clear').addEventListener('click', () => {
      this.selStart = this.selEnd = null;
      this.selFreqLo = this.selFreqHi = null;
      this._drawOverlay();
    });
    document.getElementById('f0ed-sel-preview').addEventListener('click', () => this._previewPitch());

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
      // Don't steal keypresses from text inputs / textareas
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.ctrlKey && !e.shiftKey && e.key === 'z') { e.preventDefault(); this.undo(); }
      if ((e.ctrlKey && e.shiftKey && e.key === 'Z') ||
          (e.ctrlKey && e.key === 'y'))               { e.preventDefault(); this.redo(); }
      // 1–8: switch tools — only when cursor is over the editor panel
      const toolIdx = parseInt(e.key) - 1;
      if (this.isHovered && !e.ctrlKey && !e.altKey && !e.shiftKey && toolIdx >= 0 && toolIdx < TOOL_KEYS.length) {
        this._activateTool(TOOL_KEYS[toolIdx]);
      }
    });
  }

  _bindCanvas() {
    const ov   = this.ovCanvas;
    const root = document.getElementById('f0ed-root');
    // Track hover over the whole editor panel (not just canvas) so tool shortcuts
    // don't fire when the user is typing in Gradio inputs elsewhere on the page.
    if (root) {
      root.addEventListener('mouseenter', () => { this.isHovered = true;  });
      root.addEventListener('mouseleave', () => { this.isHovered = false; });
    }
    ov.addEventListener('mousedown',   e => this._onMouseDown(e));
    ov.addEventListener('mousemove',   e => this._onMouseMove(e));
    ov.addEventListener('mouseup',     e => this._onMouseUp(e));
    ov.addEventListener('mouseleave',  e => {
      if (this.isDrawing)  this._onMouseUp(e);
      if (this.isYPanning) this.isYPanning = false;
      if (this.isPanning)  this.isPanning  = false;
      // Drop envelope drag without pushing history (drag left canvas)
      if (this.envDragging !== null) {
        this.envDragging = null;
        this._renderAll();
      }
    });
    ov.addEventListener('dblclick',    e => this._onDblClick(e));
    ov.addEventListener('wheel',       e => this._onWheel(e), { passive: false });
    ov.addEventListener('contextmenu', e => e.preventDefault());  // suppress right-click menu
  }

  _resizeObserver() {
    new ResizeObserver(() => this._renderAll()).observe(this.wrap);
  }

  // ── Gradio sync ──────────────────────────────────────────────────────────
  // Debounced: serialising the full F0 array on every keystroke/mouseup is
  // expensive. We wait 200 ms after the last call before actually writing.
  _syncToGradio() {
    if (this._syncTimer !== null) clearTimeout(this._syncTimer);
    this._syncTimer = setTimeout(() => {
      this._syncTimer = null;
      const writeToGradio = (selector, value) => {
        const el = document.querySelector(selector + ' textarea');
        if (!el) return;
        const setter = Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype, 'value'
        ).set;
        setter.call(el, value);
        el.dispatchEvent(new Event('input', { bubbles: true }));
      };
      writeToGradio('#edited_f0_output', this.toJSON());
    }, 200);
  }

  // ── Preset helpers ────────────────────────────────────────────────────────
  // Communication via two hidden Gradio textboxes:
  //   #f0ed_preset_cmd    — JS writes command JSON → Python handles → writes result
  //   #f0ed_preset_result — Python writes result JSON → JS reads via MutationObserver

  _writePresetCmd(payload) {
    const el = document.querySelector('#f0ed_preset_cmd textarea');
    if (!el) { console.warn('[F0Editor] preset_cmd textarea not found'); return; }
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value'
    ).set;
    setter.call(el, JSON.stringify({ ...payload, _t: Date.now() }));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  _setPresetStatus(msg, isError = false) {
    const el = document.getElementById('f0ed-preset-status');
    if (!el) return;
    el.textContent = msg;
    el.style.color = isError ? '#ff6677' : '#66bb88';
    if (msg) setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 3000);
  }

  _refreshPresetList(names) {
    const sel = document.getElementById('f0ed-preset-select');
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = '';
    if (!names || names.length === 0) {
      sel.innerHTML = '<option value="">— no presets saved —</option>';
    } else {
      names.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n; opt.textContent = n;
        sel.appendChild(opt);
      });
      // Restore previous selection if still present
      if (names.includes(prev)) sel.value = prev;
    }
  }

  savePreset() {
    if (!this.currentF0.length) { this._setPresetStatus('No curve loaded', true); return; }
    const nameEl = document.getElementById('f0ed-preset-name');
    const name   = (nameEl ? nameEl.value : '').trim();
    if (!name) { this._setPresetStatus('Enter a name first', true); return; }
    this._writePresetCmd({
      action: 'save',
      name:   name,
      freqs:  Array.from(this.currentF0),
    });
    this._setPresetStatus('Saving...');
  }

  loadPreset() {
    const sel  = document.getElementById('f0ed-preset-select');
    const name = sel ? sel.value : '';
    if (!name) { this._setPresetStatus('Select a preset first', true); return; }
    this._writePresetCmd({ action: 'load', name });
    this._setPresetStatus('Loading...');
  }

  importPreset() {
    const fi = document.getElementById('f0ed-preset-file-input');
    if (fi) fi.click();
  }

  _onPresetFileChosen(file) {
    if (!file) return;
    const stem = file.name.replace(/\.csv$/i, '').trim() || 'imported';
    const reader = new FileReader();
    reader.onload = (e) => {
      const text  = e.target.result || '';
      const freqs = [];
      for (const line of text.split(/\r?\n/)) {
        const v = parseFloat(line.trim());
        if (!isNaN(v)) freqs.push(v);
      }
      if (!freqs.length) { this._setPresetStatus('No valid freq data in file', true); return; }
      this._applyPresetFreqs(freqs);
      this._setPresetStatus('Applying...');
      this._writePresetCmd({ action: 'save', name: stem, freqs });
    };
    reader.onerror = () => this._setPresetStatus('Could not read file', true);
    reader.readAsText(file);
  }

  _handlePresetResult(resultJson) {
    let res;
    try { res = JSON.parse(resultJson); } catch { return; }
    if (!res) return;

    if (res.action === 'save') {
      if (res.ok) {
        this._refreshPresetList(res.names || []);
        // Auto-select the just-saved preset
        const nameEl = document.getElementById('f0ed-preset-name');
        if (nameEl) {
          const sel = document.getElementById('f0ed-preset-select');
          if (sel) sel.value = nameEl.value.trim();
        }
        this._setPresetStatus('Preset saved');
      } else {
        this._setPresetStatus((res.error || 'Save failed'), true);
      }
    } else if (res.action === 'load') {
      if (res.ok && res.freqs && res.freqs.length > 0) {
        this._applyPresetFreqs(res.freqs);
        this._setPresetStatus('Preset loaded');
      } else {
        this._setPresetStatus((res.error || 'Load failed'), true);
      }
    } else if (res.action === 'list') {
      this._refreshPresetList(res.names || []);
    }
  }

  _applyPresetFreqs(presetFreqs) {
    const n    = this.currentF0.length;
    const pLen = presetFreqs.length;
    if (!n || !pLen) return;

    let result;
    if (pLen === n) {
      // Same song — copy 1:1, no resampling needed
      result = Float32Array.from(presetFreqs);
    } else {
      // Different song length — linear resample to fit
      result = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const t  = (i / (n - 1)) * (pLen - 1);
        const lo = Math.floor(t), hi = Math.min(lo + 1, pLen - 1);
        const fLo = presetFreqs[lo], fHi = presetFreqs[hi];
        if (fLo <= 0) {
          result[i] = 0;
        } else if (fHi <= 0 || lo === hi) {
          result[i] = fLo;
        } else {
          result[i] = fLo + (fHi - fLo) * (t - lo);
        }
      }
    }
    this.currentF0 = result;
    this._pushHistory();
    this._renderAll();
    this._syncToGradio();
  }

  _bindPresets() {
    document.getElementById('f0ed-preset-save')
      ?.addEventListener('click', () => this.savePreset());
    document.getElementById('f0ed-preset-load')
      ?.addEventListener('click', () => this.loadPreset());
    document.getElementById('f0ed-preset-import')
      ?.addEventListener('click', () => this.importPreset());

    // File input: fires when user picks a file
    const fi = document.getElementById('f0ed-preset-file-input');
    if (fi) {
      fi.addEventListener('change', () => {
        if (fi.files && fi.files[0]) this._onPresetFileChosen(fi.files[0]);
        fi.value = ''; // reset so the same file can be re-imported
      });
    }

    // Watch #f0ed_preset_result for Python responses
    const attachResultObserver = () => {
      const el = document.querySelector('#f0ed_preset_result textarea');
      if (!el) { setTimeout(attachResultObserver, 500); return; }
      let lastSeen = '';
      const handle = (val) => {
        if (!val || val === lastSeen) return;
        lastSeen = val;
        this._handlePresetResult(val);
        // Reset so the same result JSON can fire again (e.g. reloading the same preset)
        lastSeen = '';
      };
      handle(el.value);
      const mo = new MutationObserver(() => handle(el.value));
      mo.observe(el, { attributes: true, childList: true, subtree: true, characterData: true });
      el.addEventListener('input', () => handle(el.value));
    };
    attachResultObserver();

    // Initial preset list load
    this._writePresetCmd({ action: 'list' });
  } // end class F0Editor
}

// ════════════════════════════════════════════════════════════════════════════
//  Boot
// ════════════════════════════════════════════════════════════════════════════
let editorInstance = null;

function ensureEditor() {
  if (editorInstance) return true;
  try {
    editorInstance = new F0Editor();
    // Wire up preset bar after editor is constructed
    editorInstance._bindPresets();
    return true;
  }
  catch(e) { console.error('[F0Editor] init failed:', e); return false; }
}

function initEditor() {
  // Wait for the Gradio textarea to appear in the DOM, then watch it for changes
  // with a MutationObserver instead of a perpetual polling loop.
  function attachObserver(el) {
    let lastSeenValue = '';

    function handleValue(val) {
      if (!val || val === 'null' || val === lastSeenValue) return;
      lastSeenValue = val;
      if (!ensureEditor()) {
        // Editor DOM not ready yet — retry once after a short delay
        setTimeout(() => handleValue(val), 200);
        return;
      }
      editorInstance.loadData(val);
      requestAnimationFrame(() => requestAnimationFrame(() => editorInstance._renderAll()));
    }

    // Fire immediately in case data is already present
    handleValue(el.value);

    // Then watch for future updates from Gradio
    const mo = new MutationObserver(() => handleValue(el.value));
    mo.observe(el, { attributes: true, childList: true, subtree: true, characterData: true });
    // Also listen to the 'input' event Gradio fires when it writes the value
    el.addEventListener('input', () => handleValue(el.value));
  }

  function waitForTextarea() {
    const el = document.querySelector('#f0_data_for_editor textarea');
    if (el) { attachObserver(el); return; }
    // Textarea not in DOM yet — poll briefly until it appears, then stop
    const timer = setInterval(() => {
      const found = document.querySelector('#f0_data_for_editor textarea');
      if (found) { clearInterval(timer); attachObserver(found); }
    }, 300);
  }

  waitForTextarea();
}

initEditor();"""
