#!/usr/bin/env python3
# bat_no_bat.py — batch Bat/NoBat with RAM-safe PNGs + report

import os, time, math, gc
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
import cv2
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm

# GUI
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# Headless plotting / fast colormap path
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# ----------------------------
# Shared analysis parameters
# ----------------------------
NPERSEG = 1024
CHUNK_DURATION_S = 10
OVERLAP_S = 1

# Phase 1 noise scan defaults
NOISE_WIN_S = 2.0
NOISE_HOP_S = 1.0
NOISE_KEEP_K = 8
THRESH_C = 3.0
THRESH_OFFSET_DB = 2.0

# Mask
MASK_ENABLE_DEFAULT = True
MASK_MID_HZ = 11_000.0
MASK_WIDTH_HZ = 2_500.0
MASK_STRENGTH_DB = 12.0

# Phase 2
USE_ELONGATION_FILTER = True
MIN_ELONGATION = 0.40
DBSCAN_MIN_SAMPLES_DEFAULT = 5

# Light-ridge
LR_TOP_PCT = 10.0
LR_SMOOTH_WIN = 3
LR_J_MIN_DEFAULT = 0.30
LR_VERT_SCALE = 2.0e5
LR_DOWN_SCALE = 1.0e5
LR_ELBOW_WIN = 2
LR_SPREAD_MAX_HZ = 3000.0
LR_ENERGY_REL_MIN = 0.40
LR_USE_TRUNC_BEFORE_ELBOW = True
LR_TAIL_COLS = 4
ECHO_MIN_NEG_SLOPE_HZ_PER_S = -120_000.0
MIN_RIDGE_DT_S = 0.003
MIN_RIDGE_COLS = 4

# ----------------------------
# RAM-safe spectrogram (batch-spectro style)
# ----------------------------
@dataclass
class SpectroCfg:
    n_fft: int = 1024
    hop: int = 256
    min_hz: float = 0.0
    max_hz: float = float("inf")
    dyn_range_db: float = 60.0
    ref_power: float = 1.0
    max_width_px: int = 4000
    cmap: str = "magma"
    dpi: int = 150
    style: str = "fast"       # "fast" | "mpl"
    invert_freq: bool = True  # fast path
    px_per_khz: float = 6.0   # fast path height mapping
    axis_fontsize: int = 10
    title_fontsize: int = 12
    colorbar: bool = False

def hann_win(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n, dtype=np.float32) / n)

def frame_with_overlap(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        return np.empty((0, n_fft), dtype=np.float32)
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(hop * stride, stride)
    )

def stft_mag_db_chunked(path: Path, cfg: SpectroCfg):
    """Stream file → band-limited, time-pooled spectrogram (dB → 0..1)."""
    info = sf.info(str(path))
    sr = int(info.samplerate)
    n_frames_total = int(info.frames)
    if sr <= 0 or n_frames_total <= 0:
        return np.zeros((1, 1), np.float32), np.array([0.0], np.float32), dict(sr=sr, seconds_per_col=0.0, duration_s=0.0)

    freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / sr).astype(np.float32)
    fmin, fmax = float(cfg.min_hz), float(cfg.max_hz)
    if fmax < fmin: fmin, fmax = fmax, fmin
    fmask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(fmask): fmask[:] = True

    approx_total_frames = max(1, int(math.ceil(n_frames_total / cfg.hop)))
    decim = max(1, approx_total_frames // cfg.max_width_px)
    seconds_per_col = (decim * cfg.hop) / float(sr)

    win = hann_win(cfg.n_fft).astype(np.float32)
    eps = 1e-12

    pooled_cols = []
    pool_vec = None
    pool_count = 0

    blocksize = cfg.hop * 100
    overlap = cfg.n_fft - cfg.hop
    blocks = sf.blocks(str(path), blocksize=blocksize, overlap=overlap, dtype="float32", always_2d=True)

    for block in blocks:
        x = block[:, 0] if block.shape[1] > 0 else block.reshape(-1)
        if x.size < cfg.n_fft:
            continue
        frames = frame_with_overlap(x, cfg.n_fft, cfg.hop)
        if frames.size == 0:
            continue
        frames = frames * win
        X = np.fft.rfft(frames, axis=1)
        P = (X.real * X.real + X.imag * X.imag).astype(np.float32)
        P = P[:, fmask]
        S_db = 10.0 * np.log10(np.maximum(P / max(cfg.ref_power, eps), eps))

        for i in range(S_db.shape[0]):
            if pool_vec is None:
                pool_vec = S_db[i].copy()
                pool_count = 1
            else:
                np.maximum(pool_vec, S_db[i], out=pool_vec)
                pool_count += 1
            if pool_count >= decim:
                pooled_cols.append(pool_vec.copy())
                pool_vec = None
                pool_count = 0

    if pool_vec is not None:
        pooled_cols.append(pool_vec.copy())

    if not pooled_cols:
        S = np.zeros((np.count_nonzero(fmask), 1), dtype=np.float32)
    else:
        S = np.stack(pooled_cols, axis=1)  # (n_bins, n_cols)

    Smax = float(np.max(S)) if S.size else 0.0
    S_clamped = np.clip(S, Smax - cfg.dyn_range_db, Smax)
    rng = max(cfg.dyn_range_db, 1e-6)
    S01 = (S_clamped - (Smax - cfg.dyn_range_db)) / rng
    S01 = np.clip(S01, 0.0, 1.0).astype(np.float32)

    duration_s = S01.shape[1] * seconds_per_col
    meta = dict(sr=sr, seconds_per_col=seconds_per_col, duration_s=float(duration_s),
                fmin=float(freqs[fmask].min()) if np.any(fmask) else 0.0,
                fmax=float(freqs[fmask].max()) if np.any(fmask) else 0.0)
    return S01, freqs[fmask], meta

def save_png_fast(S01, freqs, out_png: Path, cfg: SpectroCfg, title: str | None = None):
    arr = np.clip(S01.astype("float32", copy=False), 0.0, 1.0)
    if cfg.invert_freq:
        arr = np.flipud(arr)
    rgba = (cm.get_cmap(cfg.cmap)(arr) * 255).astype("uint8")  # (H,W,4)

    # height from freq span
    f_span_khz = max(1.0, (float(freqs.max()) - float(freqs.min())) / 1000.0) if freqs.size >= 2 else 1.0
    tgt_h = max(32, int(round(f_span_khz * cfg.px_per_khz)))
    tgt_w = rgba.shape[1]

    img = Image.fromarray(rgba, mode="RGBA")
    if tgt_h != rgba.shape[0]:
        img = img.resize((tgt_w, tgt_h), Image.Resampling.BICUBIC)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)

def save_png_mpl(S01, freqs, meta: dict, out_png: Path, cfg: SpectroCfg, title: str):
    H, W = S01.shape
    extent = (0.0, meta.get("duration_s", W), float(freqs.min()), float(freqs.max()))
    fig = plt.figure(figsize=(10, 7), dpi=cfg.dpi)
    ax = fig.add_subplot(111)
    ax.imshow(S01, aspect="auto", origin="lower",
              extent=(extent[0], extent[1], extent[2], extent[3]),
              cmap=cfg.cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Time (s)", fontsize=cfg.axis_fontsize)
    ax.set_ylabel("Frequency (Hz)", fontsize=cfg.axis_fontsize)
    ax.set_title(title, fontsize=cfg.title_fontsize)
    if cfg.colorbar:
        cbar = plt.colorbar(ax.images[0], ax=ax)
        cbar.set_label("Normalized Intensity", fontsize=cfg.axis_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=cfg.axis_fontsize)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)



def _add_bat_image(self, path="batcrossingguard.png", max_w=420):
    try:
        img = Image.open(path)
        if img.width > max_w:
            h = int(img.height * (max_w / float(img.width)))
            img = img.resize((max_w, h), Image.Resampling.LANCZOS)
        self._bat_img_tk = ImageTk.PhotoImage(img)
        self.bat_label = ttk.Label(self.center, image=self._bat_img_tk)  # <—
        self.bat_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    except Exception as e:
        print(f"[bat image] not shown: {e}")   # helpful if the path is wrong
        self.bat_label = None
        self._bat_img_tk = None


# ----------------------------
# Core analysis helpers
# ----------------------------
def calculate_spectrogram_db(audio_chunk, samplerate):
    f, t, Sxx = spectrogram(
        audio_chunk, samplerate,
        nperseg=NPERSEG,
        noverlap=NPERSEG // 2,
        detrend=False,
        scaling='density',
        mode='psd'
    )
    db_Sxx = 10 * np.log10(Sxx + 1e-10)
    return f, t, db_Sxx

def _sigmoid_mask(frequencies, mid_hz, width_hz):
    f = frequencies.reshape(-1, 1)
    return 1.0 / (1.0 + np.exp(-(f - mid_hz) / max(1.0, width_hz)))

def threshold_and_clean(db_Sxx, global_threshold, *,
                        frequencies, mask_on, mask_mid, mask_width, mask_strength_db):
    if mask_on:
        m = _sigmoid_mask(frequencies, mask_mid, mask_width)
        local_thresh = global_threshold + (1.0 - m) * mask_strength_db
    else:
        local_thresh = np.full((db_Sxx.shape[0], 1), global_threshold, dtype=np.float32)
    binary_image_noisy = db_Sxx > local_thresh
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary_image_noisy.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)

def _component_elongation(patch_binary):
    ys, xs = np.nonzero(patch_binary)
    if len(xs) < 3: return 0.0, 0.0, 0.0
    pts = np.vstack([xs, ys]).astype(np.float32).T
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    vals, _ = np.linalg.eig(cov)
    vals = np.sort(np.real(vals))[::-1]
    lam_max, lam_min = float(vals[0]), float(max(vals[1], 1e-12))
    ratio = lam_min / lam_max
    return float(1.0 - ratio), lam_max, lam_min

def _moving_avg(x, win=3):
    if win <= 1 or x is None or np.size(x) == 0: return x
    win = int(max(1, win));
    if win % 2 == 0: win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xp, np.ones(win)/win, mode='valid')

def _ridge_from_patch(patch_db, patch_binary, freq_axis, time_axis):
    F, T = patch_db.shape
    if T == 0: return None, None, None, None, None
    p = max(0.1, min(50.0, LR_TOP_PCT))
    ridge_f = np.full(T, np.nan, dtype=np.float32)
    col_energy = np.full(T, np.nan, dtype=np.float32)
    col_spread = np.full(T, np.nan, dtype=np.float32)
    for j in range(T):
        col = patch_db[:, j]
        mask_col = patch_binary[:, j] > 0
        vals = col[mask_col] if np.any(mask_col) else col
        if vals.size == 0: continue
        kth = np.percentile(vals, 100.0 - p)
        top_idx = np.where(col >= kth)[0]
        if top_idx.size == 0: top_idx = np.array([np.argmax(col)])
        top_freqs = freq_axis[top_idx]
        top_vals  = col[top_idx]
        w = np.maximum(top_vals - np.min(top_vals) + 1e-3, 1e-3)
        mu = float(np.sum(w * top_freqs) / np.sum(w))
        var = float(np.sum(w * (top_freqs - mu)**2) / np.sum(w))
        ridge_f[j] = mu
        col_spread[j] = np.sqrt(max(var, 0.0))
        col_energy[j] = float(np.mean(top_vals))
    valid = ~np.isnan(ridge_f)
    if not np.any(valid): return None, None, None, None, None
    ridge_f = _moving_avg(ridge_f[valid], win=LR_SMOOTH_WIN).astype(np.float32)
    ridge_t = time_axis[valid]
    col_energy = col_energy[valid]; col_spread = col_spread[valid]
    idx_e = int(np.argmax(ridge_f)); t_e = float(ridge_t[idx_e])
    end_idx = idx_e
    if LR_USE_TRUNC_BEFORE_ELBOW:
        emax = np.nanmax(col_energy[:idx_e+1]) if idx_e >= 0 else np.nanmax(col_energy)
        for j in range(0, idx_e+1):
            if (col_spread[j] > LR_SPREAD_MAX_HZ) or (col_energy[j] < LR_ENERGY_REL_MIN * emax):
                end_idx = max(2, j); break
    end_idx = max(end_idx, min(idx_e + LR_TAIL_COLS, ridge_f.size - 1))
    return ridge_f[:end_idx+1], ridge_t[:end_idx+1], col_energy[:end_idx+1], col_spread[:end_idx+1], t_e

def calculate_coherence(patch):
    total_pixels = int(np.sum(patch))
    if total_pixels == 0: return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=4)
    if num_labels <= 1: return 1.0
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return largest / total_pixels

def calculate_tonal_purity(patch_db, patch_binary):
    active_pixels = patch_db[patch_binary.astype(bool)]
    if len(active_pixels) < 10: return 0.0
    overall_avg = float(np.mean(active_pixels))
    top_10_threshold = float(np.percentile(active_pixels, 90))
    top_pixels = active_pixels[active_pixels >= top_10_threshold]
    top_avg = float(np.mean(top_pixels)) if len(top_pixels) > 0 else overall_avg
    return top_avg - overall_avg

def classify_cluster(duration, bandwidth, min_freq, coeffs,
                     coherence, purity, purity_threshold, *,
                     simple_slope=None):
    # duration-adaptive coherence
    if duration < 0.05: coh_thresh = 0.75
    elif duration < 0.15: coh_thresh = 0.70
    elif duration < 0.30: coh_thresh = 0.65
    else: coh_thresh = 0.60
    if coherence < coh_thresh or duration < 0.01: return "Noise"
    purity_gate = purity_threshold + (1.0 if duration < 0.04 else 0.0)
    if purity < purity_gate: return "Noise"

    # Echo
    is_in_echo_freq_band = (min_freq > 17_000.0)
    is_narrow_band       = (bandwidth < 40_000.0)
    is_short_duration    = (duration < 0.10)
    ridge_available = (simple_slope is not None)
    if ridge_available:
        has_downward_sweep = (simple_slope <= ECHO_MIN_NEG_SLOPE_HZ_PER_S)
    else:
        has_downward_sweep = (coeffs[1] < 0.0)
    if is_in_echo_freq_band and is_narrow_band and is_short_duration and has_downward_sweep:
        return "Echolocation"

    # Social
    is_above_social_min_freq = (min_freq > 18_000.0)
    is_below_social_max_dur  = (duration < 0.40)
    is_long_enough           = (duration > 0.04)
    is_wide_band             = (bandwidth > 70_000.0 and duration > 0.015)
    if is_above_social_min_freq and is_below_social_max_dur and (is_long_enough or is_wide_band):
        return "Social"

    # ShortSocial
    is_above_short_min_freq = (min_freq >= 16_000.0)
    if ridge_available and (simple_slope >= -80_000.0) and is_above_short_min_freq and duration <= 0.040:
        if (bandwidth >= 6_000.0) or (purity >= purity_threshold + 2.0) or (coherence >= 0.85):
            return "ShortSocial"

    return "Noise"

# ----------------------------
# Chunk processing
# ----------------------------
def process_chunk(args):
    (audio_chunk, samplerate, global_threshold, chunk_start_time,
     purity_threshold, mask_on, mask_mid, mask_width, mask_strength,
     use_elong, min_elong, dbscan_min_samples) = args

    if len(audio_chunk) < NPERSEG:
        return []

    frequencies, times, db_Sxx = calculate_spectrogram_db(audio_chunk, samplerate)
    binary_image = threshold_and_clean(
        db_Sxx, global_threshold,
        frequencies=frequencies,
        mask_on=mask_on,
        mask_mid=mask_mid,
        mask_width=mask_width,
        mask_strength_db=mask_strength
    )

    results = []
    points = np.argwhere(binary_image)
    if len(points) < 10: return []
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=3, min_samples=dbscan_min_samples).fit(points)

    for k in set(db.labels_):
        if k == -1: continue
        cluster_points = points[db.labels_ == k]
        if len(cluster_points) < dbscan_min_samples: continue

        min_r, max_r = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
        min_c, max_c = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])

        patch_binary = binary_image[min_r:max_r + 1, min_c:max_c + 1]
        patch_db     = db_Sxx[min_r:max_r + 1, min_c:max_c + 1]
        patch_freqs  = frequencies[min_r:max_r + 1]
        patch_times  = times[min_c:max_c + 1]

        coherence       = calculate_coherence(patch_binary)
        elongation, _, _= _component_elongation(patch_binary)
        tonal_purity    = calculate_tonal_purity(patch_db, patch_binary)

        c_times   = times[cluster_points[:, 1]]
        c_freqs   = frequencies[cluster_points[:, 0]]
        duration  = float(np.max(c_times) - np.min(c_times))
        bandwidth = float(np.max(c_freqs) - np.min(c_freqs))
        min_freq  = float(np.min(c_freqs))

        try:
            coeffs = np.polyfit(c_times, c_freqs, 2)
        except (np.linalg.LinAlgError, ValueError):
            coeffs = (0.0, 0.0, 0.0)

        ridge_f, ridge_t, _, _, _ = _ridge_from_patch(patch_db, patch_binary, patch_freqs, patch_times)
        simple_slope = None
        if ridge_f is not None and ridge_t is not None and len(ridge_f) >= MIN_RIDGE_COLS:
            dt = float(ridge_t[-1] - ridge_t[0])
            if dt >= MIN_RIDGE_DT_S:
                simple_slope = float((ridge_f[-1] - ridge_f[0]) / dt)

        # dynamic elongation gate
        e_gate = float(min_elong)
        if use_elong:
            if duration < 0.05: e_gate = max(0.0, min_elong - 0.20)
            elif duration < 0.10: e_gate = max(0.0, min_elong - 0.10)
            if elongation < e_gate:
                label = "Noise"
            else:
                label = classify_cluster(duration, bandwidth, min_freq, coeffs,
                                         coherence, tonal_purity, purity_threshold,
                                         simple_slope=simple_slope)
        else:
            label = classify_cluster(duration, bandwidth, min_freq, coeffs,
                                     coherence, tonal_purity, purity_threshold,
                                     simple_slope=simple_slope)

        start_time = chunk_start_time + float(np.min(c_times))
        end_time   = chunk_start_time + float(np.max(c_times))
        bbox = (start_time, float(np.min(c_freqs)), end_time - start_time, bandwidth)
        mean_db_active = float(np.mean(patch_db[patch_binary.astype(bool)])) if np.any(patch_binary) else float(np.mean(patch_db))

        results.append({
            'label': label,
            'start_time': start_time,
            'end_time': end_time,
            'bbox': bbox,
            'duration': duration,
            'bandwidth': bandwidth,
            'min_freq': min_freq,
            'purity': float(tonal_purity),
            'coherence': float(coherence),
            'elongation': float(elongation),
            'mean_db_active': mean_db_active,
        })
    return results

# Phase-1 (threshold + noise reference above 17k)
def noise_scan_global_threshold_and_noise_ref(filepath, samplerate,
                                              win_s=NOISE_WIN_S, hop_s=NOISE_HOP_S,
                                              keep_k=NOISE_KEEP_K, c=THRESH_C):
    medians = []
    hi_band_means = []
    win_frames = int(win_s * samplerate)
    hop_frames = int(hop_s * samplerate)
    if win_frames < NPERSEG:
        win_frames = max(NPERSEG, win_frames)
    with sf.SoundFile(filepath, 'r') as f:
        total_frames = len(f)
        for start in range(0, total_frames, hop_frames):
            f.seek(start)
            audio = f.read(win_frames)
            if len(audio) < NPERSEG:
                continue
            freqs, _, db_Sxx = calculate_spectrogram_db(audio, samplerate)
            medians.append(float(np.median(db_Sxx)))
            mask_hi = freqs >= 17_000.0
            if np.any(mask_hi):
                hi_band_means.append(float(np.mean(db_Sxx[mask_hi, :])))
    if not medians:
        return -100.0, -100.0
    medians = np.array(medians)
    k = int(max(1, min(keep_k, len(medians))))
    quiet_idx = np.argpartition(medians, k - 1)[:k]
    quiet_meds = medians[quiet_idx]
    median_med = float(np.median(quiet_meds))
    mad_med = float(np.median(np.abs(quiet_meds - median_med)))
    robust_std = 1.4826 * mad_med
    threshold = median_med + c * robust_std
    if len(hi_band_means) == len(medians):
        hi = np.array(hi_band_means)[quiet_idx]
        noise_ref_db = float(np.mean(hi)) if hi.size > 0 else float(np.mean(hi_band_means))
    else:
        noise_ref_db = float(np.mean(hi_band_means)) if hi_band_means else -100.0
    return float(threshold), float(noise_ref_db)

# Full-file analysis & verdict
def analyze_file(filepath, params):
    start_wall = time.time()
    with sf.SoundFile(filepath, 'r') as f:
        sr = f.samplerate
        total_frames = len(f)
    duration_s = total_frames / sr if sr > 0 else 0.0
    if duration_s < 0.75:
        return {
            "filepath": filepath,
            "samplerate": sr,
            "duration_s": duration_s,
            "counts": {"Echolocation": 0, "Social": 0, "ShortSocial": 0, "Noise": 0},
            "verdict": "No Bat",
            "snr_ratio": None,
            "noise_ref_db": None,
            "threshold_db": None,
            "wall_time_s": 0.0,
            "calls": [],
        }

    thr, noise_ref_db = noise_scan_global_threshold_and_noise_ref(
        filepath, sr,
        win_s=params['noise_win_s'], hop_s=params['noise_hop_s'],
        keep_k=params['noise_keep_k'], c=params['noise_c']
    )
    thr += params['noise_offset_db']

    # chunks
    chunk_frames = int(CHUNK_DURATION_S * sr)
    step_frames = max(1, chunk_frames - int(OVERLAP_S * sr))
    tasks = []
    with sf.SoundFile(filepath, 'r') as f:
        for i in range(0, total_frames, step_frames):
            f.seek(i)
            audio_chunk = f.read(chunk_frames)
            if len(audio_chunk) == 0: break
            tasks.append(
                (audio_chunk, sr, thr, i / sr,
                 params['purity_threshold'], params['mask_on'],
                 params['mask_mid'], params['mask_width'], params['mask_strength'],
                 params['use_elong'], params['min_elong'], params['dbscan_min_samples'])
            )
    calls = []
    if tasks:
        # Use spawn to be safe on some platforms
        ctx = get_context("spawn")

        # Clamp/guard cores:
        n_logical = os.cpu_count() or 2
        requested = int(params.get('cores', max(1, math.floor(0.75 * n_logical))))
        # don’t spawn more processes than CPUs or tasks
        cores = max(1, min(requested, n_logical, len(tasks)))

        if cores == 1:
            # tiny jobs: avoid pool overhead
            for res in map(process_chunk, tasks):
                calls.extend(res)
        else:
            with ctx.Pool(processes=cores) as pool:
                for res in pool.imap(process_chunk, tasks):
                    calls.extend(res)

    calls.sort(key=lambda x: x['start_time'])
    merged = []
    for c in calls:
        if not merged or c['start_time'] > merged[-1]['end_time']:
            merged.append(c)
    calls = merged

    counts = {"Echolocation": 0, "Social": 0, "ShortSocial": 0, "Noise": 0}
    for c in calls: counts[c['label']] += 1

    # Rule 1: ≥10 echos within 4s
    verdict = "No Bat"
    echo_times = [c['start_time'] for c in calls if c['label'] == "Echolocation"]
    i = 0
    for j in range(len(echo_times)):
        while echo_times[j] - echo_times[i] > 4.0:
            i += 1
        if (j - i + 1) >= 10:
            verdict = "Bat"; break
    # Rule 2: ≥2 Social
    if verdict == "No Bat" and counts["Social"] >= 2:
        verdict = "Bat"
    # Rule 3: ShortSocial density
    if verdict == "No Bat" and duration_s > 0:
        short_rate = counts["ShortSocial"] / duration_s
        if short_rate > 0.33:
            verdict = "Bat"

    snr_ratio = None
    if verdict == "Bat":
        vals = [c['mean_db_active'] for c in calls if c['label'] != "Noise"]
        mean_call_db = float(np.mean(vals)) if vals else -100.0
        snr_ratio = 10 ** ((mean_call_db - noise_ref_db) / 10.0)
        verdict = "Clean Bat" if snr_ratio is not None and snr_ratio > 1.5 else "Noisy Bat"

    wall_time = time.time() - start_wall
    return {
        "filepath": filepath,
        "samplerate": sr,
        "duration_s": duration_s,
        "counts": counts,
        "verdict": verdict,
        "snr_ratio": snr_ratio,
        "noise_ref_db": noise_ref_db,
        "threshold_db": thr,
        "wall_time_s": wall_time,
        "calls": calls,
    }

# Save PNG using RAM-safe renderer, routed by verdict
def save_file_png(filepath: str, verdict: str, out_root: Path, cfg: SpectroCfg):
    sub = "CleanBat" if verdict == "Clean Bat" else ("NoisyBat" if verdict == "Noisy Bat" else "NoBat")
    out_dir = out_root / sub
    base = Path(filepath).stem + ".png"
    out_png = out_dir / base

    S01, freqs, meta = stft_mag_db_chunked(Path(filepath), cfg)
    title = f"{Path(filepath).name} — Verdict: {verdict}"
    if cfg.style == "fast":
        save_png_fast(S01, freqs, out_png, cfg, title)
    else:
        save_png_mpl(S01, freqs, meta, out_png, cfg, title)
    return str(out_png)

# Report
def write_report_txt(all_results, out_dir: Path, total_wall_s, total_bytes):
    report_path = out_dir / "batch_report.txt"
    bucket_lt10 = sum(1 for r in all_results if r["duration_s"] < 10.0)
    bucket_gt60 = sum(1 for r in all_results if r["duration_s"] > 60.0)
    final_bins = {"Clean Bat":0, "Noisy Bat":0, "No Bat":0}
    sum_counts = {"Echolocation":0, "Social":0, "ShortSocial":0, "Noise":0}
    for r in all_results:
        final_bins[r["verdict"]] = final_bins.get(r["verdict"],0) + 1
        for k in sum_counts: sum_counts[k] += r["counts"][k]
    size_gb = total_bytes / (1024**3) if total_bytes>0 else 0.0
    time_per_gb = (total_wall_s / size_gb) if size_gb>0 else float('nan')

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Bat / No-Bat Batch Report\n=========================\n\n")
        f.write(f"Files processed: {len(all_results)}\n")
        f.write(f"Total wall time: {total_wall_s:.2f} s\n")
        f.write(f"Total input size: {size_gb:.3f} GiB\n")
        f.write(f"Time per GiB: {time_per_gb:.2f} s/GiB\n")
        f.write(f"Files < 10 s: {bucket_lt10}\n")
        f.write(f"Files > 60 s: {bucket_gt60}\n\n")
        f.write("Aggregate call counts:\n")
        for k in ["Echolocation","Social","ShortSocial","Noise"]:
            f.write(f"  {k}: {sum_counts[k]}\n")
        f.write("\nFinal verdict bins:\n")
        for k in ["Clean Bat","Noisy Bat","No Bat"]:
            f.write(f"  {k}: {final_bins.get(k,0)}\n")
        f.write("\nPer-file details:\n")
        for r in all_results:
            base = os.path.basename(r["filepath"])
            snr_txt = f"{r['snr_ratio']:.2f}" if r["snr_ratio"] is not None else "NA"
            f.write(f"- {base} | {r['verdict']} | dur={r['duration_s']:.2f}s | "
                    f"E:{r['counts']['Echolocation']} S:{r['counts']['Social']} "
                    f"SS:{r['counts']['ShortSocial']} N:{r['counts']['Noise']} | "
                    f"SNR×≈{snr_txt} | noise_ref_dB={r['noise_ref_db']:.2f}\n")
    return str(report_path)

# ----------------------------
# GUI
# ----------------------------
class BatchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bat / No-Bat Batch Classifier")
        self.root.geometry("900x560")

        self.dirpath = tk.StringVar(value="")
        self.status  = tk.StringVar(value="Choose a folder and click Run.")
        self._build_ui()
        _add_bat_image(self)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Folder:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.dirpath, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse…", command=self._pick_dir).pack(side=tk.LEFT)

        self.run_btn = ttk.Button(self.root, text="Run Batch", command=self._run_batch)
        self.run_btn.pack(pady=6)

        # >>> add this center frame (the big gray area)
        self.center = ttk.Frame(self.root)
        self.center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # <<<

        # Right-side params
        right = ttk.LabelFrame(self.root, text="Parameters", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        def add_row(parent, label, val, width=8):
            r = ttk.Frame(parent); r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label).pack(side=tk.LEFT)
            e = ttk.Entry(r, width=width); e.insert(0, val); e.pack(side=tk.LEFT, padx=4)
            return e

        # Compute
        box_comp = ttk.LabelFrame(right, text="Compute"); box_comp.pack(fill=tk.X, pady=4)
        n_logical = os.cpu_count() or 2
        default_cores = max(1, math.floor(0.75 * n_logical))
        row = ttk.Frame(box_comp); row.pack(fill=tk.X)
        ttk.Label(row, text="Cores (¾ of CPU by default):").pack(side=tk.LEFT)
        self.cores_entry = ttk.Entry(row, width=6); self.cores_entry.insert(0, str(default_cores))
        self.cores_entry.pack(side=tk.LEFT, padx=4)
        row2 = ttk.Frame(box_comp); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Purity (dB):").pack(side=tk.LEFT)
        self.purity_entry = ttk.Entry(row2, width=6); self.purity_entry.insert(0, "4.5")
        self.purity_entry.pack(side=tk.LEFT, padx=4)

        # Noise Scan
        box_noise = ttk.LabelFrame(right, text="Noise Scan"); box_noise.pack(fill=tk.X, pady=4)
        self.noise_win_entry   = add_row(box_noise, "Win (s):", f"{NOISE_WIN_S:.2f}")
        self.noise_hop_entry   = add_row(box_noise, "Hop (s):", f"{NOISE_HOP_S:.2f}")
        self.noise_k_entry     = add_row(box_noise, "Keep K:", f"{NOISE_KEEP_K}")
        self.noise_c_entry     = add_row(box_noise, "C (MAD):", f"{THRESH_C:.2f}")
        self.noise_off_entry   = add_row(box_noise, "Offset (dB):", f"{THRESH_OFFSET_DB:.1f}")

        # Mask
        box_mask = ttk.LabelFrame(right, text="Sigmoid Mask"); box_mask.pack(fill=tk.X, pady=4)
        self.mask_var = tk.BooleanVar(value=MASK_ENABLE_DEFAULT)
        ttk.Checkbutton(box_mask, text="Enable mask", variable=self.mask_var).pack(anchor=tk.W, pady=2)
        self.mask_mid_entry    = add_row(box_mask, "Mid (Hz):", f"{int(MASK_MID_HZ)}", 10)
        self.mask_width_entry  = add_row(box_mask, "Width (Hz):", f"{int(MASK_WIDTH_HZ)}", 10)
        self.mask_strength_entry = add_row(box_mask, "Strength (dB):", f"{MASK_STRENGTH_DB:.1f}", 6)

        box_style = ttk.LabelFrame(right, text="PNG Style")
        box_style.pack(fill=tk.X, pady=6)
        ttk.Label(box_style, text="Style:").pack(side=tk.LEFT)
        self.png_style = tk.StringVar(value="mpl")  # default to Matplotlib
        ttk.Combobox(box_style, textvariable=self.png_style, width=8,
                     values=("mpl", "fast"), state="readonly").pack(side=tk.LEFT, padx=6)

        # Phase 2
        box_p2 = ttk.LabelFrame(right, text="Phase 2"); box_p2.pack(fill=tk.X, pady=4)
        self.use_elong_var = tk.BooleanVar(value=USE_ELONGATION_FILTER)
        ttk.Checkbutton(box_p2, text="Use elongation filter", variable=self.use_elong_var).pack(anchor=tk.W, pady=2)
        self.min_elong_entry   = add_row(box_p2, "Min elongation:", f"{MIN_ELONGATION:.2f}", 6)
        self.dbscan_entry      = add_row(box_p2, "DBSCAN min_samples:", f"{DBSCAN_MIN_SAMPLES_DEFAULT}", 6)

        # Rendering
        box_r = ttk.LabelFrame(right, text="PNG Rendering"); box_r.pack(fill=tk.X, pady=4)
        self.style_var = tk.StringVar(value="fast")
        ttk.Radiobutton(box_r, text="Fast", variable=self.style_var, value="fast").pack(anchor=tk.W)
        ttk.Radiobutton(box_r, text="Matplotlib", variable=self.style_var, value="mpl").pack(anchor=tk.W)

        # Progress + status
        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(self.root, textvariable=self.status).pack(pady=2)

    def _pick_dir(self):
        d = filedialog.askdirectory(title="Choose folder of WAV files")
        if d: self.dirpath.set(d)

    def _collect_params(self):
        return {
            "cores": int(self.cores_entry.get()),
            "purity_threshold": float(self.purity_entry.get()),
            "noise_win_s": float(self.noise_win_entry.get()),
            "noise_hop_s": float(self.noise_hop_entry.get()),
            "noise_keep_k": int(self.noise_k_entry.get()),
            "noise_c": float(self.noise_c_entry.get()),
            "noise_offset_db": float(self.noise_off_entry.get()),
            "mask_on": bool(self.mask_var.get()),
            "mask_mid": float(self.mask_mid_entry.get()),
            "mask_width": float(self.mask_width_entry.get()),
            "mask_strength": float(self.mask_strength_entry.get()),
            "use_elong": bool(self.use_elong_var.get()),
            "min_elong": float(self.min_elong_entry.get()),
            "dbscan_min_samples": int(self.dbscan_entry.get()),
            "style": self.png_style.get(),
        }

    def _run_batch(self):
        folder = self.dirpath.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please choose a valid folder.")
            return

        wavs = [os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith((".wav", ".wave"))]
        if not wavs:
            messagebox.showwarning("No files", "No WAV/WAVE files found in that folder.")
            return

        params = self._collect_params()

        out_root = Path(folder) / "_bat_no_bat_outputs"
        for sub in ("CleanBat","NoisyBat","NoBat"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)

        total_bytes = sum((os.path.getsize(p) for p in wavs if os.path.exists(p)), 0)

        self.progress["maximum"] = len(wavs)
        self.progress["value"] = 0
        self.run_btn.config(state=tk.DISABLED)
        self.status.set("Running…")

        if getattr(self, "bat_label", None) is not None:
            self.bat_label.pack_forget()

        cfg = SpectroCfg(
            n_fft=1024, hop=256,
            min_hz=0.0, max_hz=float("inf"),
            dyn_range_db=60.0, ref_power=1.0,
            max_width_px=4000, cmap="magma",
            dpi=150, style=params["style"],
            invert_freq=True, px_per_khz=6.0,
            axis_fontsize=10, title_fontsize=12, colorbar=False
        )

        t0 = time.time()
        results = []
        try:
            for i, path in enumerate(wavs, 1):
                self.status.set(f"Processing {i}/{len(wavs)}: {os.path.basename(path)}")
                self.root.update_idletasks()

                res = analyze_file(path, params)
                results.append(res)

                # Verdict-routed PNG (RAM-safe)
                try:
                    save_file_png(path, res["verdict"], out_root, cfg)
                except Exception as e:
                    print(f"[WARN] PNG failed for {path}: {e}")

                self.progress["value"] = i
                self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Error", f"Batch error: {e}")
        finally:
            total_wall = time.time() - t0
            try:
                report_path = write_report_txt(results, out_root, total_wall, total_bytes)
            except Exception as e:
                report_path = None
                print(f"[WARN] Could not write report: {e}")

            self.status.set("Done. " + (f"Report: {report_path}" if report_path else ""))
            self.run_btn.config(state=tk.NORMAL)

            if getattr(self, "bat_label", None) is not None:
                self.bat_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

def _save_png_mpl(S01, freqs, duration_s, out_png, *, dpi=150, cmap="magma", title=None):
    """
    S01: (freq_bins, time_cols) ∈ [0,1], freqs in Hz, duration_s in seconds.
    RAM-safe: renders directly from the normalized array.
    """
    H, W = S01.shape
    fig_w = 12
    fig_h = 8
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)

    extent = (0.0, float(duration_s), float(freqs.min()) if freqs.size else 0.0,
              float(freqs.max()) if freqs.size else 1.0)

    im = ax.imshow(S01, origin="lower", aspect="auto",
                   extent=extent, cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Intensity (dB)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = BatchGUI(root)
    root.mainloop()
