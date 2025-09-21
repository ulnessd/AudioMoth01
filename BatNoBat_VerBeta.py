#!/usr/bin/env python3
# bat_no_bat_pyqt.py — A modern PyQt6 GUI for the Bat/No-Bat classifier.
# Ported from the original Tkinter version with an improved, non-freezing UI.

import multiprocessing
import os
import sys

# --- Environment setup for multiprocessing reliability ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import shutil
import time
import math
from dataclasses import dataclass
from pathlib import Path

# --- Core Scientific Libraries (Backend) ---
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
import cv2
from multiprocessing import get_context
import psutil
from PIL import Image

# --- PyQt6 Libraries (Frontend) ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
                             QMessageBox, QProgressBar, QGroupBox, QCheckBox, QFrame)
from PyQt6.QtGui import QPixmap, QIcon, QFont
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal

# --- Headless plotting for PNG generation ---
import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import matplotlib.pyplot as plt

# =============================================================================
#
# 1. BACKEND ANALYSIS LOGIC (UNCHANGED FROM ORIGINAL)
#    This entire section contains the core scientific code. It is kept
#    identical to the original script to preserve the analysis logic.
#
# =============================================================================

# --- Shared analysis parameters ---
NPERSEG = 1024
CHUNK_DURATION_S = 10
OVERLAP_S = 1
NOISE_WIN_S, NOISE_HOP_S, NOISE_KEEP_K, THRESH_C, THRESH_OFFSET_DB = 2.0, 1.0, 8, 3.0, 2.0
MASK_ENABLE_DEFAULT, MASK_MID_HZ, MASK_WIDTH_HZ, MASK_STRENGTH_DB = True, 11_000.0, 2_500.0, 12.0
USE_ELONGATION_FILTER, MIN_ELONGATION, DBSCAN_MIN_SAMPLES_DEFAULT = True, 0.40, 5
LR_TOP_PCT, LR_SMOOTH_WIN, LR_SPREAD_MAX_HZ, LR_ENERGY_REL_MIN = 10.0, 3, 3000.0, 0.40
LR_USE_TRUNC_BEFORE_ELBOW, LR_TAIL_COLS = True, 4
ECHO_MIN_NEG_SLOPE_HZ_PER_S, MIN_RIDGE_DT_S, MIN_RIDGE_COLS = -120_000.0, 0.003, 4
DEFAULT_ECHO_BURST_N, DEFAULT_ECHO_WINDOW_S, DEFAULT_MIN_SOCIAL = 10, 4.0, 2
DEFAULT_SNR_CLEAN_THRESHOLD, DEFAULT_SHORTSOCIAL_RATE = 11.0, 0.33


# --- Spectrogram Generation ---
@dataclass
class SpectroCfg:
    n_fft: int = 1024;
    hop: int = 256;
    min_hz: float = 0.0;
    max_hz: float = float("inf")
    dyn_range_db: float = 60.0;
    ref_power: float = 1.0;
    max_width_px: int = 4000
    cmap: str = "magma";
    dpi: int = 150;
    style: str = "fast";
    invert_freq: bool = True
    px_per_khz: float = 6.0;
    axis_fontsize: int = 10;
    title_fontsize: int = 12;
    colorbar: bool = False


def hann_win(n: int) -> np.ndarray: return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n, dtype=np.float32) / n)


def frame_with_overlap(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0: return np.empty((0, n_fft), dtype=np.float32)
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n_frames, n_fft), strides=(hop * stride, stride))


def stft_mag_db_chunked(path: Path, cfg: SpectroCfg):
    info = sf.info(str(path));
    sr, n_frames_total = int(info.samplerate), int(info.frames)
    if sr <= 0 or n_frames_total <= 0: return np.zeros((1, 1), np.float32), np.array([0.0], np.float32), dict(sr=sr,
                                                                                                              seconds_per_col=0.0,
                                                                                                              duration_s=0.0)
    freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / sr).astype(np.float32)
    fmask = (freqs >= float(cfg.min_hz)) & (freqs <= float(cfg.max_hz))
    if not np.any(fmask): fmask[:] = True
    approx_total_frames = max(1, int(math.ceil(n_frames_total / cfg.hop)));
    decim = max(1, approx_total_frames // cfg.max_width_px)
    seconds_per_col = (decim * cfg.hop) / float(sr);
    win = hann_win(cfg.n_fft).astype(np.float32);
    eps = 1e-12
    pooled_cols, pool_vec, pool_count = [], None, 0
    for block in sf.blocks(str(path), blocksize=cfg.hop * 100, overlap=cfg.n_fft - cfg.hop, dtype="float32",
                           always_2d=True):
        x = block[:, 0] if block.shape[1] > 0 else block.reshape(-1)
        if x.size < cfg.n_fft: continue
        frames = frame_with_overlap(x, cfg.n_fft, cfg.hop)
        if frames.size == 0: continue
        X = np.fft.rfft(frames * win, axis=1)
        P = (X.real * X.real + X.imag * X.imag)[:, fmask].astype(np.float32)
        S_db = 10.0 * np.log10(np.maximum(P / max(cfg.ref_power, eps), eps))
        for i in range(S_db.shape[0]):
            if pool_vec is None:
                pool_vec, pool_count = S_db[i].copy(), 1
            else:
                np.maximum(pool_vec, S_db[i], out=pool_vec); pool_count += 1
            if pool_count >= decim: pooled_cols.append(pool_vec.copy()); pool_vec = None
    if pool_vec is not None: pooled_cols.append(pool_vec)
    S = np.stack(pooled_cols, axis=1) if pooled_cols else np.zeros((np.count_nonzero(fmask), 1), dtype=np.float32)
    Smax = float(np.max(S)) if S.size else 0.0;
    S_clamped = np.clip(S, Smax - cfg.dyn_range_db, Smax)
    S01 = (S_clamped - (Smax - cfg.dyn_range_db)) / max(cfg.dyn_range_db, 1e-6)
    S01 = np.clip(S01, 0.0, 1.0).astype(np.float32)
    meta = dict(sr=sr, seconds_per_col=seconds_per_col, duration_s=float(S01.shape[1] * seconds_per_col),
                fmin=float(freqs[fmask].min()) if np.any(fmask) else 0.0,
                fmax=float(freqs[fmask].max()) if np.any(fmask) else 0.0)
    return S01, freqs[fmask], meta


def save_png_fast(S01, freqs, out_png: Path, cfg: SpectroCfg, title: str | None = None):
    arr = np.flipud(S01) if cfg.invert_freq else S01;
    rgba = (colormaps[cfg.cmap](arr) * 255).astype("uint8")
    f_span_khz = max(1.0, (float(freqs.max()) - float(freqs.min())) / 1000.0) if freqs.size >= 2 else 1.0
    tgt_h = max(32, int(round(f_span_khz * cfg.px_per_khz)));
    tgt_w = rgba.shape[1]

    # FIX: Removed deprecated 'mode' parameter. Pillow infers it correctly.
    img = Image.fromarray(rgba).resize((tgt_w, tgt_h), Image.Resampling.BICUBIC)

    out_png.parent.mkdir(parents=True, exist_ok=True);
    img.save(out_png)


def save_png_mpl(S01, freqs, meta: dict, out_png: Path, cfg: SpectroCfg, title: str):
    H, W = S01.shape;
    extent = (0.0, meta.get("duration_s", W), float(freqs.min()), float(freqs.max()))
    fig, ax = plt.subplots(figsize=(10, 7), dpi=cfg.dpi)
    ax.imshow(S01, aspect="auto", origin="lower", extent=extent, cmap=cfg.cmap, vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.set_xlabel("Time (s)", fontsize=cfg.axis_fontsize);
    ax.set_ylabel("Frequency (Hz)", fontsize=cfg.axis_fontsize)
    ax.set_title(title, fontsize=cfg.title_fontsize)
    if cfg.colorbar: plt.colorbar(ax.images[0], ax=ax).set_label("Normalized Intensity", fontsize=cfg.axis_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=cfg.axis_fontsize)
    fig.tight_layout();
    out_png.parent.mkdir(parents=True, exist_ok=True);
    fig.savefig(out_png, dpi=cfg.dpi);
    plt.close(fig)


def save_file_png_wrapper(filepath: str, verdict: str, out_root: Path, cfg: SpectroCfg):
    sub = "CleanBat" if verdict == "Clean Bat" else ("NoisyBat" if verdict == "Noisy Bat" else "NoBat")
    out_png = out_root / sub / (Path(filepath).stem + ".png")
    S01, freqs, meta = stft_mag_db_chunked(Path(filepath), cfg)
    title = f"{Path(filepath).name} — Verdict: {verdict}"
    if cfg.style == "fast":
        save_png_fast(S01, freqs, out_png, cfg, title)
    else:
        save_png_mpl(S01, freqs, meta, out_png, cfg, title)


# --- Core Analysis Helpers ---
def calculate_spectrogram_db(audio_chunk, samplerate):
    f, t, Sxx = spectrogram(audio_chunk, samplerate, nperseg=NPERSEG, noverlap=NPERSEG // 2, detrend=False,
                            scaling='density', mode='psd')
    return f, t, 10 * np.log10(Sxx + 1e-10)


def threshold_and_clean(db_Sxx, global_threshold, *, frequencies, mask_on, mask_mid, mask_width, mask_strength_db):
    if mask_on:
        m = 1.0 / (1.0 + np.exp(-(frequencies.reshape(-1, 1) - mask_mid) / max(1.0, mask_width)))
        local_thresh = global_threshold + (1.0 - m) * mask_strength_db
    else:
        local_thresh = np.full((db_Sxx.shape[0], 1), global_threshold, dtype=np.float32)
    binary_image_noisy = db_Sxx > local_thresh
    return cv2.morphologyEx(binary_image_noisy.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                            iterations=1)


def _component_elongation(patch_binary):
    ys, xs = np.nonzero(patch_binary)
    if len(xs) < 3: return 0.0, 0.0, 0.0
    pts = np.vstack([xs, ys]).astype(np.float32).T
    cov = np.cov((pts - np.mean(pts, axis=0)).T)
    vals, _ = np.linalg.eig(cov)
    vals = np.sort(np.real(vals))[::-1]
    return float(1.0 - (vals[1] / vals[0])), float(vals[0]), float(vals[1])


def _ridge_from_patch(patch_db, patch_binary, freq_axis, time_axis):
    _, T = patch_db.shape;
    ridge_f = np.full(T, np.nan, dtype=np.float32)
    for j in range(T):
        col, mask_col = patch_db[:, j], patch_binary[:, j] > 0
        vals = col[mask_col] if np.any(mask_col) else col
        if vals.size == 0: continue
        top_idx = np.where(col >= np.percentile(vals, 100.0 - LR_TOP_PCT))[0]
        if top_idx.size == 0: top_idx = np.array([np.argmax(col)])
        w = np.maximum(col[top_idx] - np.min(col[top_idx]) + 1e-3, 1e-3)
        ridge_f[j] = float(np.sum(w * freq_axis[top_idx]) / np.sum(w))
    valid = ~np.isnan(ridge_f)
    if not np.any(valid): return None
    ridge_f, ridge_t = ridge_f[valid], time_axis[valid]
    if ridge_f.size < MIN_RIDGE_COLS: return None
    xp = np.pad(ridge_f, (LR_SMOOTH_WIN // 2, LR_SMOOTH_WIN // 2), mode='edge')
    ridge_f = np.convolve(xp, np.ones(LR_SMOOTH_WIN) / LR_SMOOTH_WIN, mode='valid')
    dt = ridge_t[-1] - ridge_t[0]
    return (ridge_f[-1] - ridge_f[0]) / dt if dt >= MIN_RIDGE_DT_S else None

def calculate_coherence(patch):
    total_pixels = int(np.sum(patch))
    if total_pixels == 0: return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(patch.astype(np.uint8), connectivity=4)
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
    if duration < 0.05: coh_thresh = 0.75
    elif duration < 0.15: coh_thresh = 0.70
    else: coh_thresh = 0.60
    if coherence < coh_thresh or duration < 0.01: return "Noise"
    purity_gate = purity_threshold + (1.0 if duration < 0.04 else 0.0)
    if purity < purity_gate: return "Noise"
    has_downward_sweep = (simple_slope <= ECHO_MIN_NEG_SLOPE_HZ_PER_S) if simple_slope is not None else (coeffs[1] < 0.0)
    if (min_freq > 17_000.0) and (bandwidth < 40_000.0) and (duration < 0.10) and has_downward_sweep: return "Echolocation"
    if (min_freq > 18_000.0) and (duration < 0.40) and ((duration > 0.04) or (bandwidth > 70_000.0 and duration > 0.015)): return "Social"
    if simple_slope is not None and (simple_slope >= -80_000.0) and (min_freq >= 16_000.0) and duration <= 0.040:
        if (bandwidth >= 6_000.0) or (purity >= purity_threshold + 2.0) or (coherence >= 0.85): return "ShortSocial"
    return "Noise"


# --- Main Chunk Processor ---
def process_chunk(args):
    (audio_chunk, samplerate, global_threshold, chunk_start_time, params) = args
    if len(audio_chunk) < NPERSEG: return []
    current_process = psutil.Process(os.getpid())
    if current_process.memory_percent() > params['max_ram_pct']: return []
    frequencies, times, db_Sxx = calculate_spectrogram_db(audio_chunk, samplerate)
    binary_image = threshold_and_clean(db_Sxx, global_threshold, frequencies=frequencies, **params['mask_params'])
    points = np.argwhere(binary_image);
    npts = points.shape[0]
    if (float(npts) / binary_image.size) > params['max_occ'] and npts > params['max_npts']: return []
    if npts < 10: return []  # Changed from DBSCAN_MIN_SAMPLES_DEFAULT to match Tkinter version's logic path
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=3, min_samples=params['dbscan_min_samples']).fit(points)
    results = []
    for k in set(db.labels_):
        if k == -1: continue
        cluster_points = points[db.labels_ == k]
        min_r, max_r, min_c, max_c = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0]), np.min(
            cluster_points[:, 1]), np.max(cluster_points[:, 1])
        patch_binary = binary_image[min_r:max_r + 1, min_c:max_c + 1]
        patch_db = db_Sxx[min_r:max_r + 1, min_c:max_c + 1]
        patch_freqs, patch_times = frequencies[min_r:max_r + 1], times[min_c:max_c + 1]

        # This is the logic that was missing:
        coherence = calculate_coherence(patch_binary)
        tonal_purity = calculate_tonal_purity(patch_db, patch_binary)

        c_times, c_freqs = times[cluster_points[:, 1]], frequencies[cluster_points[:, 0]]
        duration, bandwidth, min_freq = float(np.max(c_times) - np.min(c_times)), float(
            np.max(c_freqs) - np.min(c_freqs)), float(np.min(c_freqs))
        if params['use_elong']:
            elongation, _, _ = _component_elongation(patch_binary)
            e_gate = max(0.0, params['min_elong'] - (0.20 if duration < 0.05 else 0.10))
            if elongation < e_gate: continue
        try:
            coeffs = np.polyfit(c_times, c_freqs, 2)
        except (np.linalg.LinAlgError, ValueError):
            coeffs = (0.0, 0.0, 0.0)
        simple_slope = _ridge_from_patch(patch_db, patch_binary, patch_freqs, patch_times)

        # This now calls the correct, more complex classifier:
        label = classify_cluster(duration, bandwidth, min_freq, coeffs, coherence, tonal_purity,
                                 params['purity_threshold'], simple_slope=simple_slope)

        results.append({
            'label': label, 'start_time': chunk_start_time + np.min(c_times),
            'end_time': chunk_start_time + np.max(c_times),
            'mean_db_active': float(np.mean(patch_db[patch_binary.astype(bool)])) if np.any(patch_binary) else float(
                np.mean(patch_db))
        })
    return results


# --- File-level Analysis Orchestrator ---
def noise_scan_global_threshold_and_noise_ref(filepath, samplerate, win_s, hop_s, keep_k, c):
    medians, hi_band_means = [], []
    win_frames, hop_frames = int(win_s * samplerate), int(hop_s * samplerate)
    if win_frames < NPERSEG: win_frames = max(NPERSEG, win_frames)
    with sf.SoundFile(filepath, 'r') as f:
        for start in range(0, len(f), hop_frames):
            f.seek(start);
            audio = f.read(win_frames)
            if len(audio) < NPERSEG: continue
            freqs, _, db_Sxx = calculate_spectrogram_db(audio, samplerate)
            medians.append(float(np.median(db_Sxx)))
            if np.any(freqs >= 17_000.0): hi_band_means.append(float(np.mean(db_Sxx[freqs >= 17_000.0, :])))
    if not medians: return -100.0, -100.0
    medians = np.array(medians);
    k = int(max(1, min(keep_k, len(medians))))
    quiet_idx = np.argpartition(medians, k - 1)[:k];
    quiet_meds = medians[quiet_idx]
    median_med = float(np.median(quiet_meds));
    mad_med = float(np.median(np.abs(quiet_meds - median_med)))
    threshold = median_med + c * (1.4826 * mad_med)
    noise_ref_db = float(np.mean(np.array(hi_band_means)[quiet_idx])) if hi_band_means and len(hi_band_means) == len(
        medians) else -100.0
    return float(threshold), float(noise_ref_db)


def analyze_single_file(filepath, params):
    with sf.SoundFile(filepath, 'r') as f:
        sr, total_frames = f.samplerate, len(f)
    duration_s = total_frames / sr if sr > 0 else 0.0
    if duration_s < 0.75: return {"filepath": filepath, "duration_s": duration_s, "counts": {}, "verdict": "No Bat",
                                  "snr_ratio": None, "noise_ref_db": None}
    thr, noise_ref_db = noise_scan_global_threshold_and_noise_ref(filepath, sr, **params['noise_scan_params']);
    thr += params['noise_offset_db']
    task_params = params.copy()
    task_params['mask_params'] = {'mask_on': params['mask_on'], 'mask_mid': params['mask_mid'],
                                  'mask_width': params['mask_width'], 'mask_strength_db': params['mask_strength']}
    tasks = []
    with sf.SoundFile(filepath, 'r') as f:
        for i in range(0, total_frames, int((CHUNK_DURATION_S - OVERLAP_S) * sr)):
            tasks.append((f.read(int(CHUNK_DURATION_S * sr)), sr, thr, i / sr, task_params))
    calls = []
    if tasks:
        cores = max(1, min(params.get('cores', 1), os.cpu_count() or 1, len(tasks)))
        if cores == 1:
            for res in map(process_chunk, tasks): calls.extend(res)
        else:
            ctx = get_context("spawn")
            with ctx.Pool(processes=cores, maxtasksperchild=50) as pool:
                for res in pool.imap_unordered(process_chunk, tasks): calls.extend(res)
    counts = {"Echolocation": 0, "Social": 0, "ShortSocial": 0, "Noise": 0}
    for c in calls: counts[c.get('label', 'Noise')] += 1
    verdict = "No Bat"
    echo_times = sorted([c['start_time'] for c in calls if c.get('label') == "Echolocation"])
    if len(echo_times) >= params['echo_burst_n']:
        for i in range(len(echo_times) - params['echo_burst_n'] + 1):
            if echo_times[i + params['echo_burst_n'] - 1] - echo_times[i] < params['echo_window_s']:
                verdict = "Bat";
                break
    if verdict == "No Bat" and counts["Social"] >= params['min_social_calls']: verdict = "Bat"

    # FIX: Initialize snr_ratio to a default value before the 'if' block.
    snr_ratio = None

    if verdict == "Bat":
        call_dbs = [c['mean_db_active'] for c in calls if c['label'] != 'Noise' and 'mean_db_active' in c]
        mean_db = np.mean(call_dbs) if call_dbs else -100.0
        snr_ratio = 10 ** ((mean_db - noise_ref_db) / 10.0) if noise_ref_db > -100 else 0
        verdict = "Clean Bat" if snr_ratio > params['snr_clean_threshold'] else "Noisy Bat"
    return {"filepath": filepath, "duration_s": duration_s, "counts": counts, "verdict": verdict,
            "snr_ratio": snr_ratio, "noise_ref_db": noise_ref_db}


# --- Reporting ---
def write_report_txt(all_results, out_dir: Path, total_wall_s, total_bytes):
    def fmt_hms(seconds: float) -> str:
        seconds = max(0, int(round(seconds)));
        h = seconds // 3600;
        m = (seconds % 3600) // 60;
        s = seconds % 60
        return f"{h:d}:{m:02d}:{s:02d}"

    report_path = out_dir / "batch_report.txt"
    final_bins = {"Clean Bat": 0, "Noisy Bat": 0, "No Bat": 0}
    sum_counts = {"Echolocation": 0, "Social": 0, "ShortSocial": 0, "Noise": 0}
    bytes_by_verdict = {"Clean Bat": 0, "Noisy Bat": 0, "No Bat": 0}
    secs_by_verdict = {"Clean Bat": 0.0, "Noisy Bat": 0.0, "No Bat": 0.0}

    for r in all_results:
        v = r["verdict"]
        final_bins[v] = final_bins.get(v, 0) + 1
        if r.get('counts'):
            for k in sum_counts: sum_counts[k] += r["counts"].get(k, 0)

        try:
            sz = os.path.getsize(r["filepath"]) if os.path.exists(r["filepath"]) else 0
        except Exception:
            sz = 0
        bytes_by_verdict[v] = bytes_by_verdict.get(v, 0) + sz
        secs_by_verdict[v] = secs_by_verdict.get(v, 0) + float(r.get("duration_s", 0.0))

    size_gb = total_bytes / (1024 ** 3) if total_bytes > 0 else 0.0

    def gb(x):
        return x / (1024 ** 3)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Bat / No-Bat Batch Report\n=========================\n\n")
        f.write(f"Files processed: {len(all_results)}\n")
        f.write(f"Total wall time: {total_wall_s:.2f} s\n")
        if size_gb > 0:
            f.write(f"Total input size: {size_gb:.3f} GiB\n")
            f.write(
                f"Total size Clean Bat: {gb(bytes_by_verdict.get('Clean Bat', 0)):.3f} GiB (record time = {fmt_hms(secs_by_verdict.get('Clean Bat', 0.0))})\n")
            f.write(
                f"Total size Noisy Bat: {gb(bytes_by_verdict.get('Noisy Bat', 0)):.3f} GiB (record time = {fmt_hms(secs_by_verdict.get('Noisy Bat', 0.0))})\n")
            f.write(f"Time per GiB: {total_wall_s / size_gb:.2f} s/GiB\n\n")

        f.write("Aggregate call counts:\n")
        for k in ["Echolocation", "Social", "ShortSocial", "Noise"]:
            f.write(f"  {k}: {sum_counts[k]}\n")

        f.write("\nFinal verdict bins:\n")
        for k in ["Clean Bat", "Noisy Bat", "No Bat"]:
            f.write(f"  {k}: {final_bins.get(k, 0)}\n")


# =============================================================================
#
# 2. PYQT6 GUI APPLICATION
#    This section contains the new PyQt6 frontend, which replaces the
#    original Tkinter GUI. It uses a worker thread for non-blocking analysis.
#
# =============================================================================

class AnalysisWorker(QObject):
    """
    A QObject worker that runs the analysis in a separate thread to prevent
    the GUI from freezing. Communicates with the main window via signals.
    """
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(list, float)  # results, duration
    request_confirmation = pyqtSignal(str, list)  # message, files_to_delete

    def __init__(self, wav_files, params, out_root):
        super().__init__()
        self.wav_files = wav_files
        self.params = params
        self.out_root = Path(out_root)
        self.is_running = True

    def run(self):
        start_time = time.time()
        results, to_delete = [], []
        cfg = SpectroCfg(style="fast")

        for i, filepath in enumerate(self.wav_files):
            if not self.is_running: break
            self.progress.emit(i, len(self.wav_files), Path(filepath).name)
            res = analyze_single_file(filepath, self.params)
            results.append(res)
            verdict = res.get("verdict", "No Bat")

            if self.params['save_pngs']:
                try:
                    save_file_png_wrapper(filepath, verdict, self.out_root, cfg)
                except Exception as e:
                    print(f"[ERROR] PNG generation failed for {filepath}: {e}")

            copied_ok = False
            if self.params['save_wavs'] and verdict in ("Clean Bat", "Noisy Bat"):
                try:
                    sub = "CleanBat" if verdict == "Clean Bat" else "NoisyBat"
                    rel_parent = Path(filepath).parent.relative_to(self.params['base_folder']) if self.params[
                        "recurse"] else Path()
                    dst = self.out_root / sub / rel_parent / Path(filepath).name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists(): shutil.copy2(filepath, dst)
                    copied_ok = True
                except Exception as e:
                    print(f"[ERROR] WAV copy failed for {filepath}: {e}")

            if self.params['delete_unsorted']:
                if verdict in ("Clean Bat", "Noisy Bat"):
                    if copied_ok: to_delete.append(filepath)
                else:
                    to_delete.append(filepath)

        if self.is_running and to_delete:
            self.request_confirmation.emit(f"Delete {len(to_delete)} original sorted files?", to_delete)

        self.finished.emit(results, time.time() - start_time)

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bat / No-Bat Classifier")
        self.setGeometry(100, 100, 1100, 750)  # Increased size for better layout
        self.thread = None
        self.worker = None

        # --- Setup main layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.create_left_panel()
        self.create_right_panel()
        self.apply_stylesheet()
        self.set_icon()

    def set_icon(self, path="batcrossingguard.png"):
        if Path(path).exists(): self.setWindowIcon(QIcon(path))

    def create_left_panel(self):
        left_panel = QFrame();
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        self.bat_image_label = QLabel()
        if Path("batcrossingguard.png").exists():
            pixmap = QPixmap("batcrossingguard.png")
            self.bat_image_label.setPixmap(
                pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.bat_image_label.setText("batcrossingguard.png not found.")
        self.bat_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar();
        self.progress_bar.setTextVisible(False);
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Select a folder and click 'Run Analysis'");
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.bat_image_label)
        left_layout.addStretch()
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        self.main_layout.addWidget(left_panel, 1)

    def create_right_panel(self):
        right_panel = QFrame();
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        # --- Top controls: Folder selection ---
        top_controls_box = QWidget()
        top_controls_layout = QHBoxLayout(top_controls_box)
        top_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.folder_path_edit = QLineEdit();
        self.folder_path_edit.setPlaceholderText("Select folder containing .wav files...")
        browse_button = QPushButton("Browse...");
        browse_button.clicked.connect(self.browse_folder)
        top_controls_layout.addWidget(self.folder_path_edit);
        top_controls_layout.addWidget(browse_button)
        # --- Run Button ---
        self.run_button = QPushButton("Run Analysis");
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self.start_analysis)
        right_layout.addWidget(top_controls_box)
        right_layout.addWidget(self.run_button)
        # --- Parameter Groups ---
        self.create_parameter_groups(right_layout)
        self.main_layout.addWidget(right_panel, 1)

    def create_parameter_groups(self, parent_layout):
        # --- Helper to create a labeled entry field ---
        def add_row(layout, label, val, width=60):
            entry = QLineEdit(str(val));
            entry.setFixedWidth(width)
            layout.addRow(label, entry)
            return entry

        # --- Performance Group ---
        perf_group = QGroupBox("Performance & Stability")
        perf_layout = QGridLayout(perf_group)
        default_cores = max(1, math.floor((os.cpu_count() or 2) * 0.75))
        self.cores_entry = QLineEdit(str(default_cores))
        self.max_ram_entry = QLineEdit("85")
        self.max_occ_entry = QLineEdit("65.0")
        self.max_npts_entry = QLineEdit("150000")
        perf_layout.addWidget(QLabel("Worker Cores:"), 0, 0);
        perf_layout.addWidget(self.cores_entry, 0, 1)
        perf_layout.addWidget(QLabel("Max Worker RAM (%):"), 1, 0);
        perf_layout.addWidget(self.max_ram_entry, 1, 1)
        perf_layout.addWidget(QLabel("Skip Chunk if Occupancy > (%):"), 2, 0);
        perf_layout.addWidget(self.max_occ_entry, 2, 1)
        perf_layout.addWidget(QLabel("AND Points >:"), 3, 0);
        perf_layout.addWidget(self.max_npts_entry, 3, 1)
        # --- Verdict Rules Group ---
        verdict_group = QGroupBox("Verdict Rules")
        verdict_layout = QGridLayout(verdict_group)
        self.echo_burst_entry = QLineEdit(str(DEFAULT_ECHO_BURST_N))
        self.echo_window_entry = QLineEdit(str(DEFAULT_ECHO_WINDOW_S))
        self.min_social_entry = QLineEdit(str(DEFAULT_MIN_SOCIAL))
        self.snr_clean_entry = QLineEdit(str(DEFAULT_SNR_CLEAN_THRESHOLD))
        verdict_layout.addWidget(QLabel("Echo Burst N:"), 0, 0);
        verdict_layout.addWidget(self.echo_burst_entry, 0, 1)
        verdict_layout.addWidget(QLabel("Echo Window (s):"), 1, 0);
        verdict_layout.addWidget(self.echo_window_entry, 1, 1)
        verdict_layout.addWidget(QLabel("Min Social Calls:"), 2, 0);
        verdict_layout.addWidget(self.min_social_entry, 2, 1)
        verdict_layout.addWidget(QLabel("Clean SNR Threshold (×):"), 3, 0);
        verdict_layout.addWidget(self.snr_clean_entry, 3, 1)
        # --- File Handling Group ---
        files_group = QGroupBox("File Handling")
        files_layout = QVBoxLayout(files_group)
        self.save_wavs_check = QCheckBox("Save sorted .wav files");
        self.save_wavs_check.setChecked(True)
        self.save_pngs_check = QCheckBox("Save spectrograms (PNGs)");
        self.save_pngs_check.setChecked(False)
        self.delete_unsorted_check = QCheckBox("Delete original sorted files");
        self.delete_unsorted_check.setChecked(False)
        self.recurse_check = QCheckBox("Recurse into sub-folders");
        self.recurse_check.setChecked(True)
        for cb in [self.save_wavs_check, self.save_pngs_check, self.delete_unsorted_check,
                   self.recurse_check]: files_layout.addWidget(cb)
        # --- Two-column layout for parameter groups ---
        param_layout = QGridLayout()
        param_layout.addWidget(perf_group, 0, 0)
        param_layout.addWidget(verdict_group, 0, 1)
        param_layout.addWidget(files_group, 1, 0, 1, 2)  # Span across two columns
        parent_layout.addLayout(param_layout)
        parent_layout.addStretch()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder");
        if folder: self.folder_path_edit.setText(folder)

    def collect_params(self):
        try:
            return {
                'base_folder': self.folder_path_edit.text(),
                'cores': int(self.cores_entry.text()), 'purity_threshold': 4.5,
                'max_occ': float(self.max_occ_entry.text()) / 100.0, 'max_npts': int(self.max_npts_entry.text()),
                'max_ram_pct': float(self.max_ram_entry.text()),
                'noise_scan_params': {'win_s': NOISE_WIN_S, 'hop_s': NOISE_HOP_S, 'keep_k': NOISE_KEEP_K,
                                      'c': THRESH_C},
                'noise_offset_db': THRESH_OFFSET_DB,
                'mask_on': MASK_ENABLE_DEFAULT, 'mask_mid': MASK_MID_HZ, 'mask_width': MASK_WIDTH_HZ,
                'mask_strength': MASK_STRENGTH_DB,
                'use_elong': USE_ELONGATION_FILTER, 'min_elong': MIN_ELONGATION,
                'dbscan_min_samples': DBSCAN_MIN_SAMPLES_DEFAULT,
                'echo_burst_n': int(self.echo_burst_entry.text()),
                'echo_window_s': float(self.echo_window_entry.text()),
                'min_social_calls': int(self.min_social_entry.text()),
                'snr_clean_threshold': float(self.snr_clean_entry.text()),
                'save_wavs': self.save_wavs_check.isChecked(), 'save_pngs': self.save_pngs_check.isChecked(),
                'delete_unsorted': self.delete_unsorted_check.isChecked(), 'recurse': self.recurse_check.isChecked(),
            }
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input",
                                f"Please check your parameters. All numeric fields must be valid numbers.\nError: {e}")
            return None

    def start_analysis(self):
        folder = self.folder_path_edit.text()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Input Error", "Please select a valid folder.");
            return
        params = self.collect_params()
        if not params: return

        path_obj = Path(folder)
        glob_pattern = "**/*" if params['recurse'] else "*"
        all_files = path_obj.glob(glob_pattern)
        wav_files = [f for f in all_files if f.suffix.lower() == ".wav"]
        self.total_bytes = sum(f.stat().st_size for f in wav_files)
        if not wav_files:
            QMessageBox.information(self, "No Files", "No .wav files found in the selected directory.");
            return

        self.run_button.setEnabled(False);
        self.run_button.setText("Analysis in Progress...")
        self.progress_bar.setRange(0, len(wav_files))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        out_root = path_obj / "_bat_no_bat_outputs_pyqt"
        for sub in ("CleanBat", "NoisyBat", "NoBat"): (out_root / sub).mkdir(parents=True, exist_ok=True)

        self.thread = QThread();
        self.worker = AnalysisWorker(wav_files, params, out_root)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.request_confirmation.connect(self.confirm_deletion)
        self.thread.start()

    def update_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Processing {current}/{total}: {filename}")

    def on_analysis_finished(self, results, duration):
        out_root = Path(self.folder_path_edit.text()) / "_bat_no_bat_outputs_pyqt"

        # This is the line you were looking for, using self.total_bytes
        report_path = write_report_txt(results, out_root, duration, self.total_bytes)

        self.status_label.setText(f"Analysis complete! Report saved to: {report_path}")
        self.run_button.setEnabled(True);
        self.run_button.setText("Run Analysis")
        self.progress_bar.setRange(0, 100);
        self.progress_bar.setValue(100)  # Show 100%
        params = self.collect_params()
        if not (params and params['delete_unsorted'] and any(res['verdict'] != 'No Bat' for res in results)):
            QMessageBox.information(self, "Success", f"Batch analysis is complete.\nReport saved to:\n{report_path}")
        self.thread.quit();
        self.thread.wait();
        self.thread.deleteLater();
        self.worker.deleteLater()

    def confirm_deletion(self, message, files_to_delete):
        reply = QMessageBox.question(self, 'Confirm Deletion', message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            deleted, failed = 0, 0
            for f in files_to_delete:
                try:
                    os.remove(f); deleted += 1
                except OSError as e:
                    failed += 1; print(f"[ERROR] Failed to delete {f}: {e}")
            self.status_label.setText(f"Deletion complete. Deleted: {deleted}, Failed: {failed}.")
        # Show final message here after deletion logic is handled.
        report_path = Path(self.folder_path_edit.text()) / "_bat_no_bat_outputs_pyqt" / "batch_report.txt"
        QMessageBox.information(self, "Success", f"Batch analysis is complete.\nReport saved to:\n{report_path}")

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2E3440; color: #ECEFF4;
                font-family: "Segoe UI", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
            }
            QMainWindow { border: 1px solid #4C566A; }
            QGroupBox {
                font-size: 14px; font-weight: bold; color: #88C0D0;
                border: 1px solid #4C566A; border-radius: 5px; margin-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }
            QLabel { font-size: 12px; }
            #statusLabel { font-size: 14px; color: #A3BE8C; font-weight: bold; }
            QLineEdit {
                background-color: #434C5E; border: 1px solid #4C566A;
                border-radius: 4px; padding: 6px; font-size: 12px;
            }
            QLineEdit:focus { border: 1px solid #88C0D0; }
            QPushButton {
                background-color: #5E81AC; color: #ECEFF4; border: none;
                padding: 8px 16px; border-radius: 4px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #81A1C1; }
            #runButton { background-color: #A3BE8C; color: #2E3440; }
            #runButton:hover { background-color: #B48EAD; }
            #runButton:disabled { background-color: #4C566A; color: #D8DEE9; }
            QProgressBar {
                border: 1px solid #4C566A; border-radius: 5px; text-align: center;
                color: #2E3440; font-weight: bold;
            }
            QProgressBar::chunk { background-color: #A3BE8C; border-radius: 4px; }
            QCheckBox { font-size: 12px; padding: 5px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #4C566A; }
            QCheckBox::indicator:checked { background-color: #A3BE8C; }
        """)


# =============================================================================
# 4. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Necessary for Windows/macOS executables
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
