#!/usr/bin/env python3
"""
batnobat_onepass.py
-------------------
One-step pipeline for large-scale bat bioacoustics triage + optional finalization.

PHASE 1 (always): Classify WAVs into Bat / Bat_lowconf / NoBat
  - Primary thresholds (UNCHANGED): MIN_CALLS=15, SNR_DB_MIN=3.0 dB, TONAL_RATIO_MAX=0.65
  - Low-confidence pass: applied ONLY to NoBat, with liberal thresholds (CLI --lc-*)
  - Streaming, RAM-friendly, flattened output; CSV + logging

PHASE 2 (optional, with --finalize): Cleanup + Archive (zip via archive_zip.py)
  - Deletes all PNGs
  - Deletes NoBat/
  - Zips the OUT directory (which now mostly contains Bat/ + Bat_lowconf/)
  - Deletes all remaining files except the .zip (and split parts)
  - By default, keeps summary.csv and run.log unless you pass --finalize-purge-metadata

Usage examples:
  # Classify only
  python batnobat_onepass.py --in Bishop_Aug21_Aug22 --out OutBNB --workers 20 --recursive --png-all

  # Classify + finalize (live)
  python batnobat_onepass.py --in Bishop_Aug21_Aug22 --out OutBNB --workers 20 --recursive --png-all --finalize --force

  # Classify + finalize with split zip + custom name
  python batnobat_onepass.py --in Bishop_Aug21_Aug22 --out OutBNB --workers 20 --recursive \
      --finalize --force --split 2000m --zip-out /archives/Bishop_Aug21_Aug22.zip

Notes:
  - Requires: numpy, soundfile, matplotlib
  - Requires system 'zip' tool for archive_zip.py to run (sudo apt-get install -y zip)
"""

import argparse, os, sys, csv, math, shutil, traceback, subprocess, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from numpy.fft import rfft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Primary thresholds (UNCHANGED) ----
DEF_SR = 384000
BAT_BAND = (18000, 52000)   # Hz
NOISE_BAND = (8000, 12000)  # Hz
NFFT = 2048
HOP = 1024
WINDOW = "hann"
MIN_CALLS = 15
SNR_DB_MIN = 3.0
PEAK_K = 2.5
TONAL_RATIO_MAX = 0.65
TIME_SMOOTH_SEC = 0.010

# Viz downsampling
SPEC_MAX_KHZ = 80.0
SPEC_TARGET_COLS = 1000

# dB histogram bounds (for pixel SNR mode)
DB_MIN = -160.0
DB_MAX = 60.0

# -------- helpers --------
def get_window_vec(name: str, n: int) -> np.ndarray:
    if name.lower() == "hann":
        return np.hanning(n).astype(np.float32)
    raise ValueError("Only 'hann' supported.")

def medfilt_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k < 3 or (k % 2 == 0): return x
    r = k // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        a = max(0, i - r); b = min(len(x), i + r + 1)
        out[i] = np.median(x[a:b])
    return out

def mad(x: np.ndarray) -> float:
    m = np.median(x)
    return float(np.median(np.abs(x - m)))

def classify_with_thresholds(call_count, snr_db, tonal_r, min_calls, snr_min, tonal_max):
    return (call_count >= min_calls) and (snr_db >= snr_min) and (tonal_r <= tonal_max)

def score_for_csv(call_count, snr_db, tonal_r):
    s  = 1.0 * (math.log1p(call_count))
    s += 0.8 * max(0.0, snr_db)
    s -= 1.0 * max(0.0, (tonal_r - TONAL_RATIO_MAX) * 10.0)
    return s

class DBMedian:
    def __init__(self, db_min=DB_MIN, db_max=DB_MAX, bin_width=0.1):
        self.db_min = float(db_min); self.db_max = float(db_max); self.bin_w = float(bin_width)
        n_bins = int(math.ceil((self.db_max - self.db_min) / self.bin_w))
        self.counts = np.zeros(n_bins, dtype=np.int64); self.total = 0
    def add_values(self, db_array: np.ndarray):
        clipped = np.clip(db_array, self.db_min, self.db_max - 1e-9)
        bins = ((clipped - self.db_min) / self.bin_w).astype(np.int64)
        self.counts += np.bincount(bins, minlength=self.counts.size)
        self.total += db_array.size
    def median(self) -> float:
        if self.total == 0: return float("-inf")
        half = (self.total - 1) // 2
        csum = np.cumsum(self.counts)
        idx = int(np.searchsorted(csum, half + 1))
        return self.db_min + (idx + 0.5) * self.bin_w

def _log(logfh, msg):
    if logfh:
        logfh.write(msg + "\n"); logfh.flush()
        try: os.fsync(logfh.fileno())
        except Exception: pass
    else:
        print(msg)

# ---- analysis (streaming) ----
def analyze_streaming(wav_path: Path,
                      sr_expected: int,
                      block_sec: float,
                      db_bin_width: float,
                      snr_mode: str,
                      want_png: bool,
                      debug: bool,
                      log=None) -> dict:
    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate; frames_total = info.frames
    except Exception as e:
        return {"error": f"[ERROR] probing {wav_path}: {e}"}
    if sr != sr_expected:
        _log(log, f"[WARN] {wav_path.name}: samplerate {sr} != expected {sr_expected}. Proceeding.")

    overlap = NFFT - HOP; win = get_window_vec(WINDOW, NFFT)
    freqs = np.fft.rfftfreq(NFFT, d=1.0/sr).astype(np.float32)

    def band_slice(band):
        i0 = int(np.searchsorted(freqs, band[0], side='left'))
        i1 = int(np.searchsorted(freqs, band[1], side='left'))
        return slice(i0, i1)
    bat_sl = band_slice(BAT_BAND); noi_sl = band_slice(NOISE_BAND)

    # Accumulators
    K_list = []       # bat-band linear power per frame
    ratio_list = []   # per-frame maxbin/energy
    Pn_lists = []     # noise-band sums per frame (for band SNR)

    # For pixel SNR mode
    bat_med_est = DBMedian(bin_width=db_bin_width)
    noi_med_est = DBMedian(bin_width=db_bin_width)

    # Optional tiny spectrogram
    want_png = bool(want_png)
    spec_cols = []; col_accum = None; col_count = 0
    approx_total_frames = max(1, (frames_total + overlap) // HOP)
    frames_per_col = max(1, approx_total_frames // SPEC_TARGET_COLS) if want_png else 0
    fmax_bins = int(np.searchsorted(freqs, SPEC_MAX_KHZ*1000.0, side='left')) if want_png else 0

    block_frames = int(block_sec * sr); tail = np.zeros(overlap, dtype=np.float32); read_pos = 0

    with sf.SoundFile(str(wav_path), mode="r") as f:
        while read_pos < frames_total:
            to_read = min(block_frames, frames_total - read_pos)
            block = f.read(to_read, dtype="float32", always_2d=True); read_pos += to_read
            x = block[:, 0] if block.shape[1] > 1 else block[:, 0]
            if tail.size: x = np.concatenate([tail, x], axis=0)
            if len(x) < NFFT:
                tail = x[-overlap:].copy() if len(x) >= overlap else x.copy(); continue

            n_frames = 1 + (len(x) - NFFT) // HOP
            stride = x.strides[0]
            frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, NFFT),
                                                     strides=(HOP*stride, stride), writeable=False)
            X = rfft(frames * win, n=NFFT, axis=1)
            P = (np.abs(X)**2).astype(np.float32)

            Pb = P[:, bat_sl].sum(axis=1) + 1e-20
            K_list.append(Pb)
            maxbin = P[:, bat_sl].max(axis=1)
            ratio_list.append(np.clip(maxbin / Pb, 0.0, 1.0))

            Pn_sum = P[:, noi_sl].sum(axis=1) + 1e-20
            Pn_lists.append(Pn_sum)

            # Pixel SNR streams
            Sdb_bat = 10.0 * np.log10(P[:, bat_sl].astype(np.float64) + 1e-12)
            Sdb_noi = 10.0 * np.log10(P[:, noi_sl].astype(np.float64) + 1e-12)
            bat_med_est.add_values(Sdb_bat.ravel())
            noi_med_est.add_values(Sdb_noi.ravel())

            if want_png:
                bandP = P[:, :fmax_bins]
                for i in range(bandP.shape[0]):
                    if col_accum is None:
                        col_accum = bandP[i].copy(); col_count = 1
                    else:
                        col_accum += bandP[i]; col_count += 1
                    if frames_per_col and (col_count >= frames_per_col):
                        spec_cols.append((col_accum / float(col_count))); col_accum = None; col_count = 0

            tail = x[-overlap:].copy()

    if not K_list:
        duration_s = frames_total / float(sr) if sr else 0.0
        return {"file": str(wav_path), "sr": sr, "duration_s": duration_s,
                "calls": 0, "snr_db": 0.0, "tonal_ratio": 0.0, "decision": "NoBat",
                "score": 0.0, "lowconf": 0, "spec_cols": None, "freqs": freqs}

    K_all = np.concatenate(K_list).astype(np.float32)
    ratio_all = np.concatenate(ratio_list).astype(np.float32)

    # Kinetics peaks
    smooth_n = max(1, int(round(TIME_SMOOTH_SEC / (HOP / sr))))
    if smooth_n % 2 == 0: smooth_n += 1
    K_s = medfilt_1d(K_all, smooth_n)
    thr = float(np.median(K_s) + PEAK_K * mad(K_s))
    peaks = np.where((K_s[1:-1] > K_s[:-2]) & (K_s[1:-1] > K_s[2:]) & (K_s[1:-1] > thr))[0] + 1
    call_count = int(peaks.size)

    # SNR
    if snr_mode == "pixel":
        bat_med_db = bat_med_est.median()
        noi_med_db = noi_med_est.median()
        snr_db = float(bat_med_db - noi_med_db)
    else:  # "band" default
        Pb_all = np.concatenate(K_list).astype(np.float64)
        Pn_all = np.concatenate(Pn_lists).astype(np.float64)
        bat_med_db = 10.0 * np.log10(np.median(Pb_all) + 1e-20)
        noi_med_db = 10.0 * np.log10(np.median(Pn_all) + 1e-20)
        snr_db = float(bat_med_db - noi_med_db)

    tonal_r = float(np.mean(ratio_all))
    decision_primary = "Bat" if classify_with_thresholds(call_count, snr_db, tonal_r,
                                                         MIN_CALLS, SNR_DB_MIN, TONAL_RATIO_MAX) else "NoBat"
    score = score_for_csv(call_count, snr_db, tonal_r)

    # Prepare small spectrogram for optional PNG
    S_small = None
    if spec_cols:
        if col_accum is not None and col_count > 0:
            spec_cols.append((col_accum / float(col_count)))
        S_small = np.vstack(spec_cols).T

    duration_s = frames_total / float(sr) if sr else 0.0
    return {"file": str(wav_path), "sr": sr, "duration_s": duration_s,
            "calls": call_count, "snr_db": round(float(snr_db), 2),
            "tonal_ratio": round(float(tonal_r), 3), "decision": decision_primary,
            "score": round(float(score), 2), "lowconf": 0,
            "spec_cols": S_small, "freqs": freqs}

def save_spectrogram_png_from_small(out_png: Path, S_small: np.ndarray, freqs: np.ndarray):
    if S_small is None or S_small.size == 0: return
    kmax = int(np.searchsorted(freqs, SPEC_MAX_KHZ*1000.0, side='left'))
    S_small = S_small[:kmax, :]
    S_db = 10.0 * np.log10(S_small + 1e-20)
    vmax = np.percentile(S_db, 98); vmin = vmax - 60.0
    t_axis = np.arange(S_db.shape[1]); f_axis_khz = freqs[:S_db.shape[0]] / 1000.0
    plt.figure(figsize=(10, 3), dpi=150)
    plt.pcolormesh(t_axis, f_axis_khz, S_db, shading='nearest', vmin=vmin, vmax=vmax)
    b0 = BAT_BAND[0]/1000.0; b1 = BAT_BAND[1]/1000.0
    plt.axhspan(b0, b1, alpha=0.12, color='white')
    plt.xlabel("Coarse time bin"); plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram (downsampled, dB)"); plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()

# ---- worker: classify + lowconf rescue + copy + png ----
def process_file(args_tuple):
    (wav_path_s, out_root_s, sr_expected, block_sec, no_png, png_all,
     db_bin_width, snr_mode, lc_min_calls, lc_snr_min, lc_tonal_max,
     debug, log_path_s) = args_tuple
    wav_path = Path(wav_path_s); out_root = Path(out_root_s); logfh = None
    try:
        logfh = open(log_path_s, "a", buffering=1, encoding="utf-8")
        res = analyze_streaming(wav_path, sr_expected, block_sec, db_bin_width,
                                snr_mode=snr_mode, want_png=(not no_png),
                                debug=debug, log=logfh)
        if "error" in res:
            _log(logfh, res["error"]);
            return {"error": res["error"]}

        decision = res["decision"]
        lowconf_flag = 0

        # Low-confidence pass on NoBat only
        if decision == "NoBat":
            if classify_with_thresholds(res["calls"], res["snr_db"], res["tonal_ratio"],
                                        lc_min_calls, lc_snr_min, lc_tonal_max):
                decision = "Bat_lowconf"
                lowconf_flag = 1

        # Copy destination (flattened)
        out_dir = out_root / decision
        wav_out = out_dir / wav_path.name
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Copy + verify + fsync dir
        if not wav_path.exists():
            msg = f"[ERROR] source missing before copy: {wav_path}"
            _log(logfh, msg); return {"error": msg}
        src_sz = wav_path.stat().st_size
        shutil.copy2(str(wav_path), str(wav_out))
        if not wav_out.exists():
            msg = f"[ERROR] copy reported success but dest missing: {wav_out}"
            _log(logfh, msg); return {"error": msg}
        dst_sz = wav_out.stat().st_size
        if src_sz != dst_sz:
            msg = f"[ERROR] size mismatch after copy: {wav_path} ({src_sz}) -> {wav_out} ({dst_sz})"
            _log(logfh, msg); return {"error": msg}
        try:
            dir_fd = os.open(str(out_dir), os.O_RDONLY)
            try: os.fsync(dir_fd)
            finally: os.close(dir_fd)
        except Exception:
            pass

        # PNG policy (default: Bat only; --png-all for all; --no-png to skip)
        png_field = ""
        if (not no_png) and (png_all or decision == "Bat"):
            png_path = out_root / decision / f"{wav_path.stem}.png"
            save_spectrogram_png_from_small(png_path, res["spec_cols"], res["freqs"])
            png_field = str(png_path)

        res.pop("spec_cols", None); res.pop("freqs", None)
        res["decision"] = decision
        res["lowconf"] = lowconf_flag

        return res | {"png": png_field, "dst_wav": str(wav_out)}
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"[ERROR] worker crashed on {wav_path}: {e}\n{tb}"
        if logfh: _log(logfh, msg)
        return {"error": msg}
    finally:
        try:
            if logfh: logfh.close()
        except Exception:
            pass

# ---- finalization helpers ----
def bytes_h(n: int) -> str:
    units = ["B","KB","MB","GB","TB","PB","EB"]
    x = float(n)
    for u in units:
        if x < 1024.0: return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} ZB"

def finalize_cleanup_and_zip(out_dir: Path, archive_script: Path, zip_level: int,
                             split: str|None, zip_out: Path|None, manifest: Path|None,
                             force: bool, keep_metadata: bool):
    out = out_dir.resolve()
    has_bat = (out / "Bat").is_dir()
    has_low = (out / "Bat_lowconf").is_dir()
    has_nbt = (out / "NoBat").is_dir()
    if not (has_bat or has_low or has_nbt):
        raise RuntimeError(f"{out} does not contain Bat/, Bat_lowconf/, or NoBat/")

    # 1) delete PNGs
    n_png = 0; bytes_png = 0
    for pat in ("*.png","*.PNG"):
        for p in out.rglob(pat):
            try: bytes_png += p.stat().st_size
            except Exception: pass
            if force: p.unlink(missing_ok=True)
            n_png += 1

    # 2) delete NoBat/
    if has_nbt and force:
        shutil.rmtree(out / "NoBat", ignore_errors=True)

    # 3) run archive_zip.py on OUT directory
    if zip_out is None:
        zip_out = out.with_suffix(".zip")
    cmd = [sys.executable, str(archive_script), str(out), "--level", str(zip_level), "--out", str(zip_out)]
    if split:    cmd += ["--split", split]
    if manifest: cmd += ["--manifest", str(manifest)]

    t0 = time.time()
    rc = subprocess.call(cmd)
    t1 = time.time()
    if rc != 0:
        raise RuntimeError(f"archive_zip.py exited with {rc}")

    base_zip = zip_out.with_suffix(".zip")
    keep = {base_zip}
    for part in base_zip.parent.glob(base_zip.stem + ".z[0-9][0-9]"):
        keep.add(part)

    # 4) delete everything except keep set (+ optionally metadata)
    removed = 0; removed_bytes = 0
    for p in out.rglob("*"):
        if not p.is_file(): continue
        if p in keep:       continue
        if keep_metadata and p.name in ("summary.csv", "run.log"): continue
        try:
            if force:
                removed_bytes += p.stat().st_size
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass

    # clean now-empty dirs (preserve root)
    for root, dirs, _ in os.walk(out, topdown=False):
        for d in dirs:
            dp = Path(root) / d
            try:
                # don't delete the directory containing the zips
                if any(str(k).startswith(str(dp)) for k in keep):
                    continue
                if force:
                    dp.rmdir()
            except Exception:
                pass

    return {
        "png_removed": n_png, "png_bytes": bytes_png,
        "zip": base_zip, "elapsed_zip_s": t1 - t0,
        "files_removed": removed, "bytes_removed": removed_bytes,
        "kept": sorted(str(k) for k in keep),
    }

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="One-step bat/no-bat triage with optional low-confidence rescue and finalization.")
    # Phase 1 (classify)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--sr", dest="sr", type=int, default=DEF_SR)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--block-sec", type=float, default=20.0)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--png-all", action="store_true")
    ap.add_argument("--db-bin-width", type=float, default=0.1, help="dB histogram bin width (pixel SNR mode)")
    ap.add_argument("--snr-mode", choices=["pixel","band"], default="band",
                    help="pixel: median of dB pixels; band: median over time of band sums → dB (default).")
    ap.add_argument("--lc-min-calls", type=int, default=10, help="Low-confidence minimum call count (default 10)")
    ap.add_argument("--lc-snr-min", type=float, default=2.5, help="Low-confidence minimum SNR dB (default 2.5)")
    ap.add_argument("--lc-tonal-max", type=float, default=0.75, help="Low-confidence maximum tonal ratio (default 0.75)")
    ap.add_argument("--csv-fsync-every", type=int, default=10)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log-file", type=Path, default=None)

    # Phase 2 (finalize)
    ap.add_argument("--finalize", action="store_true", help="After classification, cleanup + archive the OUT directory.")
    ap.add_argument("--force", action="store_true", help="Perform finalization actions (required for deletion); otherwise just classify.")
    ap.add_argument("--archive-script", type=Path, default=None, help="Path to archive_zip.py (default: alongside this script)")
    ap.add_argument("--zip-level", type=int, default=9, choices=range(1,10))
    ap.add_argument("--split", type=str, default=None, help="Split size for zip (e.g., 2000m)")
    ap.add_argument("--zip-out", type=Path, default=None, help="Output zip path (default: <out>.zip)")
    ap.add_argument("--manifest", type=Path, default=None, help="Optional manifest CSV (passed to archive_zip)")
    ap.add_argument("--finalize-purge-metadata", action="store_true",
                    help="Also delete summary.csv and run.log during finalization.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover WAVs
    if args.recursive:
        wavs = list(in_dir.rglob("*.WAV")) + list(in_dir.rglob("*.wav"))
    else:
        wavs = list(in_dir.glob("*.WAV")) + list(in_dir.glob("*.wav"))
    wavs = sorted({w.resolve() for w in wavs})
    if not wavs:
        print("No WAV files found."); return

    # logging
    if args.log_file is None:
        log_path = out_dir / "run.log"
    else:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # prepare output dirs
    for d in ("Bat", "Bat_lowconf", "NoBat"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "summary.csv"
    headers = ["file","sr","duration_s","calls","snr_db","tonal_ratio","decision","lowconf","score","png","dst_wav"]

    with open(csv_path, "w", newline="") as fcsv, open(log_path, "a", encoding="utf-8") as logfh:
        writer = csv.DictWriter(fcsv, fieldnames=headers)
        writer.writeheader(); fcsv.flush(); os.fsync(fcsv.fileno())
        _log(logfh, f"[INFO] Start classify. in={in_dir} out={out_dir} workers={args.workers} wavs={len(wavs)} snr_mode={args.snr_mode}")

        tasks = [
            (str(w), str(out_dir), args.sr, args.block_sec, args.no_png, args.png_all,
             args.db_bin_width, args.snr_mode, args.lc_min_calls, args.lc_snr_min, args.lc_tonal_max,
             args.debug, str(log_path))
            for w in wavs
        ]

        total = len(tasks); done = 0; since_sync = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_file, t) for t in tasks]
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    _log(logfh, f"[ERROR] future crashed: {e}\n{traceback.format_exc()}")
                    continue

                done += 1
                if isinstance(res, dict) and "error" in res:
                    _log(logfh, res["error"]); continue

                writer.writerow(res)
                since_sync += 1
                if args.csv_fsync_every > 0 and since_sync >= args.csv_fsync_every:
                    fcsv.flush(); os.fsync(fcsv.fileno()); since_sync = 0
                if done % 10 == 0 or done == total:
                    print(f"[{done}/{total}] processed …")

        fcsv.flush(); os.fsync(fcsv.fileno())
        _log(logfh, "[INFO] Finished classification.")

    # counts
    def count_wavs(p: Path):
        return sum(1 for _ in p.glob("*.WAV")) + sum(1 for _ in p.glob("*.wav"))
    bat_n   = count_wavs(out_dir / "Bat")
    low_n   = count_wavs(out_dir / "Bat_lowconf")
    nobat_n = count_wavs(out_dir / "NoBat")
    print(f"\n[SUMMARY] Bat={bat_n}  Bat_lowconf={low_n}  NoBat={nobat_n}")
    print(f"CSV: {csv_path}")
    print(f"Log: {log_path}")

    # ---- Finalization (optional) ----
    if args.finalize:
        # resolve archive_zip.py
        if args.archive_script is None:
            archive_script = (Path(__file__).parent / "archive_zip.py").resolve()
        else:
            archive_script = args.archive_script.resolve()
        if not archive_script.exists():
            print(f"[FINALIZE] archive_zip.py not found at {archive_script}; aborting finalization.")
            return

        if not args.force:
            print("\n[FINALIZE] --finalize requested but --force not set.")
            print("           Classification is complete; skipping destructive steps.")
            print("           Re-run with --finalize --force to actually cleanup + archive.")
            return

        print("\n[FINALIZE] Starting cleanup + archive...")
        try:
            stats = finalize_cleanup_and_zip(
                out_dir=out_dir,
                archive_script=archive_script,
                zip_level=args.zip_level,
                split=args.split,
                zip_out=args.zip_out,
                manifest=args.manifest,
                force=True,
                keep_metadata = not args.finalize_purge_metadata
            )
        except Exception as e:
            print(f"[FINALIZE] FAILED: {e}")
            return

        print(f"[FINALIZE] PNGs removed: {stats['png_removed']} (~{bytes_h(stats['png_bytes'])})")
        print(f"[FINALIZE] Archive: {stats['zip']}  (elapsed {stats['elapsed_zip_s']:.2f}s)")
        print(f"[FINALIZE] Files removed: {stats['files_removed']} (~{bytes_h(stats['bytes_removed'])})")
        print("[FINALIZE] Kept:")
        for k in stats["kept"]:
            print(f"  {k}")

        print("\n[FINALIZE] Done. Your OUT directory now contains only the archive file(s).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
