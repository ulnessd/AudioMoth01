# -*- coding: utf-8 -*-
"""
A robust, console-based script to generate a video (.mp4) of a scrolling
spectrogram with synchronized, frequency-shifted audio.

This script is designed to be run directly from an IDE like PyCharm.
Simply edit the variables in the "SETTINGS" section below and run the file.

Required Libraries:
- soundfile, scipy, matplotlib, numpy, imageio, imageio-ffmpeg

You can install these with pip:
pip install soundfile scipy matplotlib numpy imageio imageio-ffmpeg
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
import imageio.v2 as imageio
import tempfile
import shutil
import subprocess

# Use a non-interactive backend for Matplotlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process_audio(original_data, sample_rate, shift_freq, amplification=2.0):
    """
    Applies frequency shifting and returns the processed audio data.
    """
    nperseg = int(sample_rate * 0.02)
    f, t, Zxx = stft(original_data, sample_rate, nperseg=nperseg)

    freq_resolution = f[1] - f[0]
    shift_bins = int(shift_freq / freq_resolution)

    Zxx_shifted = np.zeros_like(Zxx)
    if shift_bins < Zxx.shape[0]:
        source_slice = Zxx[shift_bins:, :]
        Zxx_shifted[:source_slice.shape[0], :] = source_slice

    _, istft_result = istft(Zxx_shifted, sample_rate, nperseg=nperseg)

    istft_result = istft_result[:len(original_data)]

    max_val = np.max(np.abs(istft_result))
    if max_val > 0:
        normalized_signal = istft_result / max_val
    else:
        normalized_signal = istft_result

    amplified_signal = normalized_signal * amplification
    return np.clip(amplified_signal, -1.0, 1.0).astype(np.float32)


def main():
    """
    Main function to run the video generation process.
    """
    # ===================================================================
    # --- SETTINGS: Updated with your parameters ---
    # ===================================================================

    INPUT_FILE = "20250802_010000T.WAV"
    OUTPUT_FILE = "20250802_010000T.mp4"
    SHIFT_FREQUENCY = 20000
    FRAMES_PER_SECOND = 30

    # ===================================================================
    # --- End of settings. No need to edit below this line. ---
    # ===================================================================

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"Error: Input file not found at '{input_path}'")
        print("Please update the INPUT_FILE variable in the script.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("Step 1/4: Loading and processing audio...")
        original_data, sample_rate = sf.read(input_path, dtype='float32')
        if original_data.ndim > 1:
            original_data = original_data[:, 0]

        processed_audio = process_audio(original_data, sample_rate, SHIFT_FREQUENCY)
        duration_s = len(processed_audio) / sample_rate
        total_frames = int(duration_s * FRAMES_PER_SECOND)

        print("Step 2/4: Generating base spectrogram (this may take a moment)...")

        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        nperseg = int(sample_rate * 0.02)
        Pxx, freqs, bins, im = ax.specgram(processed_audio + 1e-20, NFFT=nperseg, Fs=sample_rate, noverlap=nperseg // 2,
                                           cmap='inferno')
        Pxx_dB = 10 * np.log10(Pxx)
        vmax = Pxx_dB.max()
        vmin = vmax - 90
        im.set_clim(vmin, vmax)
        ax.set_title(f"Spectrogram of {input_path.name} (Shifted by {SHIFT_FREQUENCY} Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 20000)
        fig.colorbar(im, ax=ax).set_label('Intensity [dB]')
        fig.tight_layout()

        line = ax.axvline(x=0, color='cyan', linewidth=2)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            print(f"Step 3/4: Rendering {total_frames} video frames...")

            frame_paths = []
            for i in range(total_frames):
                current_time = i / FRAMES_PER_SECOND
                line.set_xdata([current_time, current_time])

                frame_path = temp_dir / f"frame_{i:05d}.png"
                plt.savefig(frame_path, dpi=100)
                frame_paths.append(frame_path)

                print(f"  ...wrote frame {i + 1}/{total_frames}", end='\r')

            plt.close(fig)

            print("\nStep 4/4: Combining frames and audio into MP4...")
            temp_audio_path = temp_dir / "audio.wav"
            sf.write(temp_audio_path, processed_audio, sample_rate)

            temp_video_path = temp_dir / "silent_video.mp4"
            with imageio.get_writer(temp_video_path, fps=FRAMES_PER_SECOND, codec='libx264', quality=8) as writer:
                for frame_path in frame_paths:
                    writer.append_data(imageio.imread(frame_path))

            # --- FIX: Use a direct and robust ffmpeg call via subprocess ---
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', str(temp_video_path),
                '-i', str(temp_audio_path),
                '-c:v', 'copy',  # Copy the video stream without re-encoding
                '-c:a', 'aac',  # Re-encode audio to aac
                '-shortest',  # Finish encoding when the shortest input stream ends
                str(output_path)
            ]

            subprocess.run(command, check=True, capture_output=True)

        print(f"\nSuccessfully created video: {output_path}")

    except subprocess.CalledProcessError as e:
        print("\n--- FFMPEG ERROR ---")
        print(f"FFmpeg failed with exit code {e.returncode}")
        print("STDERR:")
        print(e.stderr.decode())
        print("--------------------")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
