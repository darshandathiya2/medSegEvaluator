"""
===========================================================
📘 image_quality.py — Medical Image Quality Evaluation Tool
===========================================================

Computes objective image quality metrics for medical images:
- SNR (Signal-to-Noise Ratio)
- Entropy (Image detail richness)
- Blur (Variance of Laplacian)
- SSIM (Structural Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- NIQE (No-reference natural image quality, optional)

Generates:
1. CSV summary file with all metrics
2. Histogram visualizations for quality comparison
===========================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Optional: NIQE (available in skimage >= 0.19)
try:
    from skimage.metrics import niqe
    HAS_NIQE = True
except ImportError:
    HAS_NIQE = False


# -----------------------------------------------------------
# 🔹 Basic Quality Functions
# -----------------------------------------------------------

def compute_snr(image):
    mean_signal = np.mean(image)
    noise = np.std(image)
    return 10 * np.log10((mean_signal ** 2) / (noise ** 2 + 1e-8))

def compute_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,1])
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def compute_blur(image):
    # Normalize and convert to uint8
    img_uint8 = (255 * image / (np.max(image) + 1e-8)).astype(np.uint8)

    # If the image is flat (no edges), return 0 blur variance
    if np.std(img_uint8) == 0:
        return 0.0

    lap_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    return lap_var  # lower = blurrier
# -----------------------------------------------------------
# 🔹 Quality Report for a Single Image
# -----------------------------------------------------------
def image_quality_metrics(image, ref=None):
    """Compute quality metrics for one image (optionally vs reference)."""
    image = img_as_float(image)
    report = {
        "SNR": compute_snr(image),
        "Entropy": compute_entropy(image),
        "Blur": compute_blur(image),
    }

    if ref is not None:
        ref = img_as_float(ref)
        report["PSNR"] = psnr(ref, image, data_range=1.0)
        report["SSIM"] = ssim(ref, image, data_range=1.0)
    else:
        report["PSNR"] = np.nan
        report["SSIM"] = np.nan

    if HAS_NIQE:
        try:
            report["NIQE"] = niqe(image)
        except Exception:
            report["NIQE"] = np.nan
    else:
        report["NIQE"] = np.nan

    return report



def evaluate_volume_from_array(volume):
    """
    Evaluates SNR, entropy, and blur for each slice of a given 3D image volume.

    Args:
        volume (numpy.ndarray): 3D NumPy array, shape (H, W, D)
    """
    snr_list, entropy_list, blur_list = [], [], []

    for i in tqdm(range(volume.shape[2]), desc="Processing slices"):
        slice_img = volume[:, :, i]

        if np.all(slice_img == 0):
            continue  # skip empty slices

        # Normalize slice to [0,1]
        slice_img = (slice_img - np.min(slice_img)) / (np.ptp(slice_img) + 1e-8)

        snr_list.append(compute_snr(slice_img))
        entropy_list.append(compute_entropy(slice_img))
        blur_list.append(compute_blur(slice_img))

    print("\n✅ Average Metrics for Volume:")
    print(f"SNR     : {np.mean(snr_list):.3f}")
    print(f"Entropy : {np.mean(entropy_list):.3f}")
    print(f"Blur    : {np.mean(blur_list):.3f}")

    return snr_list, entropy_list, blur_list


def plot_quality_per_slice(snr_list, entropy_list, blur_list, save_path=None):
  """Visualize SNR, Entropy, and Blur across all slices in the 3D volume."""
  num_slices = len(snr_list)
  slices = np.arange(1, num_slices + 1)

  plt.figure(figsize=(12, 6))

  # --- SNR ---
  plt.subplot(3, 1, 1)
  plt.plot(slices, snr_list, color='orange', linewidth=2)
  plt.title("Per-Slice Signal-to-Noise Ratio (SNR)")
  plt.ylabel("SNR (dB)")
  plt.grid(True, linestyle='--', alpha=0.5)

  # --- Entropy ---
  plt.subplot(3, 1, 2)
  plt.plot(slices, entropy_list, color='teal', linewidth=2)
  plt.title("Per-Slice Entropy (Detail Richness)")
  plt.ylabel("Entropy")
  plt.grid(True, linestyle='--', alpha=0.5)

  # --- Blur ---
  plt.subplot(3, 1, 3)
  plt.plot(slices, blur_list, color='purple', linewidth=2)
  plt.title("Per-Slice Blur (Laplacian Variance)")
  plt.xlabel("Slice Index")
  plt.ylabel("Blur")
  plt.grid(True, linestyle='--', alpha=0.5)

  plt.tight_layout()

  if save_path:
      plt.savefig(save_path, dpi=300)
      print(f"📊 Saved plot to: {save_path}")
  else:
      plt.show()
