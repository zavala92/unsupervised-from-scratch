#!/usr/bin/env python3
"""Colour quantisation with K-means: reduce an image to K colours.

    python3 compress_image.py                 # bundled sample photo, K = 16
    python3 compress_image.py photo.jpg 8     # your own image, K = 8

Each pixel is a point in [0, 1]^3. The K centroids are the K colours that
minimise the mean squared reconstruction error, and every pixel is then stored
as the index of its nearest centroid. Writing out K centroids plus one index
per pixel is far cheaper than 24 bits per pixel, which is what makes this
compression; it is lossy, and the distortion is exactly the error you accept.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kmeans import kmeans, assign_clusters

SUBSAMPLE = 8000        # centroids are estimated fine from a sample of pixels


def load(path=None):
    if path is None:
        import matplotlib.cbook as cbook
        return plt.imread(cbook.get_sample_data("grace_hopper.jpg"))
    return plt.imread(path)


def quantise(img, K, seed=2):
    px = img.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    sub = px[rng.choice(len(px), min(SUBSAMPLE, len(px)), replace=False)]
    centroids, _, hist = kmeans(sub, K, np.random.default_rng(seed), max_iters=40)
    idx = assign_clusters(px, centroids)
    return centroids[idx].reshape(img.shape), centroids, hist[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    img = load(path).astype(float)
    if img.max() > 1.0:
        img /= 255.0
    img = img[..., :3]
    h, w, _ = img.shape

    n_colours = len(np.unique((img.reshape(-1, 3) * 255).astype(np.uint8), axis=0))
    recon, centroids, J = quantise(img, K)

    bits_before = h * w * 24
    bits_after = h * w * int(np.ceil(np.log2(K))) + K * 24

    print(f"  image           {w} x {h}, {n_colours:,} distinct colours")
    print(f"  palette         K = {K}")
    print(f"  distortion      {J:.5f}  (mean squared error per pixel, RGB in [0,1])")
    print(f"  before          {bits_before:,} bits")
    print(f"  after           {bits_after:,} bits "
          f"= {h*w} x {int(np.ceil(np.log2(K)))} + {K} x 24")
    print(f"  ratio           {bits_before / bits_after:.1f} x smaller")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(img, 0, 1)); axes[0].axis("off")
    axes[0].set_title(f"original, {n_colours:,} colours")
    axes[1].imshow(np.clip(recon, 0, 1)); axes[1].axis("off")
    axes[1].set_title(f"K = {K}, {bits_before / bits_after:.1f}x smaller")
    out = Path(__file__).resolve().parent / f"compressed_K{K}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote           {out.name}")


if __name__ == "__main__":
    main()
