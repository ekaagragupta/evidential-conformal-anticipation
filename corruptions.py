"""Temporally consistent synthetic corruptions for dashcam video (numpy uint8 [N,H,W,3], RGB).
Severity 1..5. Fog and low-light are video-level effects (same haze / gain on every frame);
noise and rain are re-sampled per frame (sensor noise / falling streaks). Use the SAME corruption seed for
every method so comparisons are fair. Cite the video-robustness benchmarks (Kinetics-C/UCF101-C style) in the paper."""
import cv2
import numpy as np


def _clip(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def gaussian_noise(x, s, rng):
    sigma = [8, 16, 26, 38, 50][s - 1]
    return _clip(x.astype(np.float32) + rng.normal(0, sigma, x.shape))


def fog(x, s, rng):
    a = [0.15, 0.25, 0.35, 0.45, 0.6][s - 1]
    H, W = x.shape[1:3]
    low = rng.random((6, 8)).astype(np.float32)
    haze = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)[None, :, :, None]
    haze = (haze - haze.min()) / (haze.max() - haze.min() + 1e-6)
    m = a * (0.5 + 0.5 * haze)
    return _clip(x.astype(np.float32) * (1 - m) + 225.0 * m)


def rain(x, s, rng):
    n = [200, 400, 700, 1000, 1500][s - 1]
    N, H, W, _ = x.shape
    out = x.astype(np.float32) * 0.9
    for i in range(N):
        layer = np.zeros((H, W), np.float32)
        for _ in range(n):
            x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
            L = int(rng.integers(8, 20))
            x1, y1 = x0 + int(L * 0.2), y0 + L
            cv2.line(layer, (x0, y0), (x1, y1), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None]
        out[i] = out[i] * (1 - 0.5 * layer) + 210.0 * 0.5 * layer
    return _clip(out)


def motion_blur(x, s, rng):
    k = [3, 5, 7, 9, 13][s - 1]
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    return np.stack([cv2.filter2D(f, -1, ker) for f in x])


def low_light(x, s, rng):
    gamma = [1.4, 1.8, 2.3, 2.9, 3.6][s - 1]
    y = 255.0 * (x.astype(np.float32) / 255.0) ** gamma * 0.8
    return _clip(y + rng.normal(0, 2 + 2 * s, x.shape))


CORRUPTIONS = dict(gaussian_noise=gaussian_noise, fog=fog, rain=rain, motion_blur=motion_blur, low_light=low_light)


def corrupt_video(frames, name, severity, seed=0):
    if name in (None, "none"):
        return frames
    rng = np.random.default_rng(seed)
    return CORRUPTIONS[name](frames, int(severity), rng)
