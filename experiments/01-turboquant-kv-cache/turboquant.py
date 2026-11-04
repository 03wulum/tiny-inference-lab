"""
Minimal NumPy implementation of TurboQuant (Zandieh, Daliri, Hadian, Mirrokni).
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
arXiv:2504.19874, ICLR 2026.

Two quantizers from the paper:

  TurboQuantMSE  - randomly rotate the (unit-normalised) vector, then quantize every
                   coordinate independently with a Lloyd-Max codebook that is optimal for
                   the known marginal distribution of a coordinate of a random unit vector.
                   Minimises mean squared error.

  TurboQuantProd - spend (b-1) bits on TurboQuantMSE, then 1 bit per coordinate on a
                   QJL sketch (sign of a Gaussian projection) of the residual. Gives an
                   UNBIASED inner-product estimator, which is what attention needs.

Plus one baseline, `UniformMinMax`, the usual per-vector (per-token) asymmetric
min/max integer quantizer used by most KV-cache quantization schemes (e.g. KIVI).

Everything here is data-oblivious: no calibration set, no training. That is the point
of the paper and why it is attractive for on-device / streaming KV caches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

NORM_BITS = 16  # we store each vector's L2 norm as fp16


# --------------------------------------------------------------------------------------
# Codebook: Lloyd-Max quantizer for one coordinate of a uniformly random unit vector
# --------------------------------------------------------------------------------------
def coord_density(d: int, n_grid: int = 200_001) -> tuple[np.ndarray, np.ndarray]:
    """Density of one coordinate of a random point on S^{d-1}:
    f(t) proportional to (1 - t^2)^((d-3)/2) on [-1, 1]. Returned on a grid, normalised."""
    # The mass lives in roughly +-8/sqrt(d); a tighter grid gives a more accurate codebook.
    lim = min(1.0, 10.0 / math.sqrt(d))
    t = np.linspace(-lim, lim, n_grid)
    logf = 0.5 * (d - 3) * np.log1p(-np.clip(t * t, 0, 1 - 1e-12))
    f = np.exp(logf - logf.max())
    f /= np.trapezoid(f, t)
    return t, f


@lru_cache(maxsize=64)
def lloyd_max_codebook(d: int, bits: int, iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Optimal scalar quantizer (centroids, decision boundaries) for coord_density(d)."""
    t, f = coord_density(d)
    k = 2 ** bits
    cdf = np.cumsum(f)
    cdf /= cdf[-1]
    # initialise centroids at quantile midpoints
    centroids = np.interp((np.arange(k) + 0.5) / k, cdf, t)
    for _ in range(iters):
        bounds = 0.5 * (centroids[1:] + centroids[:-1])
        idx = np.searchsorted(bounds, t)
        mass = np.bincount(idx, weights=f, minlength=k)
        first = np.bincount(idx, weights=f * t, minlength=k)
        new = first / np.maximum(mass, 1e-300)
        if np.max(np.abs(new - centroids)) < 1e-12:
            centroids = new
            break
        centroids = new
    bounds = 0.5 * (centroids[1:] + centroids[:-1])
    return centroids, bounds


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random orthogonal matrix via QR of a Gaussian matrix."""
    g = rng.standard_normal((d, d))
    q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))


# --------------------------------------------------------------------------------------
# Quantizers. Each has quantize(X) -> codes and dequantize(codes) -> X_hat, and reports
# bits_per_coord(d) INCLUDING per-vector side information (norms, scales, zero points).
# --------------------------------------------------------------------------------------
@dataclass
class TurboQuantMSE:
    d: int
    bits: int
    seed: int = 0

    def __post_init__(self):
        # Separate seed stream so the rotation can never coincide with data drawn from
        # np.random.default_rng(seed) (a correlated rotation silently breaks the guarantees).
        rng = np.random.default_rng((0x7B0, self.seed))
        self.rot = random_rotation(self.d, rng)
        self.centroids, self.bounds = lloyd_max_codebook(self.d, self.bits)

    def quantize(self, X: np.ndarray):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        U = X / np.maximum(norms, 1e-12)
        Y = U @ self.rot.T  # rotated unit vectors: every coord ~ coord_density(d)
        idx = np.searchsorted(self.bounds, Y).astype(np.uint8)
        return idx, norms.astype(np.float16)

    def dequantize(self, codes) -> np.ndarray:
        idx, norms = codes
        return (self.centroids[idx] @ self.rot) * norms.astype(np.float64)

    def __call__(self, X):
        return self.dequantize(self.quantize(X))

    def bits_per_coord(self) -> float:
        return self.bits + NORM_BITS / self.d


@dataclass
class TurboQuantProd:
    """(bits-1)-bit TurboQuantMSE + 1-bit QJL on the residual => unbiased inner products."""
    d: int
    bits: int
    seed: int = 0

    def __post_init__(self):
        assert self.bits >= 1
        rng = np.random.default_rng((0x51E, self.seed))
        self.mse = TurboQuantMSE(self.d, self.bits - 1, self.seed) if self.bits > 1 else None
        self.S = rng.standard_normal((self.d, self.d))  # QJL projection

    def quantize(self, X: np.ndarray):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        U = X / np.maximum(norms, 1e-12)
        if self.mse is not None:
            mse_codes = self.mse.quantize(U)
            U_hat = self.mse.dequantize(mse_codes)
        else:
            mse_codes, U_hat = None, np.zeros_like(U)
        R = U - U_hat
        r_norm = np.linalg.norm(R, axis=1, keepdims=True)
        signs = np.signbit(R @ self.S.T)  # 1 bit per coordinate
        return mse_codes, signs, r_norm.astype(np.float16), norms.astype(np.float16)

    def dequantize(self, codes) -> np.ndarray:
        mse_codes, signs, r_norm, norms = codes
        U_hat = self.mse.dequantize(mse_codes) if mse_codes is not None else 0.0
        s = np.where(signs, -1.0, 1.0)
        qjl = math.sqrt(math.pi / 2) / self.d * r_norm.astype(np.float64) * (s @ self.S)
        return (U_hat + qjl) * norms.astype(np.float64)

    def __call__(self, X):
        return self.dequantize(self.quantize(X))

    def bits_per_coord(self) -> float:
        # MSE part stores its own (unit) norm, which we do not need to keep: count norm + r_norm.
        return self.bits + 2 * NORM_BITS / self.d


@dataclass
class UniformMinMax:
    """Baseline: per-vector asymmetric integer quantization with fp16 scale + zero point."""
    d: int
    bits: int

    def quantize(self, X: np.ndarray):
        lo = X.min(axis=1, keepdims=True)
        hi = X.max(axis=1, keepdims=True)
        levels = 2 ** self.bits - 1
        scale = np.maximum(hi - lo, 1e-12) / levels
        q = np.clip(np.round((X - lo) / scale), 0, levels).astype(np.uint8)
        return q, scale.astype(np.float16), lo.astype(np.float16)

    def dequantize(self, codes) -> np.ndarray:
        q, scale, lo = codes
        return q * scale.astype(np.float64) + lo.astype(np.float64)

    def __call__(self, X):
        return self.dequantize(self.quantize(X))

    def bits_per_coord(self) -> float:
        return self.bits + 2 * NORM_BITS / self.d


# Paper's analytical guarantees (Theorems 1-3), for unit vectors.
def mse_upper_bound(bits: int) -> float:
    return math.sqrt(3) * math.pi / 2 * 4.0 ** (-bits)


def mse_lower_bound(bits: int) -> float:
    return 4.0 ** (-bits)
