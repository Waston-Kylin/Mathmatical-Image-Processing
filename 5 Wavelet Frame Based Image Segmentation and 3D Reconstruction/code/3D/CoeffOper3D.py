# -*- coding: utf-8 -*-
"""Operations on nested 3-D wavelet-frame coefficient arrays."""

import numpy as np


def wthresh(x, opt, threshold):
    """Elementwise soft thresholding."""
    if opt == "s":
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
    raise ValueError(f"Unsupported thresholding option: {opt}")


def _validate_coeff_structure(alpha, beta=None):
    if not isinstance(alpha, (list, tuple)) or len(alpha) == 0:
        raise ValueError("alpha must be a non-empty list/tuple of levels.")

    if beta is not None:
        if not isinstance(beta, (list, tuple)) or len(beta) != len(alpha):
            raise ValueError("alpha and beta must have the same number of levels.")

    for level_idx, level in enumerate(alpha):
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            raise ValueError("Each level must contain at least two filter bands.")
        n = len(level)
        if beta is not None:
            other = beta[level_idx]
            if not isinstance(other, (list, tuple)) or len(other) != n:
                raise ValueError("alpha and beta must have identical band layouts.")
        for i in range(n):
            if len(level[i]) != n:
                raise ValueError("Expected an n x n x n coefficient layout.")
            if beta is not None and len(other[i]) != n:
                raise ValueError("alpha and beta must have identical band layouts.")
            for j in range(n):
                if len(level[i][j]) != n:
                    raise ValueError("Expected an n x n x n coefficient layout.")
                if beta is not None and len(other[i][j]) != n:
                    raise ValueError("alpha and beta must have identical band layouts.")
                for k in range(n):
                    arr = np.asarray(level[i][j][k])
                    if beta is not None:
                        brr = np.asarray(other[i][j][k])
                        if arr.shape != brr.shape:
                            raise ValueError(
                                "Coefficient and threshold shapes differ: "
                                f"{arr.shape} vs {brr.shape}."
                            )


def CoeffOper3D(op, alpha, beta=None):
    """
    Apply an operation to nested 3-D wavelet-frame coefficients.

    Supported operations:
      '='  copy coefficients
      '+'  elementwise addition
      '-'  elementwise subtraction
      '*'  scalar multiplication
      's'  elementwise soft thresholding
      'vs' isotropic/group soft thresholding across the high-pass
           subbands at each spatial location; the low-pass (0,0,0) band
           is left unchanged.

    For 'vs', at each spatial position x:
        r(x) = sqrt(sum_{j != (0,0,0)} |c_j(x)|^2)
        c_j(x) <- max(r(x)-tau(x), 0) * c_j(x)/r(x)

    This is the joint shrinkage required by the isotropic wavelet-frame
    regularizer, rather than seven independent scalar shrinkage operations.
    """
    if op not in {"=", "+", "-", "*", "s", "vs"}:
        raise ValueError(f"Unknown operation: {op}")

    if op == "*":
        if beta is None or np.isscalar(beta) is False:
            raise ValueError("For '*' beta must be a scalar.")
    elif op != "=":
        if beta is None:
            raise ValueError(f"Operation '{op}' requires beta.")

    _validate_coeff_structure(alpha, beta if op not in {"=", "*"} else None)

    result = []

    for level_idx, level in enumerate(alpha):
        n = len(level)
        # Compute the joint norm of all high-pass bands.
        if op == "vs":
            first = np.asarray(level[0][0][1])
            joint_sq = np.zeros_like(first, dtype=first.dtype)

            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if (i, j, k) == (0, 0, 0):
                            continue
                        band = np.asarray(level[i][j][k], dtype=first.dtype)
                        if band.shape != first.shape:
                            raise ValueError(
                                "All high-pass bands must have "
                                "identical spatial shapes for isotropic shrinkage."
                            )
                        joint_sq += band * band

            joint_norm = np.sqrt(joint_sq)
            safe_norm = np.maximum(joint_norm, 1e-20)

        out_level = []
        for i in range(n):
            out_i = []
            for j in range(n):
                out_j = []
                for k in range(n):
                    a = np.asarray(level[i][j][k])

                    if op == "=":
                        value = a.copy()
                    elif op == "+":
                        value = a + beta[level_idx][i][j][k]
                    elif op == "-":
                        value = a - beta[level_idx][i][j][k]
                    elif op == "*":
                        value = a * beta
                    elif op == "s":
                        value = wthresh(
                            a, "s", beta[level_idx][i][j][k]
                        )
                    else:  # op == "vs"
                        if (i, j, k) == (0, 0, 0):
                            value = a.copy()
                        else:
                            threshold = np.asarray(
                                beta[level_idx][i][j][k],
                                dtype=first.dtype,
                            )
                            shrink_factor = (
                                np.maximum(joint_norm - threshold, 0.0)
                                / safe_norm
                            )
                            value = a * shrink_factor

                    out_j.append(value)
                out_i.append(out_j)
            out_level.append(out_i)
        result.append(out_level)

    return result
