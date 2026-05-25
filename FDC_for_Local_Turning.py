#!/usr/bin/env python3
"""
Factorized Derivative Calculus for Local Turning (FDC-LT).

Reference implementation and reproducibility script for the FDC-LT numerical
experiments. The script demonstrates analytic derivative-symbol construction,
shifted-coordinate safeguards, real-channel aggregation, fixed-structure least
squares fitting, derivative-aware stacked least squares, finite-difference
validation, turning-region clustering, and high-noise synthetic stress tests.

The implementation is intentionally self-contained so that it can be published
as a single GitHub script. It does not require external data. All generated
outputs are written to a user-selected output directory.

Dependencies:
    numpy
    matplotlib

Example:
    python FDC_for_Local_Turning.py
    python FDC_for_Local_Turning.py --output-dir outputs/fdc_lt --dtype float64

Default outputs:
    FDC-LT_high_noise_outputs/results/*.csv
    FDC-LT_high_noise_outputs/results/run_summary.txt
    FDC-LT_high_noise_outputs/figures/*.png
"""

from __future__ import annotations

import argparse
import csv

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray
DEFAULT_OUTPUT_DIR = Path("FDC-LT_high_noise_outputs")
DEFAULT_DTYPE = np.float64
DEFAULT_RANDOM_SEED = 20260522


def scalar(x: object, dtype: object = np.float64) -> object:
    """Return x as a NumPy scalar with the requested dtype."""

    return np.asarray(x, dtype=dtype).reshape(())


def real_array(x: object, dtype: object = np.float64) -> Array:
    """Return x as a real NumPy array with the requested dtype."""

    return np.asarray(x, dtype=dtype)


def complex_array(x: object, dtype: object = np.float64) -> Array:
    """Return x as a complex array associated with the requested real dtype."""

    complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
    return np.asarray(x, dtype=complex_dtype)


def ensure_output_dirs(base_dir: Path) -> tuple[Path, Path]:
    """Create output directories and return result and figure paths."""

    result_dir = base_dir / "results"
    figure_dir = base_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, figure_dir


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Write rows to a CSV file with standard quoting and newline handling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def factorial_int(n: int, dtype: object = np.float64) -> object:
    """Compute n! as a NumPy scalar using an integer loop."""

    value = scalar(1.0, dtype)
    for k in range(2, n + 1):
        value = value * scalar(k, dtype)
    return value


def binom_int(n: int, k: int, dtype: object = np.float64) -> object:
    """Compute a binomial coefficient as a NumPy scalar."""

    if k < 0 or k > n:
        return scalar(0.0, dtype)
    k_eff = k if k <= n - k else n - k
    value = scalar(1.0, dtype)
    for j in range(1, k_eff + 1):
        value = value * scalar(n - k_eff + j, dtype) / scalar(j, dtype)
    return value


def falling_factorial(m: object, k: int, dtype: object = np.float64) -> object:
    """Compute the generalized falling factorial (m)_k."""

    value = scalar(1.0, dtype)
    m_value = scalar(m, dtype)
    for j in range(k):
        value = value * (m_value - scalar(j, dtype))
    return value


def safe_relative_error(a: object, b: object, dtype: object = np.float64) -> object:
    """Return |a-b|/(1+|b|) with dtype-aware arithmetic."""

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    numerator = np.abs(a_arr - b_arr)
    denominator = scalar(1.0, dtype) + np.abs(b_arr)
    return np.max(numerator / denominator)


def polynomial_value(coeffs: Array, tau: Array, dtype: object = np.float64) -> Array:
    """Evaluate sum_j coeffs[j] tau^j using Horner's method."""

    tau_arr = real_array(tau, dtype)
    coeff_arr = real_array(coeffs, dtype)
    value = np.zeros_like(tau_arr, dtype=dtype) + scalar(0.0, dtype)
    for coeff in coeff_arr[::-1]:
        value = value * tau_arr + scalar(coeff, dtype)
    return value


def polynomial_derivative_value(coeffs: Array, tau: Array, order: int, dtype: object = np.float64) -> Array:
    """Evaluate the requested derivative of a polynomial envelope."""

    tau_arr = real_array(tau, dtype)
    coeff_arr = real_array(coeffs, dtype)
    if order < 0:
        raise ValueError("Derivative order must be nonnegative.")
    if order == 0:
        return polynomial_value(coeff_arr, tau_arr, dtype)
    value = np.zeros_like(tau_arr, dtype=dtype) + scalar(0.0, dtype)
    for power in range(order, coeff_arr.size):
        multiplier = falling_factorial(scalar(power, dtype), order, dtype)
        value = value + scalar(coeff_arr[power], dtype) * multiplier * np.power(tau_arr, scalar(power - order, dtype), dtype=dtype)
    return value


@dataclass
class OscillatoryEnvelopeNode:
    """Sparse oscillatory-envelope node.

    Z(tau) = b scale(tau)^m exp(alpha(tau)) exp(i(omega tau + phi)).

    For unshifted nodes, scale(tau)=tau.
    For shifted nodes, scale(tau)=(tau+delta)/(T+delta), which leaves quotient
    symbols unchanged except for replacing m/tau by m/(tau+delta).
    """

    amplitude: object
    m: object
    omega: object
    phi: object
    alpha_coeffs: Array
    T: object = np.float64(1.0)
    shifted_delta: Optional[object] = None
    label: str = "node"

    def active(self, dtype: object = np.float64) -> bool:
        """Return whether the node has nonzero effective amplitude."""

        return bool(np.abs(scalar(self.amplitude, dtype)) > scalar(0.0, dtype))

    def denominator(self, tau: Array, dtype: object = np.float64) -> Array:
        """Return tau for unshifted nodes or tau+delta for shifted nodes."""

        tau_arr = real_array(tau, dtype)
        if self.shifted_delta is None:
            return tau_arr
        return tau_arr + scalar(self.shifted_delta, dtype)

    def scale(self, tau: Array, dtype: object = np.float64) -> Array:
        """Return the positive scale used in the node value."""

        denom = self.denominator(tau, dtype)
        if self.shifted_delta is None:
            return denom
        return denom / (scalar(self.T, dtype) + scalar(self.shifted_delta, dtype))

    def value(self, tau: Array, dtype: object = np.float64) -> Array:
        """Evaluate the complex node value."""

        tau_arr = real_array(tau, dtype)
        scale_arr = self.scale(tau_arr, dtype)
        if np.any(scale_arr <= scalar(0.0, dtype)):
            raise ValueError(f"{self.label}: scale must be positive for the chosen power package.")
        alpha_val = polynomial_value(self.alpha_coeffs, tau_arr, dtype)
        angle = scalar(self.omega, dtype) * tau_arr + scalar(self.phi, dtype)
        magnitude = scalar(self.amplitude, dtype) * np.power(scale_arr, scalar(self.m, dtype), dtype=dtype) * np.exp(alpha_val, dtype=dtype)
        return complex_array(magnitude, dtype) * np.exp(scalar(1j, np.complex128) * complex_array(angle, dtype))

    def q_derivatives(self, tau: Array, max_order: int, dtype: object = np.float64) -> list[Array]:
        """Return q^(j)(tau), j=0,...,max_order."""

        if max_order < 0:
            raise ValueError("max_order must be nonnegative.")
        tau_arr = real_array(tau, dtype)
        denom = self.denominator(tau_arr, dtype)
        if np.any(denom <= scalar(0.0, dtype)):
            raise ValueError(f"{self.label}: quotient denominator must be positive.")
        m_value = scalar(self.m, dtype)
        q_list: list[Array] = []
        alpha_first = polynomial_derivative_value(self.alpha_coeffs, tau_arr, 1, dtype)
        q0 = m_value / denom + alpha_first + scalar(1j, np.complex128) * complex_array(scalar(self.omega, dtype), dtype)
        q_list.append(complex_array(q0, dtype))
        for j in range(1, max_order + 1):
            sign = scalar(1.0, dtype) if j % 2 == 0 else scalar(-1.0, dtype)
            scale_part = sign * factorial_int(j, dtype) * m_value / np.power(denom, scalar(j + 1, dtype), dtype=dtype)
            alpha_part = polynomial_derivative_value(self.alpha_coeffs, tau_arr, j + 1, dtype)
            q_list.append(complex_array(scale_part + alpha_part, dtype))
        return q_list

    def derivative_symbols(self, tau: Array, max_order: int, dtype: object = np.float64) -> list[Array]:
        """Return P_n(tau)=Z^(n)(tau)/Z(tau), n=0,...,max_order."""

        tau_arr = real_array(tau, dtype)
        q_list = self.q_derivatives(tau_arr, max_order - 1, dtype) if max_order >= 1 else []
        one_complex = complex_array(np.ones_like(tau_arr, dtype=dtype), dtype)
        p_list: list[Array] = [one_complex]
        for n in range(0, max_order):
            p_next = complex_array(np.zeros_like(tau_arr, dtype=dtype), dtype)
            for k in range(0, n + 1):
                p_next = p_next + scalar(binom_int(n, k, dtype), dtype) * p_list[n - k] * q_list[k]
            p_list.append(p_next)
        return p_list

    def derivative(self, tau: Array, order: int, dtype: object = np.float64) -> Array:
        """Evaluate Z^(order)(tau) analytically through the quotient symbol."""

        if order < 0:
            raise ValueError("order must be nonnegative.")
        tau_arr = real_array(tau, dtype)
        p_list = self.derivative_symbols(tau_arr, order, dtype)
        return self.value(tau_arr, dtype) * p_list[order]

    def derivative_constant_slope_binomial(self, tau: Array, order: int, dtype: object = np.float64) -> Array:
        """Evaluate derivatives using the constant-envelope-slope binomial law."""

        if order < 0:
            raise ValueError("order must be nonnegative.")
        tau_arr = real_array(tau, dtype)
        denom = self.denominator(tau_arr, dtype)
        if self.alpha_coeffs.size > 2:
            higher = self.alpha_coeffs[2:]
            if np.max(np.abs(higher)) > scalar(0.0, dtype):
                raise ValueError("Binomial law requires constant envelope slope.")
        r_value = scalar(0.0, dtype)
        if self.alpha_coeffs.size >= 2:
            r_value = scalar(self.alpha_coeffs[1], dtype)
        lam = complex_array(r_value + scalar(1j, np.complex128) * scalar(self.omega, dtype), dtype)
        symbol = complex_array(np.zeros_like(tau_arr, dtype=dtype), dtype)
        for k in range(0, order + 1):
            coefficient = binom_int(order, k, dtype) * falling_factorial(self.m, k, dtype)
            symbol = symbol + scalar(coefficient, dtype) * np.power(denom, scalar(-k, dtype), dtype=dtype) * np.power(lam, order - k)
        return self.value(tau_arr, dtype) * symbol

    def dimensionless_q_derivatives(self, u: Array, max_order: int, dtype: object = np.float64) -> list[Array]:
        """Return q_u^(j)(u) for the rescaled function u -> Z(Tu)."""

        if self.shifted_delta is not None:
            raise ValueError("This dimensionless helper is for the unshifted positive-coordinate package.")
        u_arr = real_array(u, dtype)
        if np.any(u_arr <= scalar(0.0, dtype)):
            raise ValueError("dimensionless coordinate u must be positive.")
        T_value = scalar(self.T, dtype)
        m_value = scalar(self.m, dtype)
        tau_arr = T_value * u_arr
        q_list: list[Array] = []
        alpha_first = polynomial_derivative_value(self.alpha_coeffs, tau_arr, 1, dtype)
        q0 = m_value / u_arr + T_value * alpha_first + scalar(1j, np.complex128) * complex_array(T_value * scalar(self.omega, dtype), dtype)
        q_list.append(complex_array(q0, dtype))
        for j in range(1, max_order + 1):
            sign = scalar(1.0, dtype) if j % 2 == 0 else scalar(-1.0, dtype)
            scale_part = sign * factorial_int(j, dtype) * m_value / np.power(u_arr, scalar(j + 1, dtype), dtype=dtype)
            alpha_part = np.power(T_value, scalar(j + 1, dtype), dtype=dtype) * polynomial_derivative_value(self.alpha_coeffs, tau_arr, j + 1, dtype)
            q_list.append(complex_array(scale_part + alpha_part, dtype))
        return q_list

    def dimensionless_derivative_symbols(self, u: Array, max_order: int, dtype: object = np.float64) -> list[Array]:
        """Return derivative symbols for the rescaled function u -> Z(Tu)."""

        u_arr = real_array(u, dtype)
        q_list = self.dimensionless_q_derivatives(u_arr, max_order - 1, dtype) if max_order >= 1 else []
        p_list: list[Array] = [complex_array(np.ones_like(u_arr, dtype=dtype), dtype)]
        for n in range(0, max_order):
            p_next = complex_array(np.zeros_like(u_arr, dtype=dtype), dtype)
            for k in range(0, n + 1):
                p_next = p_next + scalar(binom_int(n, k, dtype), dtype) * p_list[n - k] * q_list[k]
            p_list.append(p_next)
        return p_list

    def imaginary_axis_symbols_from_tau(self, tau: Array, max_order: int, dtype: object = np.float64) -> list[Array]:
        """Return contour symbols mathcal P_n at zeta=i tau by equivalence."""

        p_tau = self.derivative_symbols(tau, max_order, dtype)
        p_zeta: list[Array] = []
        minus_i = complex_array(-scalar(1j, np.complex128), dtype)
        for n, symbol in enumerate(p_tau):
            p_zeta.append(np.power(minus_i, n) * symbol)
        return p_zeta


@dataclass
class RealSignalModel:
    """Finite real-channel aggregation x(tau)=a0+Re sum Gamma_i Z_i(tau)."""

    intercept: object
    nodes: list[OscillatoryEnvelopeNode]
    coefficients: Array
    label: str = "real_signal"

    def active_nodes_and_coeffs(self, dtype: object = np.float64) -> tuple[list[OscillatoryEnvelopeNode], Array]:
        """Remove zero-amplitude or zero-coefficient nodes before evaluation."""

        active_nodes: list[OscillatoryEnvelopeNode] = []
        active_coeffs: list[complex] = []
        coeffs = complex_array(self.coefficients, dtype)
        for node, coeff in zip(self.nodes, coeffs):
            if node.active(dtype) and np.abs(coeff) > scalar(0.0, dtype):
                active_nodes.append(node)
                active_coeffs.append(complex(coeff))
        return active_nodes, complex_array(active_coeffs, dtype)

    def value(self, tau: Array, dtype: object = np.float64) -> Array:
        """Evaluate the real signal."""

        tau_arr = real_array(tau, dtype)
        value = np.zeros_like(tau_arr, dtype=dtype) + scalar(self.intercept, dtype)
        active_nodes, active_coeffs = self.active_nodes_and_coeffs(dtype)
        for node, coeff in zip(active_nodes, active_coeffs):
            value = value + np.real(coeff * node.value(tau_arr, dtype)).astype(dtype)
        return value

    def derivative(self, tau: Array, order: int, dtype: object = np.float64) -> Array:
        """Evaluate the real-channel derivative x^(order)(tau)."""

        tau_arr = real_array(tau, dtype)
        if order == 0:
            return self.value(tau_arr, dtype)
        value = np.zeros_like(tau_arr, dtype=dtype) + scalar(0.0, dtype)
        active_nodes, active_coeffs = self.active_nodes_and_coeffs(dtype)
        for node, coeff in zip(active_nodes, active_coeffs):
            value = value + np.real(coeff * node.derivative(tau_arr, order, dtype)).astype(dtype)
        return value

    def derivative_block_table(self, tau: Array, dtype: object = np.float64) -> Array:
        """Return columns tau, x, direction, curvature, curvature_momentum."""

        tau_arr = real_array(tau, dtype)
        x0 = self.derivative(tau_arr, 0, dtype)
        x1 = self.derivative(tau_arr, 1, dtype)
        x2 = self.derivative(tau_arr, 2, dtype)
        x3 = self.derivative(tau_arr, 3, dtype)
        return np.column_stack((tau_arr, x0, x1, x2, x3))


def real_design_matrix(tau: Array, nodes: Sequence[OscillatoryEnvelopeNode], include_intercept: bool = True, dtype: object = np.float64) -> Array:
    """Build the real linear least-squares matrix for Re(Gamma_i Z_i)."""

    tau_arr = real_array(tau, dtype)
    columns: list[Array] = []
    if include_intercept:
        columns.append(np.ones_like(tau_arr, dtype=dtype))
    for node in nodes:
        z = node.value(tau_arr, dtype)
        columns.append(np.real(z).astype(dtype))
        columns.append((-np.imag(z)).astype(dtype))
    return np.column_stack(columns)


def build_model_from_lstsq(beta: Array, nodes: Sequence[OscillatoryEnvelopeNode], dtype: object = np.float64) -> RealSignalModel:
    """Convert least-squares coefficients into a RealSignalModel."""

    beta_arr = real_array(beta, dtype)
    intercept = scalar(beta_arr[0], dtype)
    coeffs: list[complex] = []
    index = 1
    for _ in nodes:
        real_part = scalar(beta_arr[index], dtype)
        imag_part = scalar(beta_arr[index + 1], dtype)
        coeffs.append(complex(real_part, imag_part))
        index += 2
    return RealSignalModel(intercept, list(nodes), complex_array(coeffs, dtype), label="least_squares_fit")


def real_design_matrix_with_metadata(
    tau: Array,
    nodes: Sequence[OscillatoryEnvelopeNode],
    include_intercept: bool = True,
    dtype: object = np.float64,
) -> tuple[Array, list[tuple[str, int]]]:
    """Build a design matrix and drop numerically inactive columns.

    The real-channel identity is Re((a+ib)Z)=a Re(Z)-b Im(Z).
    Purely real nonoscillatory nodes therefore have no useful imaginary column.
    Dropping inactive columns avoids artificial rank deficiency in the linear
    coefficient-estimation subproblem.
    """

    tau_arr = real_array(tau, dtype)
    columns: list[Array] = []
    metadata: list[tuple[str, int]] = []
    tol = scalar(100.0, dtype) * scalar(np.finfo(dtype).eps, dtype) * np.sqrt(scalar(tau_arr.size, dtype), dtype=dtype)
    if include_intercept:
        columns.append(np.ones_like(tau_arr, dtype=dtype))
        metadata.append(("intercept", -1))
    for node_index, node in enumerate(nodes):
        z = node.value(tau_arr, dtype)
        real_col = np.real(z).astype(dtype)
        imag_col = (-np.imag(z)).astype(dtype)
        if np.linalg.norm(real_col.astype(np.float64)) > float(tol):
            columns.append(real_col)
            metadata.append(("node_real", node_index))
        if np.linalg.norm(imag_col.astype(np.float64)) > float(tol):
            columns.append(imag_col)
            metadata.append(("node_imag", node_index))
    return np.column_stack(columns), metadata


def build_model_from_lstsq_metadata(
    beta: Array,
    nodes: Sequence[OscillatoryEnvelopeNode],
    metadata: Sequence[tuple[str, int]],
    dtype: object = np.float64,
) -> RealSignalModel:
    """Convert pruned least-squares coefficients into a RealSignalModel."""

    beta_arr = real_array(beta, dtype)
    intercept = scalar(0.0, dtype)
    coeffs = np.zeros(len(nodes), dtype=np.complex128 if dtype == np.float64 else np.complex64)
    for value, item in zip(beta_arr, metadata):
        kind, node_index = item
        if kind == "intercept":
            intercept = scalar(value, dtype)
        elif kind == "node_real":
            coeffs[node_index] = coeffs[node_index] + complex(float(value), 0.0)
        elif kind == "node_imag":
            coeffs[node_index] = coeffs[node_index] + complex(0.0, float(value))
    return RealSignalModel(intercept, list(nodes), complex_array(coeffs, dtype), label="least_squares_fit")


def ridge_lstsq(A: Array, y: Array, ridge: object = np.float64(0.0), dtype: object = np.float64) -> tuple[Array, object]:
    """Solve a linear least-squares problem with optional ridge stabilization."""

    A_arr = real_array(A, dtype)
    y_arr = real_array(y, dtype)
    ridge_value = scalar(ridge, dtype)
    if ridge_value > scalar(0.0, dtype):
        cols = A_arr.shape[1]
        augmented_A = np.vstack((A_arr, np.sqrt(ridge_value, dtype=dtype) * np.eye(cols, dtype=dtype)))
        augmented_y = np.concatenate((y_arr, np.zeros(cols, dtype=dtype)))
        beta, _, _, _ = np.linalg.lstsq(augmented_A, augmented_y, rcond=None)
    else:
        beta, _, _, _ = np.linalg.lstsq(A_arr, y_arr, rcond=None)
    singular_values = np.linalg.svd(A_arr.astype(np.float64), compute_uv=False)
    min_sv = np.min(singular_values)
    if min_sv <= np.finfo(np.float64).tiny:
        cond = np.asarray(np.inf, dtype=np.float64)
    else:
        cond = np.max(singular_values) / min_sv
    return real_array(beta, dtype), scalar(cond, np.float64)


def finite_difference_derivative(f_callable: object, tau: Array, order: int, h: object, dtype: object = np.float64) -> Array:
    """Centered finite-difference derivative used only for validation."""

    tau_arr = real_array(tau, dtype)
    h_value = scalar(h, dtype)
    if order == 1:
        return (f_callable(tau_arr - scalar(2.0, dtype) * h_value) - scalar(8.0, dtype) * f_callable(tau_arr - h_value) + scalar(8.0, dtype) * f_callable(tau_arr + h_value) - f_callable(tau_arr + scalar(2.0, dtype) * h_value)) / (scalar(12.0, dtype) * h_value)
    if order == 2:
        return (-f_callable(tau_arr - scalar(2.0, dtype) * h_value) + scalar(16.0, dtype) * f_callable(tau_arr - h_value) - scalar(30.0, dtype) * f_callable(tau_arr) + scalar(16.0, dtype) * f_callable(tau_arr + h_value) - f_callable(tau_arr + scalar(2.0, dtype) * h_value)) / (scalar(12.0, dtype) * h_value * h_value)
    if order == 3:
        return (f_callable(tau_arr + scalar(2.0, dtype) * h_value) - scalar(2.0, dtype) * f_callable(tau_arr + h_value) + scalar(2.0, dtype) * f_callable(tau_arr - h_value) - f_callable(tau_arr - scalar(2.0, dtype) * h_value)) / (scalar(2.0, dtype) * h_value * h_value * h_value)
    raise ValueError("finite_difference_derivative supports orders 1, 2, and 3.")


def local_turning_candidates(block_table: Array, dtype: object = np.float64) -> Array:
    """Select pointwise candidate turning rows using derivative blocks.

    This low-level diagnostic is intentionally pointwise. For reporting and
    model validation, use clustered_turning_regions below to avoid treating
    adjacent grid points as independent events.
    """

    table = real_array(block_table, dtype)
    tau = table[:, 0]
    direction = table[:, 2]
    curvature = table[:, 3]
    curvature_momentum = table[:, 4]
    abs_direction = np.abs(direction)
    direction_threshold = np.quantile(abs_direction, scalar(0.15, dtype))
    curvature_threshold = np.quantile(np.abs(curvature), scalar(0.70, dtype))
    mask = (abs_direction <= direction_threshold) & (np.abs(curvature) >= curvature_threshold)
    rows: list[list[object]] = []
    for i in range(1, tau.size - 1):
        sign_change = bool(direction[i - 1] * direction[i + 1] <= scalar(0.0, dtype))
        if bool(mask[i]) or sign_change:
            label = "min_candidate" if curvature[i] > scalar(0.0, dtype) else "max_candidate"
            rows.append([tau[i], direction[i], curvature[i], curvature_momentum[i], sign_change, label])
    if len(rows) == 0:
        return np.empty((0, 6), dtype=object)
    return np.asarray(rows, dtype=object)


def clustered_turning_regions(block_table: Array, dtype: object = np.float64) -> Array:
    """Cluster adjacent candidate rows into turning-region intervals.

    Pointwise sign changes and threshold hits frequently appear on adjacent
    grid rows. Treating them as separate events exaggerates the number of
    turns. This routine reports one interval per contiguous block and records
    the strongest representative point in that block.
    """

    table = real_array(block_table, dtype)
    tau = table[:, 0]
    direction = table[:, 2]
    curvature = table[:, 3]
    curvature_momentum = table[:, 4]
    abs_direction = np.abs(direction)
    direction_scale = np.maximum(np.quantile(abs_direction, scalar(0.75, dtype)), scalar(100.0 * np.finfo(dtype).eps, dtype))
    curvature_scale = np.maximum(np.quantile(np.abs(curvature), scalar(0.75, dtype)), scalar(100.0 * np.finfo(dtype).eps, dtype))
    momentum_scale = np.maximum(np.quantile(np.abs(curvature_momentum), scalar(0.75, dtype)), scalar(100.0 * np.finfo(dtype).eps, dtype))

    direction_threshold = np.quantile(abs_direction, scalar(0.15, dtype))
    curvature_threshold = np.quantile(np.abs(curvature), scalar(0.70, dtype))
    sign_change_mask = np.zeros(tau.size, dtype=bool)
    sign_change_mask[1:-1] = direction[:-2] * direction[2:] <= scalar(0.0, dtype)
    magnitude_mask = (abs_direction <= direction_threshold) & (np.abs(curvature) >= curvature_threshold)
    mask = sign_change_mask | magnitude_mask

    rows: list[list[object]] = []
    i = 1
    while i < tau.size - 1:
        if not bool(mask[i]):
            i += 1
            continue
        start = i
        while i + 1 < tau.size - 1 and bool(mask[i + 1]):
            i += 1
        end = i
        block = np.arange(start, end + 1)
        score = (
            (scalar(1.0, dtype) - np.minimum(abs_direction[block] / direction_scale, scalar(1.0, dtype)))
            + np.minimum(np.abs(curvature[block]) / curvature_scale, scalar(3.0, dtype))
            + scalar(0.25, dtype) * np.minimum(np.abs(curvature_momentum[block]) / momentum_scale, scalar(3.0, dtype))
        )
        rep_local = int(np.argmax(score))
        rep = int(block[rep_local])
        label = "min_region" if curvature[rep] > scalar(0.0, dtype) else "max_region"
        rows.append([
            float(tau[start]),
            float(tau[end]),
            float(tau[rep]),
            float(direction[rep]),
            float(curvature[rep]),
            float(curvature_momentum[rep]),
            float(score[rep_local]),
            int(end - start + 1),
            bool(np.any(sign_change_mask[block])),
            label,
        ])
        i += 1
    if len(rows) == 0:
        return np.empty((0, 10), dtype=object)
    return np.asarray(rows, dtype=object)


def real_derivative_design_matrix_from_metadata(
    tau: Array,
    nodes: Sequence[OscillatoryEnvelopeNode],
    metadata: Sequence[tuple[str, int]],
    order: int,
    dtype: object = np.float64,
) -> Array:
    """Build derivative design matrix using metadata from the level matrix."""

    tau_arr = real_array(tau, dtype)
    columns: list[Array] = []
    for kind, node_index in metadata:
        if kind == "intercept":
            if order == 0:
                columns.append(np.ones_like(tau_arr, dtype=dtype))
            else:
                columns.append(np.zeros_like(tau_arr, dtype=dtype))
        else:
            z_der = nodes[node_index].derivative(tau_arr, order, dtype)
            if kind == "node_real":
                columns.append(np.real(z_der).astype(dtype))
            elif kind == "node_imag":
                columns.append((-np.imag(z_der)).astype(dtype))
            else:
                raise ValueError(f"Unknown metadata kind: {kind}")
    return np.column_stack(columns)


def scaled_condition_number(A: Array) -> object:
    """Return the 2-norm condition number of a matrix after conversion to float64."""

    singular_values = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    min_sv = np.min(singular_values)
    if min_sv <= np.finfo(np.float64).tiny:
        return np.asarray(np.inf, dtype=np.float64)
    return np.asarray(np.max(singular_values) / min_sv, dtype=np.float64)


def derivative_aware_lstsq(
    tau: Array,
    y: Array,
    nodes: Sequence[OscillatoryEnvelopeNode],
    derivative_targets: dict[int, Array],
    derivative_weights: dict[int, object],
    ridge: object = np.float64(1.0e-10),
    dtype: object = np.float64,
) -> tuple[RealSignalModel, object, dict[str, object]]:
    """Solve a scaled stacked least-squares problem with derivative targets.

    The derivative targets must come from a smooth reference, a synthetic
    benchmark, or validated model-implied derivatives. They should not be raw
    high-order finite differences of noisy observations.
    """

    tau_arr = real_array(tau, dtype)
    y_arr = real_array(y, dtype)
    A0, metadata = real_design_matrix_with_metadata(tau_arr, nodes, True, dtype)
    y_scale = np.maximum(np.sqrt(np.mean(y_arr * y_arr, dtype=dtype), dtype=dtype), scalar(100.0 * np.finfo(dtype).eps, dtype))
    stacked_A = [A0 / y_scale]
    stacked_y = [y_arr / y_scale]
    info: dict[str, object] = {"level_scale": float(y_scale), "num_columns": int(A0.shape[1])}

    for order in sorted(derivative_targets):
        target = real_array(derivative_targets[order], dtype)
        weight = scalar(derivative_weights.get(order, scalar(0.0, dtype)), dtype)
        if weight <= scalar(0.0, dtype):
            continue
        scale = np.maximum(np.sqrt(np.mean(target * target, dtype=dtype), dtype=dtype), scalar(100.0 * np.finfo(dtype).eps, dtype))
        A_order = real_derivative_design_matrix_from_metadata(tau_arr, nodes, metadata, order, dtype)
        row_factor = np.sqrt(weight, dtype=dtype) / scale
        stacked_A.append(row_factor * A_order)
        stacked_y.append(row_factor * target)
        info[f"x{order}_scale"] = float(scale)
        info[f"x{order}_weight"] = float(weight)

    A_stack = np.vstack(stacked_A)
    y_stack = np.concatenate(stacked_y)
    beta, cond = ridge_lstsq(A_stack, y_stack, ridge, dtype)
    info["stacked_condition_number"] = float(cond)
    info["unscaled_level_condition_number"] = float(scaled_condition_number(A0))
    model = build_model_from_lstsq_metadata(beta, nodes, metadata, dtype)
    return model, cond, info


def rmse(a: Array, b: Array, dtype: object = np.float64) -> object:
    """Compute root-mean-squared error."""

    diff = real_array(a, dtype) - real_array(b, dtype)
    return scalar(np.sqrt(np.mean(diff * diff, dtype=dtype), dtype=dtype), dtype)


def run_numerical_verification(result_dir: Path, figure_dir: Path, dtype: object = np.float64) -> list[str]:
    """Run identity and derivative checks for a single node."""

    messages: list[str] = []
    T = scalar(1.0, dtype)
    tau = np.linspace(scalar(0.12, dtype), scalar(0.95, dtype), 80, dtype=dtype)
    alpha_coeffs = real_array([0.03, -0.18], dtype)
    node = OscillatoryEnvelopeNode(np.float64(1.7), np.float64(2.0), np.float64(8.0), np.float64(0.4), alpha_coeffs, T=T, label="constant_slope_node")

    rows: list[list[object]] = []
    for order in range(0, 7):
        analytic = node.derivative(tau, order, dtype)
        binomial = node.derivative_constant_slope_binomial(tau, order, dtype)
        error = safe_relative_error(analytic, binomial, dtype)
        tolerance = scalar(5.0e-11, dtype) if dtype == np.float64 else scalar(5.0e-4, dtype)
        status = "PASS" if error <= tolerance else "CHECK"
        rows.append([order, float(error), float(tolerance), status])
    write_csv(result_dir / "identity_verification_recursion_vs_binomial.csv", ["order", "max_relative_error", "tolerance", "status"], rows)
    messages.append("Recursion-versus-binomial verification written to identity_verification_recursion_vs_binomial.csv.")

    safe_tau = np.linspace(scalar(0.25, dtype), scalar(0.80, dtype), 35, dtype=dtype)
    h = scalar(1.0e-4, dtype) if dtype == np.float64 else scalar(2.5e-3, dtype)
    fd_rows: list[list[object]] = []
    for order in [1, 2, 3]:
        analytic_real = np.real(node.derivative(safe_tau, order, dtype)).astype(dtype)
        fd_real = finite_difference_derivative(lambda x: np.real(node.value(x, dtype)).astype(dtype), safe_tau, order, h, dtype)
        error = safe_relative_error(fd_real, analytic_real, dtype)
        fd_rows.append([order, float(h), float(error)])
    write_csv(result_dir / "finite_difference_validation.csv", ["order", "step_h", "max_relative_error"], fd_rows)
    messages.append("Finite-difference validation written to finite_difference_validation.csv.")

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(tau, np.real(node.value(tau, dtype)), label="Re Z")
    plt.plot(tau, np.real(node.derivative(tau, 1, dtype)), label="Re Z prime")
    plt.plot(tau, np.real(node.derivative(tau, 2, dtype)), label="Re Z double prime")
    plt.plot(tau, np.real(node.derivative(tau, 3, dtype)), label="Re Z triple prime")
    plt.xlabel("tau")
    plt.ylabel("real-channel value")
    plt.title("FDC-LT node and analytic derivative blocks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "single_node_derivative_blocks.png", dpi=180)
    plt.close()

    return messages


def run_finite_difference_h_sweep(result_dir: Path, figure_dir: Path, dtype: object = np.float64) -> list[str]:
    """Show finite-difference step-size sensitivity against analytic derivatives."""

    messages: list[str] = []
    tau = np.linspace(scalar(0.30, dtype), scalar(0.70, dtype), 40, dtype=dtype)
    node = OscillatoryEnvelopeNode(np.float64(1.2), np.float64(1.0), np.float64(11.0), np.float64(-0.2), real_array([0.0, 0.08, -0.04], dtype), T=np.float64(1.0), label="quadratic_envelope_node")
    h_values = np.logspace(-7.0, -1.5, 18, dtype=dtype) if dtype == np.float64 else np.logspace(-4.5, -1.2, 14, dtype=dtype)
    rows: list[list[object]] = []
    plot_data: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for h in h_values:
        for order in [1, 2, 3]:
            analytic = np.real(node.derivative(tau, order, dtype)).astype(dtype)
            fd = finite_difference_derivative(lambda x: np.real(node.value(x, dtype)).astype(dtype), tau, order, h, dtype)
            error = safe_relative_error(fd, analytic, dtype)
            rows.append([order, float(h), float(error)])
            plot_data[order].append(float(error))
    write_csv(result_dir / "finite_difference_h_sweep.csv", ["order", "step_h", "max_relative_error"], rows)

    best_rows: list[list[object]] = []
    for order in [1, 2, 3]:
        order_rows = [row for row in rows if int(row[0]) == order]
        best = min(order_rows, key=lambda row: float(row[2]))
        best_rows.append([order, best[1], best[2]])
    write_csv(result_dir / "finite_difference_best_steps.csv", ["order", "best_step_h", "minimum_max_relative_error"], best_rows)

    plt.figure(figsize=(8.0, 5.0))
    for order in [1, 2, 3]:
        plt.loglog(h_values, plot_data[order], marker="o", label=f"order {order}")
    plt.xlabel("finite-difference step h")
    plt.ylabel("max relative error")
    plt.title("Finite-difference truncation and roundoff tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "finite_difference_h_sweep.png", dpi=180)
    plt.close()
    messages.append("Finite-difference h-sweep and best-step summary written to CSV/PNG outputs.")
    return messages


def run_shifted_fractional_example(result_dir: Path, figure_dir: Path, dtype: object = np.float64) -> list[str]:
    """Demonstrate the shifted package for fractional m at the origin."""

    messages: list[str] = []
    tau = np.linspace(scalar(0.0, dtype), scalar(1.0, dtype), 120, dtype=dtype)
    shifted = OscillatoryEnvelopeNode(np.float64(0.9), np.float64(0.5), np.float64(7.0), np.float64(0.1), real_array([0.0, -0.05], dtype), T=np.float64(1.0), shifted_delta=np.float64(0.04), label="shifted_fractional_node")
    p_symbols = shifted.derivative_symbols(tau, 3, dtype)
    rows: list[list[object]] = []
    for index in [0, 1, 2, 5, 20, 60, 119]:
        rows.append([
            float(tau[index]),
            float(np.real(shifted.value(tau[index:index + 1], dtype))[0]),
            float(np.abs(p_symbols[1][index])),
            float(np.abs(p_symbols[2][index])),
            float(np.abs(p_symbols[3][index])),
        ])
    write_csv(result_dir / "shifted_fractional_origin_example.csv", ["tau", "real_value", "abs_P1", "abs_P2", "abs_P3"], rows)

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(tau, np.real(shifted.value(tau, dtype)), label="Re shifted node")
    plt.plot(tau, np.real(shifted.derivative(tau, 1, dtype)), label="first derivative")
    plt.plot(tau, np.real(shifted.derivative(tau, 2, dtype)), label="second derivative")
    plt.xlabel("tau")
    plt.ylabel("real-channel value")
    plt.title("Shifted fractional-power node, evaluated at tau=0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "shifted_fractional_origin_example.png", dpi=180)
    plt.close()
    messages.append("Shifted fractional-origin example written to shifted_fractional_origin_example.csv and shifted_fractional_origin_example.png.")
    return messages


def run_shifted_origin_growth_diagnostics(result_dir: Path, dtype: object = np.float64) -> list[str]:
    """Monitor derivative-symbol growth near the shifted origin for several deltas."""

    messages: list[str] = []
    tau = np.linspace(scalar(0.0, dtype), scalar(0.05, dtype), 60, dtype=dtype)
    rows: list[list[object]] = []
    for delta in [0.01, 0.02, 0.04, 0.08, 0.16]:
        node = OscillatoryEnvelopeNode(
            np.float64(0.9), np.float64(0.5), np.float64(7.0), np.float64(0.1),
            real_array([0.0, -0.05], dtype), T=np.float64(1.0), shifted_delta=np.float64(delta),
            label=f"origin_growth_delta_{delta}"
        )
        p_symbols = node.derivative_symbols(tau, 4, dtype)
        rows.append([
            float(delta),
            float(np.max(np.abs(p_symbols[1]))),
            float(np.max(np.abs(p_symbols[2]))),
            float(np.max(np.abs(p_symbols[3]))),
            float(np.max(np.abs(p_symbols[4]))),
        ])
    write_csv(result_dir / "shifted_origin_growth_diagnostics.csv", [
        "shift_delta", "max_abs_P1_near_origin", "max_abs_P2_near_origin",
        "max_abs_P3_near_origin", "max_abs_P4_near_origin"
    ], rows)
    messages.append("Shifted-origin growth diagnostics written to shifted_origin_growth_diagnostics.csv.")
    return messages


def run_dimensionless_and_imaginary_checks(result_dir: Path, dtype: object = np.float64) -> list[str]:
    """Verify dimensionless and imaginary-axis equivalences."""

    messages: list[str] = []
    T = scalar(2.5, dtype)
    u = np.linspace(scalar(0.15, dtype), scalar(0.90, dtype), 70, dtype=dtype)
    tau = T * u
    node = OscillatoryEnvelopeNode(np.float64(1.1), np.float64(2.0), np.float64(5.0), np.float64(0.3), real_array([0.01, -0.03, 0.02], dtype), T=T, label="dimensionless_node")

    p_tau = node.derivative_symbols(tau, 4, dtype)
    p_u = node.dimensionless_derivative_symbols(u, 4, dtype)
    dim_rows: list[list[object]] = []
    for order in range(0, 5):
        expected = np.power(T, scalar(order, dtype), dtype=dtype) * p_tau[order]
        error = safe_relative_error(p_u[order], expected, dtype)
        dim_rows.append([order, float(error)])
    write_csv(result_dir / "dimensionless_equivalence.csv", ["order", "max_relative_error_vs_Tn_Ptau"], dim_rows)

    p_zeta = node.imaginary_axis_symbols_from_tau(tau, 4, dtype)
    imag_rows: list[list[object]] = []
    minus_i = complex_array(-scalar(1j, np.complex128), dtype)
    for order in range(0, 5):
        expected = np.power(minus_i, order) * p_tau[order]
        error = safe_relative_error(p_zeta[order], expected, dtype)
        imag_rows.append([order, float(error)])
    write_csv(result_dir / "imaginary_axis_equivalence.csv", ["order", "max_relative_error"], imag_rows)
    messages.append("Dimensionless and imaginary-axis equivalence checks written to CSV files.")
    return messages


def build_representative_true_model(dtype: object = np.float64) -> RealSignalModel:
    """Construct a representative sparse real-channel model."""

    nodes = [
        OscillatoryEnvelopeNode(np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), real_array([0.0, 0.08, -0.03], dtype), T=np.float64(1.0), label="base_level"),
        OscillatoryEnvelopeNode(np.float64(1.0), np.float64(1.0), np.float64(6.0), np.float64(0.2), real_array([0.0, -0.04], dtype), T=np.float64(1.0), label="slow_turning_wave"),
        OscillatoryEnvelopeNode(np.float64(1.0), np.float64(2.0), np.float64(13.0), np.float64(-0.4), real_array([0.0, 0.02], dtype), T=np.float64(1.0), label="curvature_perturbation"),
    ]
    coeffs = complex_array([1.10 + 0.00j, 0.38 - 0.16j, -0.09 + 0.07j], dtype)
    return RealSignalModel(np.float64(0.15), nodes, coeffs, label="representative_true_model")


def run_real_signal_examples(result_dir: Path, figure_dir: Path, dtype: object = np.float64) -> list[str]:
    """Demonstrate real-channel derivative blocks and candidate turning regions."""

    messages: list[str] = []
    model = build_representative_true_model(dtype)
    tau = np.linspace(scalar(0.03, dtype), scalar(1.0, dtype), 220, dtype=dtype)
    table = model.derivative_block_table(tau, dtype)
    rows = [[float(value) for value in row] for row in table]
    write_csv(result_dir / "real_signal_derivative_blocks.csv", ["tau", "x", "direction_x1", "curvature_x2", "curvature_momentum_x3"], rows)

    candidates = local_turning_candidates(table, dtype)
    candidate_rows = [[item for item in row] for row in candidates]
    write_csv(result_dir / "turning_region_candidates.csv", ["tau", "direction_x1", "curvature_x2", "curvature_momentum_x3", "direction_sign_change", "candidate_type"], candidate_rows)

    clustered = clustered_turning_regions(table, dtype)
    clustered_rows = [[item for item in row] for row in clustered]
    write_csv(result_dir / "clustered_turning_regions.csv", [
        "tau_start", "tau_end", "tau_representative", "direction_x1", "curvature_x2",
        "curvature_momentum_x3", "region_score", "num_grid_points", "contains_sign_change", "candidate_type"
    ], clustered_rows)

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(table[:, 0], table[:, 1], label="x")
    plt.plot(table[:, 0], table[:, 2], label="x prime")
    plt.plot(table[:, 0], table[:, 3], label="x double prime")
    plt.plot(table[:, 0], table[:, 4], label="x triple prime")
    if candidates.size > 0:
        plt.scatter(candidates[:, 0].astype(dtype), np.interp(candidates[:, 0].astype(dtype), table[:, 0], table[:, 1]), marker="x", label="candidate regions")
    plt.xlabel("tau")
    plt.ylabel("value")
    plt.title("Real-channel direction, curvature, and curvature-momentum blocks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "real_signal_derivative_blocks.png", dpi=180)
    plt.close()

    s1 = scalar(np.quantile(np.abs(table[:, 2]), scalar(0.75, dtype)), dtype)
    s2 = scalar(np.quantile(np.abs(table[:, 3]), scalar(0.75, dtype)), dtype)
    s3 = scalar(np.quantile(np.abs(table[:, 4]), scalar(0.75, dtype)), dtype)
    eps = scalar(100.0, dtype) * scalar(np.finfo(dtype).eps, dtype)
    scores = np.column_stack((
        table[:, 0],
        table[:, 2] / np.maximum(s1, eps),
        table[:, 3] / np.maximum(s2, eps),
        table[:, 4] / np.maximum(s3, eps),
        np.abs(table[:, 2] / np.maximum(s1, eps)) + np.abs(table[:, 3] / np.maximum(s2, eps)) + np.abs(table[:, 4] / np.maximum(s3, eps)),
    ))
    score_rows = [[float(value) for value in row] for row in scores]
    write_csv(result_dir / "normalized_derivative_scores.csv", ["tau", "direction_score", "curvature_score", "curvature_momentum_score", "aggregate_abs_score"], score_rows)
    messages.append("Real-signal derivative blocks, normalized scores, pointwise candidates, and clustered turning regions written to CSV/PNG.")
    return messages


def run_fitting_demonstration(
    result_dir: Path,
    figure_dir: Path,
    dtype: object = np.float64,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[str]:
    """Run fixed-structure fitting, held-out diagnostics, and high-noise tests.

    The demonstration compares three fitted models:
    1. a smooth base-only fit;
    2. a level-only oscillatory-envelope fit;
    3. a derivative-aware oscillatory-envelope fit using smooth benchmark
       derivative targets.

    The high-noise experiment repeats the same fitting protocol over multiple
    observation-noise levels. The derivative-aware case remains an oracle
    benchmark because its derivative targets are the analytic derivatives of
    the known synthetic reference model. Therefore, this experiment tests
    numerical robustness under controlled noise, not real-market predictive
    ability.
    """

    messages: list[str] = []
    true_model = build_representative_true_model(dtype)
    tau_all = np.linspace(scalar(0.03, dtype), scalar(1.0, dtype), 180, dtype=dtype)
    y_true = true_model.value(tau_all, dtype)

    train_mask = tau_all <= scalar(0.72, dtype)
    hold_mask = ~train_mask
    tau_train = tau_all[train_mask]
    tau_hold = tau_all[hold_mask]
    y_true_train = y_true[train_mask]
    y_true_hold = y_true[hold_mask]

    base_nodes = [true_model.nodes[0]]
    osc_nodes = true_model.nodes
    derivative_targets = {
        1: true_model.derivative(tau_train, 1, dtype),
        2: true_model.derivative(tau_train, 2, dtype),
        3: true_model.derivative(tau_train, 3, dtype),
    }
    derivative_weights = {1: np.float64(0.20), 2: np.float64(0.08), 3: np.float64(0.02)}

    def fit_three_models(y_obs_current: Array) -> tuple[list[tuple[str, RealSignalModel, object]], dict[str, object]]:
        """Fit base, level-only, and derivative-aware models to one noisy sample."""

        y_train_current = real_array(y_obs_current[train_mask], dtype)

        A_base_train, base_meta = real_design_matrix_with_metadata(tau_train, base_nodes, True, dtype)
        beta_base, cond_base = ridge_lstsq(A_base_train, y_train_current, np.float64(1.0e-10), dtype)
        fit_base = build_model_from_lstsq_metadata(beta_base, base_nodes, base_meta, dtype)

        A_osc_train, osc_meta = real_design_matrix_with_metadata(tau_train, osc_nodes, True, dtype)
        beta_osc, cond_osc = ridge_lstsq(A_osc_train, y_train_current, np.float64(1.0e-10), dtype)
        fit_osc = build_model_from_lstsq_metadata(beta_osc, osc_nodes, osc_meta, dtype)

        fit_deriv, cond_deriv, deriv_info = derivative_aware_lstsq(
            tau_train, y_train_current, osc_nodes, derivative_targets, derivative_weights,
            ridge=np.float64(1.0e-10), dtype=dtype
        )

        fitted_models = [
            ("base_only", fit_base, cond_base),
            ("base_plus_oscillatory_nodes_level_only", fit_osc, cond_osc),
            ("base_plus_oscillatory_nodes_derivative_aware_oracle", fit_deriv, cond_deriv),
        ]
        return fitted_models, deriv_info

    def append_fit_summary_rows(
        rows: list[list[object]],
        noise_scale_value: object,
        y_obs_current: Array,
        fitted_models: list[tuple[str, RealSignalModel, object]],
    ) -> None:
        """Append train/holdout and derivative errors for one noise level."""

        y_train_current = real_array(y_obs_current[train_mask], dtype)
        y_hold_current = real_array(y_obs_current[hold_mask], dtype)
        for name, fit_model, cond in fitted_models:
            train_pred = fit_model.value(tau_train, dtype)
            hold_pred = fit_model.value(tau_hold, dtype)
            derivative_rmse_1 = rmse(fit_model.derivative(tau_hold, 1, dtype), true_model.derivative(tau_hold, 1, dtype), dtype)
            derivative_rmse_2 = rmse(fit_model.derivative(tau_hold, 2, dtype), true_model.derivative(tau_hold, 2, dtype), dtype)
            derivative_rmse_3 = rmse(fit_model.derivative(tau_hold, 3, dtype), true_model.derivative(tau_hold, 3, dtype), dtype)
            train_derivative_rmse_1 = rmse(fit_model.derivative(tau_train, 1, dtype), true_model.derivative(tau_train, 1, dtype), dtype)
            train_derivative_rmse_2 = rmse(fit_model.derivative(tau_train, 2, dtype), true_model.derivative(tau_train, 2, dtype), dtype)
            train_derivative_rmse_3 = rmse(fit_model.derivative(tau_train, 3, dtype), true_model.derivative(tau_train, 3, dtype), dtype)
            rows.append([
                float(noise_scale_value),
                name,
                float(cond),
                float(rmse(train_pred, y_train_current, dtype)),
                float(rmse(hold_pred, y_hold_current, dtype)),
                float(rmse(train_pred, y_true_train, dtype)),
                float(rmse(hold_pred, y_true_hold, dtype)),
                float(train_derivative_rmse_1),
                float(train_derivative_rmse_2),
                float(train_derivative_rmse_3),
                float(derivative_rmse_1),
                float(derivative_rmse_2),
                float(derivative_rmse_3),
            ])

    baseline_noise_scale = scalar(0.01, dtype)
    rng_baseline = np.random.default_rng(random_seed)
    baseline_noise = real_array(rng_baseline.normal(0.0, float(baseline_noise_scale), tau_all.size), dtype)
    y_obs = y_true + baseline_noise
    fitted_models, deriv_info = fit_three_models(y_obs)
    fit_base = fitted_models[0][1]
    fit_osc = fitted_models[1][1]
    fit_deriv = fitted_models[2][1]

    summary_rows: list[list[object]] = []
    append_fit_summary_rows(summary_rows, baseline_noise_scale, y_obs, fitted_models)
    write_csv(result_dir / "fitting_summary.csv", [
        "noise_scale", "model", "design_condition_number", "train_rmse_vs_noisy", "holdout_rmse_vs_noisy",
        "train_rmse_vs_true", "holdout_rmse_vs_true",
        "train_x1_rmse", "train_x2_rmse", "train_x3_rmse",
        "holdout_x1_rmse", "holdout_x2_rmse", "holdout_x3_rmse"
    ], summary_rows)

    derivative_config_rows = [[key, value] for key, value in deriv_info.items()]
    derivative_config_rows.append(["derivative_target_source", "analytic derivatives of known synthetic reference model"])
    derivative_config_rows.append(["baseline_noise_scale", float(baseline_noise_scale)])
    write_csv(result_dir / "derivative_aware_fitting_configuration.csv", ["quantity", "value"], derivative_config_rows)

    fitted_table = np.column_stack((
        tau_all,
        y_obs,
        y_true,
        fit_base.value(tau_all, dtype),
        fit_osc.value(tau_all, dtype),
        fit_deriv.value(tau_all, dtype),
        true_model.derivative(tau_all, 1, dtype),
        fit_osc.derivative(tau_all, 1, dtype),
        fit_deriv.derivative(tau_all, 1, dtype),
        true_model.derivative(tau_all, 2, dtype),
        fit_osc.derivative(tau_all, 2, dtype),
        fit_deriv.derivative(tau_all, 2, dtype),
        true_model.derivative(tau_all, 3, dtype),
        fit_osc.derivative(tau_all, 3, dtype),
        fit_deriv.derivative(tau_all, 3, dtype),
    ))
    write_csv(result_dir / "fitting_path_and_derivatives.csv", [
        "tau", "observed_noisy", "true_x", "base_fit_x", "oscillatory_level_fit_x", "oscillatory_derivative_aware_fit_x",
        "true_x1", "oscillatory_level_fit_x1", "oscillatory_derivative_aware_fit_x1",
        "true_x2", "oscillatory_level_fit_x2", "oscillatory_derivative_aware_fit_x2",
        "true_x3", "oscillatory_level_fit_x3", "oscillatory_derivative_aware_fit_x3"
    ], [[float(value) for value in row] for row in fitted_table])

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(tau_all, y_true, linestyle=(0, (3, 3)), color="black", zorder=20, label="true smooth signal")
    plt.scatter(tau_all, y_obs, s=12, color="lightblue", label="noisy observations")
    plt.plot(tau_all, fit_base.value(tau_all, dtype), label="base fit")
    plt.plot(tau_all, fit_osc.value(tau_all, dtype), label="level-only oscillatory fit")
    plt.plot(tau_all, fit_deriv.value(tau_all, dtype), label="derivative-aware oscillatory fit")
    plt.axvline(float(np.max(tau_train)), linestyle="--", color="purple", label="train/holdout split")
    plt.xlabel("tau")
    plt.ylabel("x")
    plt.title("Fixed-structure fitting with held-out window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fitting_train_holdout.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(tau_all, true_model.derivative(tau_all, 1, dtype), linestyle=(0, (3, 3)), color="black", zorder=20, label="true x prime")
    plt.plot(tau_all, fit_osc.derivative(tau_all, 1, dtype), label="level-only fit x prime")
    plt.plot(tau_all, fit_deriv.derivative(tau_all, 1, dtype), label="derivative-aware fit x prime")
    plt.plot(tau_all, true_model.derivative(tau_all, 2, dtype), linestyle=(0, (3, 3)), color="blue", zorder=20, label="true x double prime")
    plt.plot(tau_all, fit_osc.derivative(tau_all, 2, dtype), label="level-only fit x double prime")
    plt.plot(tau_all, fit_deriv.derivative(tau_all, 2, dtype), label="derivative-aware fit x double prime")
    plt.axvline(float(np.max(tau_train)), linestyle="--", color="purple", label="train/holdout split")
    plt.xlabel("tau")
    plt.ylabel("derivative value")
    plt.title("Model-implied derivative diagnostics from fitted representation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fitted_model_derivative_diagnostics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(tau_all, true_model.derivative(tau_all, 3, dtype), linestyle=(0, (3, 3)), color="green", zorder=20, label="true x triple prime")
    plt.plot(tau_all, fit_osc.derivative(tau_all, 3, dtype), label="level-only fit x triple prime")
    plt.plot(tau_all, fit_deriv.derivative(tau_all, 3, dtype), label="derivative-aware fit x triple prime")
    plt.axvline(float(np.max(tau_train)), linestyle="--", color="purple", label="train/holdout split")
    plt.xlabel("tau")
    plt.ylabel("third derivative value")
    plt.title("Third-derivative diagnostic: level-only versus derivative-aware fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "third_derivative_diagnostic.png", dpi=180)
    plt.close()

    high_noise_scales = real_array([0.01, 0.03, 0.06, 0.10, 0.15], dtype)
    high_noise_rows: list[list[object]] = []
    high_noise_samples: dict[float, tuple[Array, list[tuple[str, RealSignalModel, object]]]] = {}
    for noise_scale_value in high_noise_scales:
        seed_shift = int(np.round(float(noise_scale_value) * 100000.0))
        rng = np.random.default_rng(random_seed + seed_shift)
        noise = real_array(rng.normal(0.0, float(noise_scale_value), tau_all.size), dtype)
        y_obs_current = y_true + noise
        models_current, _ = fit_three_models(y_obs_current)
        append_fit_summary_rows(high_noise_rows, noise_scale_value, y_obs_current, models_current)
        high_noise_samples[float(noise_scale_value)] = (y_obs_current, models_current)

    write_csv(result_dir / "high_noise_fitting_summary.csv", [
        "noise_scale", "model", "design_condition_number", "train_rmse_vs_noisy", "holdout_rmse_vs_noisy",
        "train_rmse_vs_true", "holdout_rmse_vs_true",
        "train_x1_rmse", "train_x2_rmse", "train_x3_rmse",
        "holdout_x1_rmse", "holdout_x2_rmse", "holdout_x3_rmse"
    ], high_noise_rows)

    high_noise_notes = [
        ["noise_scales", ";".join(str(float(value)) for value in high_noise_scales)],
        ["noise_generation", "independent Gaussian noise added to the known smooth signal"],
        ["noise_scale_interpretation", "standard deviation of additive observation noise"],
        ["derivative_aware_fit", "oracle benchmark using analytic true-derivative targets"],
        ["validation_warning", "high-noise synthetic robustness does not imply real-market predictive performance"],
    ]
    write_csv(result_dir / "high_noise_experiment_notes.csv", ["quantity", "value"], high_noise_notes)

    model_names = [
        "base_only",
        "base_plus_oscillatory_nodes_level_only",
        "base_plus_oscillatory_nodes_derivative_aware_oracle",
    ]
    noise_values = [float(value) for value in high_noise_scales]
    plt.figure(figsize=(8.5, 5.2))
    for model_name in model_names:
        values = [float(row[6]) for row in high_noise_rows if row[1] == model_name]
        plt.plot(noise_values, values, marker="o", label=model_name)
    plt.xlabel("observation-noise scale")
    plt.ylabel("holdout RMSE vs true smooth signal")
    plt.title("High-noise stress test: level fit versus true signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "high_noise_holdout_true_rmse_by_noise.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for model_name in model_names:
        values = [float(row[12]) for row in high_noise_rows if row[1] == model_name]
        plt.plot(noise_values, values, marker="o", label=model_name)
    plt.xlabel("observation-noise scale")
    plt.ylabel("holdout third-derivative RMSE vs true derivative")
    plt.title("High-noise stress test: third-derivative error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "high_noise_x3_rmse_by_noise.png", dpi=180)
    plt.close()

    highest_noise = float(high_noise_scales[-1])
    y_obs_high, models_high = high_noise_samples[highest_noise]
    fit_base_high = models_high[0][1]
    fit_osc_high = models_high[1][1]
    fit_deriv_high = models_high[2][1]
    high_table = np.column_stack((
        tau_all,
        y_obs_high,
        y_true,
        fit_base_high.value(tau_all, dtype),
        fit_osc_high.value(tau_all, dtype),
        fit_deriv_high.value(tau_all, dtype),
        true_model.derivative(tau_all, 1, dtype),
        fit_osc_high.derivative(tau_all, 1, dtype),
        fit_deriv_high.derivative(tau_all, 1, dtype),
        true_model.derivative(tau_all, 2, dtype),
        fit_osc_high.derivative(tau_all, 2, dtype),
        fit_deriv_high.derivative(tau_all, 2, dtype),
        true_model.derivative(tau_all, 3, dtype),
        fit_osc_high.derivative(tau_all, 3, dtype),
        fit_deriv_high.derivative(tau_all, 3, dtype),
    ))
    write_csv(result_dir / "high_noise_path_and_derivatives.csv", [
        "tau", "observed_high_noise", "true_x", "base_fit_x", "oscillatory_level_fit_x", "oscillatory_derivative_aware_fit_x",
        "true_x1", "oscillatory_level_fit_x1", "oscillatory_derivative_aware_fit_x1",
        "true_x2", "oscillatory_level_fit_x2", "oscillatory_derivative_aware_fit_x2",
        "true_x3", "oscillatory_level_fit_x3", "oscillatory_derivative_aware_fit_x3"
    ], [[float(value) for value in row] for row in high_table])

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(tau_all, y_true, linestyle=(0, (3, 3)), color="black", zorder=20, label="true smooth signal")
    plt.scatter(tau_all, y_obs_high, s=12, color="lightblue", label="high-noise observations")
    plt.plot(tau_all, fit_base_high.value(tau_all, dtype), label="base fit")
    plt.plot(tau_all, fit_osc_high.value(tau_all, dtype), label="level-only oscillatory fit")
    plt.plot(tau_all, fit_deriv_high.value(tau_all, dtype), label="derivative-aware oscillatory fit")
    plt.axvline(float(np.max(tau_train)), linestyle="--", color="purple", label="train/holdout split")
    plt.xlabel("tau")
    plt.ylabel("x")
    plt.title(f"High-noise fitting stress test, noise scale={highest_noise:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "high_noise_fitting_train_holdout.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(tau_all, true_model.derivative(tau_all, 3, dtype), linestyle=(0, (3, 3)), color="green", zorder=20, label="true x triple prime")
    plt.plot(tau_all, fit_osc_high.derivative(tau_all, 3, dtype), label="level-only fit x triple prime")
    plt.plot(tau_all, fit_deriv_high.derivative(tau_all, 3, dtype), label="derivative-aware fit x triple prime")
    plt.axvline(float(np.max(tau_train)), linestyle="--", color="purple", label="train/holdout split")
    plt.xlabel("tau")
    plt.ylabel("third derivative value")
    plt.title(f"High-noise third-derivative stress test, noise scale={highest_noise:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "high_noise_third_derivative_diagnostic.png", dpi=180)
    plt.close()

    messages.append("Fitting demonstration written with baseline and high-noise stress tests.")
    messages.append("High-noise outputs include high_noise_fitting_summary.csv and high-noise diagnostic figures.")
    messages.append("Derivative-aware high-noise results are oracle benchmark results because true derivatives are known only in this synthetic experiment.")
    return messages

def run_dtype_comparison(result_dir: Path) -> list[str]:
    """Compare identity errors under float64 and float32."""

    messages: list[str] = []
    rows: list[list[object]] = []
    for dtype in [np.float64, np.float32]:
        tau = np.linspace(scalar(0.15, dtype), scalar(0.90, dtype), 60, dtype=dtype)
        node = OscillatoryEnvelopeNode(np.float64(1.0), np.float64(2.0), np.float64(9.0), np.float64(0.1), real_array([0.0, -0.07], dtype), T=np.float64(1.0), label="dtype_check_node")
        for order in range(0, 6):
            analytic = node.derivative(tau, order, dtype)
            binomial = node.derivative_constant_slope_binomial(tau, order, dtype)
            error = safe_relative_error(analytic, binomial, dtype)
            rows.append([np.dtype(dtype).name, order, float(error), float(np.finfo(dtype).eps)])
    write_csv(result_dir / "dtype_identity_comparison.csv", ["dtype", "order", "max_relative_error", "machine_epsilon"], rows)
    messages.append("Dtype comparison written to dtype_identity_comparison.csv.")
    return messages



def dtype_from_name(name: str) -> object:
    """Map a command-line dtype name to a NumPy floating dtype."""

    normalized = name.lower().strip()
    if normalized == "float64":
        return np.float64
    if normalized == "float32":
        return np.float32
    raise ValueError("dtype must be either 'float64' or 'float32'.")


def resolve_output_dir(output_dir: str | Path) -> Path:
    """Resolve an output directory relative to the current working directory."""

    path = Path(output_dir).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the GitHub script."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the FDC-LT reference implementation and write CSV/PNG "
            "diagnostics to an output directory."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where result CSV files and figure PNG files are written.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default=np.dtype(DEFAULT_DTYPE).name,
        help="Primary real floating-point dtype used in the demonstration run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for the synthetic noisy fitting demonstrations.",
    )
    return parser

def main(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dtype: object = DEFAULT_DTYPE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Run all FDC-LT implementation demonstrations."""

    base_dir = resolve_output_dir(output_dir)
    result_dir, figure_dir = ensure_output_dirs(base_dir)

    messages: list[str] = []
    messages.append("FDC-LT reference implementation run")
    messages.append(f"Output directory: {base_dir}")
    messages.append(f"Primary dtype: {np.dtype(dtype).name}")
    messages.append(f"Random seed: {random_seed}")
    messages.extend(run_numerical_verification(result_dir, figure_dir, dtype))
    messages.extend(run_finite_difference_h_sweep(result_dir, figure_dir, dtype))
    messages.extend(run_shifted_fractional_example(result_dir, figure_dir, dtype))
    messages.extend(run_shifted_origin_growth_diagnostics(result_dir, dtype))
    messages.extend(run_dimensionless_and_imaginary_checks(result_dir, dtype))
    messages.extend(run_real_signal_examples(result_dir, figure_dir, dtype))
    messages.extend(run_fitting_demonstration(result_dir, figure_dir, dtype, random_seed))
    messages.extend(run_dtype_comparison(result_dir))

    messages.append("")
    messages.append("Numerical-analysis interpretation:")
    messages.append("- Analytic quotient symbols P_n are generated from q-derivatives.")
    messages.append("- Finite differences are used only as external validation and show step-size sensitivity.")
    messages.append("- Shifted-coordinate evaluation controls fractional-power origin singularities.")
    messages.append("- Real projection is delayed until final aggregation over active nodes.")
    messages.append("- Fixed nonlinear structure is separated from linear coefficient estimation and conditioning diagnostics.")
    messages.append("- Derivative-aware stacked least squares tests derivative-informed fitting under controlled synthetic targets.")
    messages.append("- Turning candidates are clustered into intervals so adjacent grid rows are not over-counted.")
    messages.append("- Shifted-origin diagnostics report finite but potentially large derivative-symbol growth near tau=0.")

    with (result_dir / "run_summary.txt").open("w", encoding="utf-8") as handle:
        for line in messages:
            handle.write(line + "\n")

    print("\n".join(messages))


def cli(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        dtype = dtype_from_name(args.dtype)
        main(output_dir=args.output_dir, dtype=dtype, random_seed=args.seed)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        parser.exit(status=1, message=f"FDC-LT run failed: {exc}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(cli())
