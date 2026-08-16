#!/usr/bin/env python3

"""
fit_parametric_curve.py

Fits the unknown parameters theta, M and X of the parametric curve:

    x(t) = t*cos(theta)
           - exp(M*|t|)*sin(0.3*t)*sin(theta)
           + X

    y(t) = 42
           + t*sin(theta)
           + exp(M*|t|)*sin(0.3*t)*cos(theta)

Assignment constraints:

    0 < theta < 50 degrees
    -0.05 < M < 0.05
    0 < X < 100

    6 < t < 60

Usage:

    python fit_parametric_curve.py --input xy_data.csv --out results

Outputs:

    1. Estimated theta, M and X
    2. L1 distance between observed and predicted curves
    3. Recovered t range
    4. LaTeX-ready parametric equation
    5. Results JSON file
    6. Comparison graph
"""


import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.interpolate import interp1d


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):
    """
    Load x,y coordinates from CSV.

    The CSV must contain columns:
        x
        y
    """

    df = pd.read_csv(path)

    required_columns = {"x", "y"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named 'x' and 'y'."
        )

    # Remove missing values
    df = df[["x", "y"]].dropna()

    if len(df) < 3:
        raise ValueError(
            "CSV must contain at least 3 valid data points."
        )

    return df


# ============================================================
# 2. TRANSFORM (x,y) -> (t,s)
# ============================================================

def compute_t_s(theta, X, x, y):
    """
    Transform observed x,y coordinates.

    From the given parametric equations:

        t = (x-X)*cos(theta)
            + (y-42)*sin(theta)

        s = -(x-X)*sin(theta)
            + (y-42)*cos(theta)

    The model then becomes:

        s = exp(M*|t|) * sin(0.3*t)
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_shift = x - X
    y_shift = y - 42.0

    t = (
        x_shift * cos_theta
        + y_shift * sin_theta
    )

    s = (
        -x_shift * sin_theta
        + y_shift * cos_theta
    )

    return t, s


# ============================================================
# 3. RESIDUAL FUNCTION
# ============================================================

def residuals(params, x, y):
    """
    Residual used by nonlinear least-squares optimization.

    Parameters:
        params = [theta, M, X]

    The model requires:

        s = exp(M*|t|) * sin(0.3*t)

    Therefore:

        residual = s_observed - s_model
    """

    theta, M, X = params

    # Recover t and s
    t, s_observed = compute_t_s(
        theta,
        X,
        x,
        y
    )

    # Model value of s
    s_model = (
        np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
    )

    # Residual
    return s_observed - s_model


# ============================================================
# 4. FIT PARAMETERS
# ============================================================

def fit_parameters(x, y):
    """
    Estimate theta, M and X using nonlinear least squares.

    Multiple initial guesses are used to make the fitting
    more robust.
    """

    # --------------------------------------------------------
    # Assignment bounds
    # --------------------------------------------------------

    theta_lower = np.deg2rad(0.000001)
    theta_upper = np.deg2rad(49.999999)

    M_lower = -0.049999
    M_upper = 0.049999

    X_lower = 0.000001
    X_upper = 99.999999

    lower_bounds = np.array([
        theta_lower,
        M_lower,
        X_lower
    ])

    upper_bounds = np.array([
        theta_upper,
        M_upper,
        X_upper
    ])

    # --------------------------------------------------------
    # Initial guesses
    # --------------------------------------------------------

    initial_guesses = [
        [np.deg2rad(10.0), -0.02, 20.0],
        [np.deg2rad(20.0),  0.00, 40.0],
        [np.deg2rad(25.0),  0.00, 50.0],
        [np.deg2rad(30.0),  0.02, 60.0],
        [np.deg2rad(40.0),  0.02, 80.0],
    ]

    best_result = None
    best_cost = np.inf

    # --------------------------------------------------------
    # Run optimization from multiple starting points
    # --------------------------------------------------------

    for initial_guess in initial_guesses:

        result = least_squares(
            residuals,
            x0=np.array(initial_guess),
            args=(x, y),
            bounds=(
                lower_bounds,
                upper_bounds
            ),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=200000
        )

        cost = np.sum(result.fun ** 2)

        if cost < best_cost:
            best_cost = cost
            best_result = result

    return best_result


# ============================================================
# 5. GENERATE PARAMETRIC CURVE
# ============================================================

def model_xy(theta, M, X, t):
    """
    Generate x,y coordinates from the fitted parameters.
    """

    oscillation = (
        np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
    )

    x_model = (
        t * np.cos(theta)
        - oscillation * np.sin(theta)
        + X
    )

    y_model = (
        42.0
        + t * np.sin(theta)
        + oscillation * np.cos(theta)
    )

    return x_model, y_model


# ============================================================
# 6. COMPUTE L1 DISTANCE
# ============================================================

def compute_l1(
    theta,
    M,
    X,
    x_observed,
    y_observed,
    n_samples=None
):
    """
    Compute L1 distance between:

        observed curve
        predicted parametric curve

    using uniformly sampled t values.

    The observed CSV points are first associated with their
    recovered t values and then linearly interpolated.
    """

    if n_samples is None:
        n_samples = len(x_observed)

    # --------------------------------------------------------
    # Recover t values from observed points
    # --------------------------------------------------------

    t_recovered, _ = compute_t_s(
        theta,
        X,
        x_observed,
        y_observed
    )

    # --------------------------------------------------------
    # Sort data by recovered t
    # --------------------------------------------------------

    sort_index = np.argsort(t_recovered)

    t_sorted = t_recovered[sort_index]
    x_sorted = x_observed[sort_index]
    y_sorted = y_observed[sort_index]

    # --------------------------------------------------------
    # Remove duplicate t values if present
    # --------------------------------------------------------

    t_unique, unique_indices = np.unique(
        t_sorted,
        return_index=True
    )

    x_unique = x_sorted[unique_indices]
    y_unique = y_sorted[unique_indices]

    if len(t_unique) < 2:
        raise ValueError(
            "Not enough unique recovered t values for interpolation."
        )

    # --------------------------------------------------------
    # Uniform t range
    #
    # Assignment says 6 < t < 60.
    #
    # We use the overlap between the assignment range and
    # the recovered data range to avoid extrapolation.
    # --------------------------------------------------------

    t_start = max(6.0, float(t_unique.min()))
    t_end = min(60.0, float(t_unique.max()))

    if t_start >= t_end:
        raise ValueError(
            "Recovered t values do not overlap the range 6 < t < 60."
        )

    t_uniform = np.linspace(
        t_start,
        t_end,
        n_samples
    )

    # --------------------------------------------------------
    # Interpolate observed curve
    # --------------------------------------------------------

    interpolate_x = interp1d(
        t_unique,
        x_unique,
        kind="linear",
        bounds_error=False
    )

    interpolate_y = interp1d(
        t_unique,
        y_unique,
        kind="linear",
        bounds_error=False
    )

    x_observed_uniform = interpolate_x(t_uniform)
    y_observed_uniform = interpolate_y(t_uniform)

    # --------------------------------------------------------
    # Predicted curve
    # --------------------------------------------------------

    x_predicted_uniform, y_predicted_uniform = model_xy(
        theta,
        M,
        X,
        t_uniform
    )

    # --------------------------------------------------------
    # L1 distance
    #
    # L1 for each point:
    #
    # |x_pred - x_obs| + |y_pred - y_obs|
    #
    # Total:
    #
    # sum over all uniformly sampled points
    # --------------------------------------------------------

    pointwise_l1 = (
        np.abs(
            x_predicted_uniform
            - x_observed_uniform
        )
        +
        np.abs(
            y_predicted_uniform
            - y_observed_uniform
        )
    )

    total_l1 = np.sum(pointwise_l1)

    return (
        total_l1,
        t_uniform,
        x_observed_uniform,
        y_observed_uniform,
        x_predicted_uniform,
        y_predicted_uniform
    )


# ============================================================
# 7. SAVE GRAPH
# ============================================================

def save_plot(
    x_observed,
    y_observed,
    x_observed_uniform,
    y_observed_uniform,
    x_predicted,
    y_predicted,
    theta_deg,
    M,
    X,
    output_path
):
    """
    Save expected vs predicted curve graph.
    """

    plt.figure(figsize=(10, 6))

    # Raw CSV points
    plt.scatter(
        x_observed,
        y_observed,
        s=10,
        alpha=0.55,
        label="Observed CSV points"
    )

    # Interpolated observed curve
    plt.plot(
        x_observed_uniform,
        y_observed_uniform,
        linewidth=2,
        label="Expected curve"
    )

    # Predicted parametric curve
    plt.plot(
        x_predicted,
        y_predicted,
        linewidth=2,
        linestyle="--",
        label="Predicted curve"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        "Expected vs Predicted Parametric Curve"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fit theta, M and X for the given "
            "parametric curve."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        default="xy_data.csv",
        help="Path to xy_data.csv"
    )

    parser.add_argument(
        "--out",
        "-o",
        default="fit_output",
        help="Output filename prefix"
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the graph"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    print("=" * 60)
    print("PARAMETRIC CURVE PARAMETER ESTIMATION")
    print("=" * 60)

    print("\nLoading data...")

    df = load_data(args.input)

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    print(f"Number of data points: {len(df)}")

    # --------------------------------------------------------
    # Fit parameters
    # --------------------------------------------------------

    print("\nFitting parameters...")

    result = fit_parameters(
        x,
        y
    )

    if not result.success:
        print("\nWARNING:")
        print("Optimization did not report success.")
        print(result.message)

    # --------------------------------------------------------
    # Extract parameters
    # --------------------------------------------------------

    theta = result.x[0]
    M = result.x[1]
    X = result.x[2]

    theta_deg = np.rad2deg(theta)

    # --------------------------------------------------------
    # Calculate L1
    # --------------------------------------------------------

    (
        l1,
        t_uniform,
        x_obs_uniform,
        y_obs_uniform,
        x_pred_uniform,
        y_pred_uniform
    ) = compute_l1(
        theta,
        M,
        X,
        x,
        y,
        n_samples=len(df)
    )

    # --------------------------------------------------------
    # Recover t statistics
    # --------------------------------------------------------

    t_recovered, _ = compute_t_s(
        theta,
        X,
        x,
        y
    )

    t_min = np.min(t_recovered)
    t_max = np.max(t_recovered)
    t_mean = np.mean(t_recovered)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULT")
    print("=" * 60)

    print(f"Success          : {result.success}")
    print(f"Function calls   : {result.nfev}")

    print("\n" + "-" * 60)
    print("FITTED PARAMETERS")
    print("-" * 60)

    print(
        f"Theta (radians)  : {theta:.12f}"
    )

    print(
        f"Theta (degrees)  : {theta_deg:.12f}"
    )

    print(
        f"M                : {M:.12f}"
    )

    print(
        f"X                : {X:.12f}"
    )

    print("\nRounded assignment values:")

    print(
        f"Theta = {theta_deg:.2f} degrees"
    )

    print(
        f"M     = {M:.2f}"
    )

    print(
        f"X     = {X:.2f}"
    )

    # --------------------------------------------------------
    # t range
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("RECOVERED t RANGE")
    print("-" * 60)

    print(f"Minimum t : {t_min:.8f}")
    print(f"Maximum t : {t_max:.8f}")
    print(f"Mean t    : {t_mean:.8f}")

    # --------------------------------------------------------
    # L1
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("L1 METRIC")
    print("-" * 60)

    print(
        f"Uniform samples : {len(t_uniform)}"
    )

    print(
        f"Total L1        : {l1:.12f}"
    )

    # --------------------------------------------------------
    # LaTeX equation
    # --------------------------------------------------------

    latex_equation = (
        "\\left("
        f"t\\cos({theta:.12f})"
        f"-e^{{{M:.12f}|t|}}"
        "\\cdot\\sin(0.3t)"
        f"\\sin({theta:.12f})"
        f"+{X:.8f},"
        "42+"
        f"t\\sin({theta:.12f})"
        f"+e^{{{M:.12f}|t|}}"
        "\\cdot\\sin(0.3t)"
        f"\\cos({theta:.12f})"
        "\\right)"
    )

    print("\n" + "-" * 60)
    print("LATEX-READY EQUATION")
    print("-" * 60)

    print(latex_equation)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {
        "theta_rad": float(theta),
        "theta_deg": float(theta_deg),
        "M": float(M),
        "X": float(X),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "t_mean": float(t_mean),
        "L1_distance": float(l1),
        "n_points": int(len(df)),
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
        "latex_equation": latex_equation
    }

    json_path = args.out + "_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"\nResults saved to: {json_path}"
    )

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    plot_path = args.out + "_fit.png"

    save_plot(
        x,
        y,
        x_obs_uniform,
        y_obs_uniform,
        x_pred_uniform,
        y_pred_uniform,
        theta_deg,
        M,
        X,
        plot_path
    )

    print(
        f"Graph saved to: {plot_path}"
    )

    # --------------------------------------------------------
    # Show graph if requested
    # --------------------------------------------------------

    if args.show_plot:

        plt.figure(figsize=(10, 6))

        plt.scatter(
            x,
            y,
            s=10,
            alpha=0.55,
            label="Observed CSV points"
        )

        plt.plot(
            x_obs_uniform,
            y_obs_uniform,
            linewidth=2,
            label="Expected curve"
        )

        plt.plot(
            x_pred_uniform,
            y_pred_uniform,
            linewidth=2,
            linestyle="--",
            label="Predicted curve"
        )

        plt.xlabel("x")
        plt.ylabel("y")

        plt.title(
            "Expected vs Predicted Parametric Curve"
        )

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
