"""
ODE model and implicit integration (Backward Euler) for youth unemployment dynamics.
"""

from typing import List, Tuple

import numpy as np

from parameters import Params


def f(x: np.ndarray, params: Params) -> np.ndarray:
    """Compute the time derivative [dU, dE, dV] at state x = [U, E, V]."""
    U, E, V = x
    beta = params.beta_rho
    sigma = params.sigma_rho
    q = params.q_rho
    Ay = params.Ay
    k = params.k
    lambda_ = params.lambda_
    alpha = params.alpha
    mu = params.mu
    omega = params.omega
    delta = params.delta

    dU = Ay - k * U * V + beta * E - (mu + lambda_ + omega) * U
    dE = k * U * V - (beta + alpha + mu + omega) * E + lambda_ * U
    dV = (lambda_ * q + sigma) * U + (alpha + mu) * E - delta * V - k * U * V
    return np.array([dU, dE, dV], dtype=float)


def jacobian_f(x: np.ndarray, params: Params) -> np.ndarray:
    """Jacobian of f with respect to [U, E, V]."""
    U, E, V = x
    beta = params.beta_rho
    sigma = params.sigma_rho
    q = params.q_rho
    k = params.k
    lambda_ = params.lambda_
    alpha = params.alpha
    mu = params.mu
    omega = params.omega
    delta = params.delta

    df_dU = -k * V - (mu + lambda_ + omega)
    df_dE = beta
    df_dV = -k * U

    dg_dU = k * V + lambda_
    dg_dE = -(beta + alpha + mu + omega)
    dg_dV = k * U

    dh_dU = lambda_ * q + sigma - k * V
    dh_dE = alpha + mu
    dh_dV = -delta - k * U

    return np.array(
        [
            [df_dU, df_dE, df_dV],
            [dg_dU, dg_dE, dg_dV],
            [dh_dU, dh_dE, dh_dV],
        ],
        dtype=float,
    )


def residual(x_next: np.ndarray, x_n: np.ndarray, params: Params) -> np.ndarray:
    """Residual F(x_{n+1}) for the implicit Euler step: x_{n+1} - x_n - dt * f(x_{n+1})."""
    return x_next - x_n - params.dt * f(x_next, params)


def step_backward_euler(x_n: np.ndarray, params: Params) -> Tuple[np.ndarray, bool, str]:
    """
    Single Backward Euler step solved by Newton-Raphson.

    Returns:
        x_next: state at the next time step (clipped to non-negative).
        success: False if Newton failed to converge.
        message: diagnostic text when success is False.
    """
    x = np.maximum(x_n, 0.0)  # start from previous step, enforce non-negativity
    success = False
    message = ""

    for _ in range(params.max_iter):
        F = residual(x, x_n, params)
        J = jacobian_f(x, params)
        J_F = np.eye(3) - params.dt * J
        try:
            delta = np.linalg.solve(J_F, -F)
        except np.linalg.LinAlgError as exc:
            message = f"Jacobian solve failed: {exc}"
            break

        x = x + delta
        x = np.maximum(x, 0.0)  # keep states feasible

        if np.linalg.norm(delta, ord=2) < params.tol:
            success = True
            break

    if not success and message == "":
        message = "Newton did not converge (max_iter reached)"

    return x, success, message


def simulate(params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run the simulation over [0, T] with Backward Euler.

    Returns:
        t: time grid
        U, E, V: trajectories
        tau: unemployment ratio U / (U + E)
        errors: list of Newton failure messages with time-step context
    """
    n_steps = params.n_steps
    t = np.linspace(0.0, params.T, n_steps + 1)
    x = np.zeros((n_steps + 1, 3), dtype=float)
    x[0, :] = np.array([params.U0, params.E0, params.V0], dtype=float)

    errors: List[str] = []
    for n in range(n_steps):
        x_next, success, msg = step_backward_euler(x[n], params)
        if not success:
            errors.append(f"Step {n + 1} at t={t[n + 1]:.3f}: {msg}")
        x[n + 1] = x_next

    U = x[:, 0]
    E = x[:, 1]
    V = x[:, 2]
    tau = U / np.maximum(U + E, 1e-12)
    return t, U, E, V, tau, errors
