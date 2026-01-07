"""
Modèle ODE et intégration implicite pour la dynamique du chômage des jeunes diplômés.
"""

from typing import List, Tuple

import numpy as np

from parameters import Params


def f(x: np.ndarray, params: Params) -> np.ndarray:
    """Calcule les dérivées [dU, dE, dV] au point x = [U, E, V]."""
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
    """Jacobienne de f par rapport à [U, E, V]."""
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
    """Résidu F(x_{n+1}) pour Euler implicite : x_{n+1} - x_n - dt * f(x_{n+1})."""
    return x_next - x_n - params.dt * f(x_next, params)


def step_backward_euler(x_n: np.ndarray, params: Params) -> Tuple[np.ndarray, bool, str]:
    """
    Effectue un pas d'Euler implicite résolu par Newton-Raphson.

    Retourne :
        x_next : état au pas suivant (recadré en valeurs positives).
        success : False si Newton n'a pas convergé.
        message : diagnostic lorsque success est False.
    """
    # Point de départ : le pas précédent, en forçant U,E,V >= 0
    x = np.maximum(x_n, 0.0)
    success = False
    message = ""

    for _ in range(params.max_iter):
        # F(x_{n+1}) = x_{n+1} - x_n - dt * f(x_{n+1})
        F = residual(x, x_n, params)
        J = jacobian_f(x, params)
        # Jacobienne du résidu : I - dt * J
        J_F = np.eye(3) - params.dt * J
        try:
            delta = np.linalg.solve(J_F, -F)  # résolution du système linéaire
        except np.linalg.LinAlgError as exc:
            message = f"Jacobian solve failed: {exc}"
            break

        x = x + delta
        x = np.maximum(x, 0.0)  # on reste dans le domaine U,E,V >= 0

        if np.linalg.norm(delta, ord=2) < params.tol:
            success = True
            break

    if not success and message == "":
        message = "Newton did not converge (max_iter reached)"

    return x, success, message


def simulate(params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Simule sur [0, T] avec Euler implicite.

    Retourne :
        t : grille temporelle
        U, E, V : trajectoires
        tau : ratio de chômage U / (U + E)
        errors : messages d'échec de Newton avec le pas concerné
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
