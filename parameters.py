"""
Parameter definitions for the youth unemployment ODE model.
The units are consistent with "per unit time" where the time unit is chosen
by the user (e.g., months). The dt and T values below assume months.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Params:
    """Container for structural, numerical, and initial-value parameters."""

    Ay: float  # Inflow of new graduates per unit time (>0)
    omega: float  # Exit rate from the 1–4 year cohort (1/time)
    k: float  # Matching efficiency between unemployed and vacancies (1/(person*job*time))
    mu: float  # Exogenous exit from unemployment/inactivity/migration (1/time)
    lambda_: float  # Transition rate from unemployment to employment via entrepreneurship (1/time)
    alpha: float  # Vacancy closure via internal recruitment/administrative processes (1/time)
    delta: float  # Vacancy destruction/closure rate (1/time)

    rho: float  # Financial stress level in [0,1]; 0 = no crisis, 1 = severe crisis
    sigma0: float  # Baseline opening of vacancies from unemployed (1/time)
    q0: float  # Baseline rate of entrepreneurs creating vacancies (1/time)
    beta0: float  # Baseline layoff/backflow rate from employment to unemployment (1/time)
    gamma: float  # Sensitivity of layoffs to financial stress (dimensionless)
    eps: float  # Curvature of the stress response (dimensionless)

    sigma_rho: float  # sigma(rho) effective vacancy creation from U
    q_rho: float  # q(rho) effective entrepreneurial channel
    beta_rho: float  # beta(rho) effective backflow from E to U

    dt: float  # Time step for integration (same unit as omega, etc.)
    T: float  # Total simulation horizon
    U0: float  # Initial unemployed graduates in cohort
    E0: float  # Initial employed graduates in cohort
    V0: float  # Initial junior vacancies

    tol: float  # Newton solver tolerance on increment norm
    max_iter: int  # Maximum Newton iterations per implicit step

    @property
    def n_steps(self) -> int:
        """Number of integration steps based on T and dt."""
        return int(self.T / self.dt)


def _financial_channels(
    rho: float, sigma0: float, q0: float, beta0: float, gamma: float, eps: float
) -> Tuple[float, float, float]:
    """Compute the rho-dependent channels (sigma, q, beta)."""
    factor = (1.0 - rho) ** eps
    sigma_rho = sigma0 * factor
    q_rho = q0 * factor
    beta_rho = beta0 * (1.0 + gamma * rho) ** eps
    return sigma_rho, q_rho, beta_rho


def make_params(rho: float) -> Params:
    """
    Build a Params instance for a given financial stress level rho in [0,1].

    The baseline values below are illustrative; adjust them to match empirical targets.
    Time is expressed in months, so rates are per month.
    """

    # Structural parameters
    Ay = 3000.0  # graduates arriving per month
    omega = 1.0 / 48.0  # exit from 1–4 year cohort (~4 years)
    k = 2.5e-6  # matching efficiency (scaled for population size)
    mu = 1.0 / 120.0  # slow exogenous exits
    lambda_ = 1.0 / 36.0  # entrepreneurship channel
    alpha = 1.0 / 18.0  # administrative vacancy closure
    delta = 1.0 / 8.0  # vacancy destruction

    # Financial-health baselines
    sigma0 = 0.05  # baseline vacancy creation from unemployed
    q0 = 0.08  # baseline entrepreneurial vacancy openings
    beta0 = 1.0 / 6.0  # baseline layoff/return to unemployment
    gamma = 1.5  # stress sensitivity of layoffs
    eps = 1.1  # curvature on stress response

    sigma_rho, q_rho, beta_rho = _financial_channels(rho, sigma0, q0, beta0, gamma, eps)

    # Initial conditions (stocks of people or positions)
    U0 = 25_000.0
    E0 = 90_000.0
    V0 = 30_000.0

    # Numerical settings
    dt = 0.5  # months
    T = 120.0  # months (10 years)
    tol = 1e-8
    max_iter = 25

    return Params(
        Ay=Ay,
        omega=omega,
        k=k,
        mu=mu,
        lambda_=lambda_,
        alpha=alpha,
        delta=delta,
        rho=rho,
        sigma0=sigma0,
        q0=q0,
        beta0=beta0,
        gamma=gamma,
        eps=eps,
        sigma_rho=sigma_rho,
        q_rho=q_rho,
        beta_rho=beta_rho,
        dt=dt,
        T=T,
        U0=U0,
        E0=E0,
        V0=V0,
        tol=tol,
        max_iter=max_iter,
    )
