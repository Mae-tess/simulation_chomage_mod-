"""
Paramètres pour le modèle ODE de chômage des jeunes diplômés.
Unités : par unité de temps (ici on pense en mois pour dt et T).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Params:
    """Contient les paramètres structurels, numériques et les conditions initiales."""

    Ay: float  # flux d'arrivée de diplômés 
    omega: float  # sortie de cohorte (plus de 4ans apres le diplome)
    k: float  # efficacité de matching 
    mu: float  # sortie exogène(mort, étude en plus ...)
    lambda_: float  # transition vers entrepreneuriat 
    alpha: float  # retraite (ici on le prend quand meme en compte meme si E = employé jeune ca nous permet de ne pas perdre cette information)
    delta: float  # destruction/fermeture de postes 

    rho: float  # santé financière dans [0,1]; 0 = pas de crise, 1 = forte crise
    sigma0: float  # ouverture de postes 
    q0: float  # création de postes par entrepreneurs 
    beta0: float  # constante du job cut 
    gamma: float  # sensibilité de beta0 au stress financier
    eps: float  # courbure de la réponse au stress

    sigma_rho: float  # sigma(rho) effectif
    q_rho: float  # q(rho) effectif
    beta_rho: float  # beta(rho) effectif

    dt: float  # pas de temps en mois 
    T: float  # horizon total de simulation
    U0: float  # U initial
    E0: float  # E initial
    V0: float  # V initial

    tol: float  # tolérance Newton (norme de l'incrément)
    max_iter: int  # itérations max de Newton par pas

    @property
    def n_steps(self) -> int:
        """Nombre de pas d'intégration déduit de T et dt."""
        return int(self.T / self.dt)


def _financial_channels(
    rho: float, sigma0: float, q0: float, beta0: float, gamma: float, eps: float
) -> Tuple[float, float, float]:
    """Calcule sigma(rho), q(rho), beta(rho)."""
    factor = (1.0 - rho) ** eps
    sigma_rho = sigma0 * factor
    q_rho = q0 * factor
    beta_rho = beta0 * (1.0 + gamma * rho) ** eps
    return sigma_rho, q_rho, beta_rho


def make_params(rho: float) -> Params:
    
    Ay = 3000.0  
    omega = 1.0 / 48.0  
    k = 2.5e-6  
    mu = 1.0 / 120.0  
    lambda_ = 1.0 / 36.0  
    alpha = 1.0 / 18.0  
    delta = 1.0 / 8.0  

    # santé financière
    sigma0 = 0.05  
    q0 = 0.08  
    beta0 = 1.0 / 6.0  
    gamma = 1.5  
    eps = 1.1  

    sigma_rho, q_rho, beta_rho = _financial_channels(rho, sigma0, q0, beta0, gamma, eps)

    # condition initiales 
    U0 = 25_000.0
    E0 = 90_000.0
    V0 = 30_000.0

    # Numerical settings
    dt = 0.5  # mois
    T = 120.0  # mois (10 years)
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
