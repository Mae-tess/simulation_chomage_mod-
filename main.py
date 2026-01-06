"""
Simulation d'un modèle ODE du chômage des jeunes diplômés (cohorte 1–4 ans).

Variables:
- U(t) : chômeurs récents
- E(t) : employés récents
- V(t) : postes vacants accessibles
- tau(t) = U / (U + E) : taux de chômage dans la cohorte
- rho ∈ [0,1] : santé financière (0 = aucune crise, 1 = crise forte)

Le système est intégré par Euler implicite avec Newton-Raphson à chaque pas.
"""

from typing import Iterable, List

import matplotlib.pyplot as plt

from model import simulate
from parameters import make_params


DEFAULT_RHOS = [0.0, 0.2, 0.5]


def _parse_rhos(user_input: str, default: Iterable[float]) -> List[float]:
    """Parse a comma-separated list of rhos in [0,1]; fallback to default on errors."""
    values: List[float] = []
    for part in user_input.split(","):
        if part.strip() == "":
            continue
        try:
            val = float(part.strip())
            if 0.0 <= val <= 1.0:
                values.append(val)
        except ValueError:
            continue
    return values if len(values) == 3 else list(default)


def ask_rhos() -> List[float]:
    prompt = (
        "Entrez trois valeurs de rho (0 à 1) séparées par des virgules "
        f"(Entrée pour défaut {DEFAULT_RHOS}): "
    )
    try:
        user_input = input(prompt)
    except EOFError:
        return list(DEFAULT_RHOS)

    if user_input.strip() == "":
        return list(DEFAULT_RHOS)
    return _parse_rhos(user_input, DEFAULT_RHOS)


def plot_tau(results: List[dict]) -> None:
    plt.figure(figsize=(7, 4))
    for res in results:
        plt.plot(res["t"], res["tau"], label=f"rho={res['rho']:.2f}")
    plt.xlabel("Temps")
    plt.ylabel("tau(t) = U / (U + E)")
    plt.title("Comparaison du taux de chômage tau(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_states(results: List[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    labels = ["U (chômeurs)", "E (employés)", "V (postes vacants)"]
    keys = ["U", "E", "V"]
    for ax, label, key in zip(axes, labels, keys):
        for res in results:
            ax.plot(res["t"], res[key], label=f"rho={res['rho']:.2f}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Temps")
    axes[0].legend()
    fig.suptitle("État des stocks U, E, V")


def main() -> None:
    print("Simulation du chômage des jeunes diplômés (Euler implicite + Newton).")
    rhos = ask_rhos()

    results: List[dict] = []
    for rho in rhos:
        params = make_params(rho)
        t, U, E, V, tau, errors = simulate(params)
        results.append({"rho": rho, "t": t, "U": U, "E": E, "V": V, "tau": tau, "errors": errors})

    plot_tau(results)
    plot_states(results)

    print("\nRésumé final (t = T):")
    for res in results:
        tau_final = res["tau"][-1]
        U_final = res["U"][-1]
        E_final = res["E"][-1]
        V_final = res["V"][-1]
        err_msg = f"{len(res['errors'])} pas Newton non convergents" if res["errors"] else "Newton OK"
        print(
            f"rho={res['rho']:.2f} | tau_final={tau_final:.4f} "
            f"| U={U_final:.0f}, E={E_final:.0f}, V={V_final:.0f} | {err_msg}"
        )
        if res["errors"]:
            print("  Dernier échec:", res["errors"][-1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
