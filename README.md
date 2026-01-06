# Simulation du chômage des jeunes diplômés (Euler implicite)

## Lancer la simulation principale
1. Installer les dépendances si besoin :
   ```bash
   pip install numpy matplotlib
   ```
2. Exécuter le script principal :
   ```bash
   python3 main.py
   ```
3. Saisir trois valeurs de `rho` (santé financière 0–1) séparées par des virgules ou valider pour utiliser les valeurs par défaut (0.0, 0.2, 0.5). Le script affiche les résultats finaux et ouvre les graphiques pour comparer `tau(t)` et les trajectoires `U, E, V`.

## Structure des fichiers
- `parameters.py` : définit la dataclass `Params`, la fonction `make_params(rho)` pour construire les paramètres dépendants de `rho`, les conditions initiales et les réglages numériques (dt, tol, max_iter).
- `model.py` : contient le système ODE `f`, sa jacobienne, le calcul du résidu implicite, la fonction `step_backward_euler` (Euler implicite résolu par Newton-Raphson) et `simulate` qui génère les trajectoires `U, E, V, tau` et remonte les échecs Newton éventuels.
- `main.py` : interface simple en console pour choisir trois scénarios de `rho`, lancer `simulate`, tracer `tau(t)` et les trajectoires d’état, puis afficher les valeurs finales (U, E, V, tau) et les erreurs Newton si elles existent.
- `data.py` : placeholder (non utilisé dans cette version car `Ay` est constant).
- `setting.py` : réservé/placeholder (non utilisé par les scripts ci-dessus).
