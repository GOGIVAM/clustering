
from init import *
# ============================================================
# Utilitaires partagés
# ============================================================

def _palette(n: int) -> list:
    """Génère n couleurs distinctes en format rgba string."""
    base = [
        "rgba(55,138,221,0.85)",   # bleu
        "rgba(211,83,46,0.85)",    # corail
        "rgba(29,158,117,0.85)",   # teal
        "rgba(172,83,172,0.85)",   # violet
        "rgba(230,160,40,0.85)",   # ambre
        "rgba(200,60,80,0.85)",    # rouge
        "rgba(60,180,60,0.85)",    # vert
        "rgba(100,100,200,0.85)",  # indigo
    ]
    # Répétition si nécessaire
    return [base[i % len(base)] for i in range(n)]


def euclidean_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice des distances euclidiennes au carré
    entre A (m, d) et B (n, d).
    Retourne (m, n).
    Formule : ||a - b||² = ||a||² + ||b||² - 2 a·b
    """
    A2 = np.sum(A ** 2, axis=1, keepdims=True)   # (m, 1)
    B2 = np.sum(B ** 2, axis=1, keepdims=True).T  # (1, n)
    AB = A @ B.T                                   # (m, n)
    return np.maximum(A2 + B2 - 2 * AB, 0.0)      # clip numériqu.


def euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance euclidienne (racine carrée de euclidean_sq)."""
    return np.sqrt(euclidean_sq(A, B))
