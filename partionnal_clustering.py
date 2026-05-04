from init import *
from base_cluster import BaseClusterer
from shared import _palette, euclidean, euclidean_sq


# ============================================================
# SECTION 1 — Clustering Partitionnel : KMeans
# ============================================================

class KMeans(BaseClusterer):
    """
    Algorithme de Lloyd (k-means) avec initialisation k-means++.

    Référence :
      - Lloyd (1957/1982) — IEEE Trans. Information Theory
      - Arthur & Vassilvitskii (2007) — SODA 2007
    """

    def __init__(self, k: int, max_iter: int = 300, tol: float = 1e-6,
                 n_init: int = 10, random_state: int = 42):
        super().__init__("K-Means (Lloyd + k-means++)")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.rng = np.random.default_rng(random_state)

        self.centroids_ = None       # (k, d)
        self.inertia_ = None         # valeur finale de Lk-means
        self.n_iter_ = None          # nombre d'itérations à convergence

    # ----------------------------------------------------------
    # Initialisation k-means++
    # ----------------------------------------------------------

    def _init_kmeanspp(self, X: np.ndarray) -> np.ndarray:
        """
        Initialise k centroïdes selon k-means++.

        Étape 1 : choisir µ₁ uniformément dans X.
        Étape j : choisir µⱼ avec probabilité ∝ D(x)²,
                  où D(x) = min_{l<j} ||x − µₗ||².

        Garantie : E[L] ≤ 8(ln k + 2) · L*  (Théorème 1.2)
        """
        n, d = X.shape
        centroids = []

        # Étape 1
        first_idx = self.rng.integers(0, n)
        centroids.append(X[first_idx])

        for _ in range(1, self.k):
            # D(x) = distance minimale au carré aux centroïdes déjà choisis
            C = np.array(centroids)                # (j, d)
            dists = euclidean_sq(X, C)             # (n, j)
            D = dists.min(axis=1)                  # (n,)

            # Tirage proportionnel à D(x)
            probs = D / D.sum()
            cum = np.cumsum(probs)
            r = self.rng.random()
            idx = np.searchsorted(cum, r)
            centroids.append(X[min(idx, n - 1)])

        return np.array(centroids)

    # ----------------------------------------------------------
    # Une exécution de l'algorithme de Lloyd
    # ----------------------------------------------------------

    def _lloyd(self, X: np.ndarray) -> tuple:
        """
        Exécute l'algorithme de Lloyd depuis une initialisation k-means++.

        Retourne : (centroids, labels, inertia, n_iter)
        """
        centroids = self._init_kmeanspp(X)
        n = X.shape[0]
        labels = np.zeros(n, dtype=int)
        prev_inertia = np.inf

        for t in range(self.max_iter):

            # ---- Étape d'affectation --------------------------------
            # C(xi) = argmin_j ||xi − µj||²
            D = euclidean_sq(X, centroids)   # (n, k)
            labels = D.argmin(axis=1)        # (n,)

            # ---- Étape de mise à jour des centroïdes ----------------
            new_centroids = np.zeros_like(centroids)
            for j in range(self.k):
                mask = labels == j
                if mask.sum() == 0:
                    # Cluster vide → réinitialiser sur un point aléatoire
                    new_centroids[j] = X[self.rng.integers(0, n)]
                else:
                    new_centroids[j] = X[mask].mean(axis=0)

            # ---- Calcul de l'inertie --------------------------------
            inertia = sum(
                euclidean_sq(X[labels == j], new_centroids[j:j+1]).sum()
                for j in range(self.k)
            )

            centroids = new_centroids

            # ---- Critère d'arrêt ------------------------------------
            if abs(prev_inertia - inertia) < self.tol:
                return centroids, labels, inertia, t + 1

            prev_inertia = inertia

        return centroids, labels, inertia, self.max_iter

    # ----------------------------------------------------------
    # Interface BaseClusterer
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Exécute n_init fois Lloyd et conserve la meilleure solution
        (inertie minimale).
        """
        best = None
        for _ in range(self.n_init):
            result = self._lloyd(X)
            if best is None or result[2] < best[2]:
                best = result

        self.centroids_, self.labels_, self.inertia_, self.n_iter_ = best
        self.n_clusters_ = self.k
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigne chaque point de X au centroïde le plus proche."""
        self._check_fitted()
        D = euclidean_sq(X, self.centroids_)
        return D.argmin(axis=1)

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """Ajoute les centroïdes comme étoiles sur la figure."""
        fig = super().visualize(X, title_suffix)
        is_3d = X.shape[1] >= 3
        colors = _palette(self.k)

        for j in range(self.k):
            c = self.centroids_[j]
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=[c[0]], y=[c[1]], z=[c[2]],
                    mode="markers",
                    marker=dict(symbol="diamond", size=8,
                                color=colors[j % len(colors)],
                                line=dict(width=2, color="black")),
                    name=f"Centroïde {j}",
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[c[0]], y=[c[1]],
                    mode="markers",
                    marker=dict(symbol="star", size=18,
                                color=colors[j % len(colors)],
                                line=dict(width=1.5, color="black")),
                    name=f"Centroïde {j}",
                    showlegend=False
                ))

        inertia_str = f"Inertie = {self.inertia_:.2f} | Itérations = {self.n_iter_}"
        fig.update_layout(title=dict(
            text=f"{self.name} {title_suffix}<br><sup>{inertia_str}</sup>"
        ))
        return fig


