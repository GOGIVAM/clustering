from init import *
from base_cluster import BaseClusterer
from shared import _palette, euclidean, euclidean_sq
from partionnal_clustering import KMeans


# ============================================================
# SECTION 5 — Clustering Spectral
# ============================================================

class SpectralClustering(BaseClusterer):
    """
    Algorithme de Ng, Jordan & Weiss (2001) — NeurIPS 2001.

    Étapes :
      1. Construire la matrice d'affinité W_ij = exp(-||xi-xj||²/2σ²)
      2. Calculer le Laplacien symétrique normalisé L_sym = D^{-1/2}WD^{-1/2}
      3. Extraire les k plus grands vecteurs propres de L_sym
      4. Normaliser les lignes de U
      5. Appliquer k-means sur Ũ

    Connexion théorique : relaxation du Normalized Cut (Shi & Malik, 2000).
    Inégalité de Cheeger : h(G)²/2 ≤ λ₂(L_sym) ≤ 2·h(G)
    """

    def __init__(self, k: int, sigma: float = 1.0,
                 affinity: str = "rbf", n_neighbors: int = 10,
                 random_state: int = 42):
        super().__init__("Clustering Spectral (NJW 2001)")
        self.k = k
        self.sigma = sigma
        self.affinity = affinity      # 'rbf' ou 'knn'
        self.n_neighbors = n_neighbors
        self.rng = np.random.default_rng(random_state)

        self.embedding_ = None        # (n, k) espace spectral normalisé
        self.eigenvalues_ = None      # (k,) valeurs propres

    # ----------------------------------------------------------
    # Construction de la matrice d'affinité
    # ----------------------------------------------------------

    def _affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construit W selon le type d'affinité :
          - 'rbf' : W_ij = exp(-||xi-xj||² / 2σ²)  (graphe gaussien complet)
          - 'knn' : W_ij = 1 si xj ∈ kNN(xi), symétrisé
        """
        D2 = euclidean_sq(X, X)    # (n, n)

        if self.affinity == "rbf":
            W = np.exp(-D2 / (2 * self.sigma ** 2))
            np.fill_diagonal(W, 0.0)

        elif self.affinity == "knn":
            n = X.shape[0]
            k = min(self.n_neighbors, n - 1)
            W = np.zeros((n, n))
            for i in range(n):
                # Indices des k plus proches voisins (en excluant i)
                row = D2[i].copy()
                row[i] = np.inf
                nn_idx = np.argpartition(row, k)[:k]
                W[i, nn_idx] = 1.0
            W = np.maximum(W, W.T)   # symmétrisation

        else:
            raise ValueError(f"Affinité inconnue : {self.affinity}")

        return W

    # ----------------------------------------------------------
    # Calcul du Laplacien normalisé et décomposition spectrale
    # ----------------------------------------------------------

    def _spectral_embedding(self, W: np.ndarray) -> np.ndarray:
        """
        Calcule L_sym = D^{-1/2} W D^{-1/2} et retourne les k
        vecteurs propres associés aux k plus grandes valeurs propres.

        Retourne Ũ (n, k) avec lignes normalisées (Algorithme 10, étape 5).
        """
        n = W.shape[0]
        d = W.sum(axis=1)                                # degrés (n,)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))  # D^{-1/2}

        # L_sym = D^{-1/2} W D^{-1/2}
        L_sym = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]

        # Décomposition propre (matrice réelle symétrique)
        eigvals, eigvecs = np.linalg.eigh(L_sym)   # trié par ordre croissant

        # Prendre les k plus grandes valeurs propres
        idx_sorted = np.argsort(eigvals)[::-1]
        top_k = idx_sorted[:self.k]
        self.eigenvalues_ = eigvals[top_k]
        U = eigvecs[:, top_k]                       # (n, k)

        # Normalisation des lignes : ũ_i = u_i / ||u_i||₂
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U_tilde = U / np.maximum(norms, 1e-12)

        return U_tilde

    # ----------------------------------------------------------
    # Interface BaseClusterer
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SpectralClustering":
        """
        Exécute l'algorithme complet NJW :
          W → L_sym → vecteurs propres → normalisation → k-means
        """
        W = self._affinity_matrix(X)
        U_tilde = self._spectral_embedding(W)
        self.embedding_ = U_tilde

        # Étape finale : k-means sur l'espace spectral
        km = KMeans(k=self.k, random_state=42)
        km.fit(U_tilde)
        self.labels_ = km.labels_
        self.n_clusters_ = self.k
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction basée sur l'embedding spectral appris."""
        self._check_fitted()
        return self.labels_

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """
        Panneau gauche  : clusters dans l'espace original (2D ou 3D).
        Panneau droit   : projection 2D de l'espace spectral Ũ.
        """
        self._check_fitted()
        labels = self.labels_
        colors = _palette(self.k)
        is_3d = X.shape[1] >= 3

        if is_3d:
            specs = [[{"type": "scene"}, {"type": "xy"}]]
        else:
            specs = [[{"type": "xy"}, {"type": "xy"}]]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Espace original", "Espace spectral (2 premières dims)"),
            column_widths=[0.5, 0.5],
            specs=specs
        )

        # ---- Panneau gauche : espace original --------------------
        for lbl in range(self.k):
            mask = labels == lbl
            color = colors[lbl % len(colors)]
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.85),
                    name=f"Cluster {lbl}"
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=X[mask, 0], y=X[mask, 1],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.85),
                    name=f"Cluster {lbl}"
                ), row=1, col=1)

        # ---- Panneau droit : espace spectral (toujours 2D) -------
        U = self.embedding_
        for lbl in range(self.k):
            mask = labels == lbl
            color = colors[lbl % len(colors)]
            fig.add_trace(go.Scatter(
                x=U[mask, 0],
                y=U[mask, 1] if U.shape[1] > 1 else np.zeros(mask.sum()),
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.85),
                name=f"Spectral {lbl}",
                showlegend=False
            ), row=1, col=2)

        fig.update_xaxes(title_text="u₁", row=1, col=2)
        fig.update_yaxes(title_text="u₂", row=1, col=2)

        if is_3d:
            fig.update_layout(scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃"
            ))

        fig.update_layout(
            title=dict(
                text=f"{self.name} {title_suffix}<br>"
                     f"<sup>k={self.k}, σ={self.sigma}, affinité={self.affinity}</sup>",
                font=dict(size=15)
            ),
            template="plotly_white",
            showlegend=True,
            margin=dict(l=40, r=20, t=80, b=40)
        )
        return fig