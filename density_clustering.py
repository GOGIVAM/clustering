from init import *
from base_cluster import BaseClusterer
from shared import _palette, euclidean, euclidean_sq


# ============================================================
# SECTION 3 — Clustering Densité : DBSCAN
# ============================================================

class DBSCAN(BaseClusterer):
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Définit les clusters comme des régions de haute densité
    séparées par des régions de faible densité.
    Identifie les outliers (étiquette −1).

    Référence : Ester et al. (1996) — KDD 1996
    Complexité : O(n²) naïf (implémenté ici sans index spatial).
    """

    def __init__(self, eps: float = 0.5, min_pts: int = 5):
        super().__init__("DBSCAN")
        self.eps = eps
        self.min_pts = min_pts
        self.core_mask_ = None    # (n,) booléen : points cœurs

    # ----------------------------------------------------------
    # Voisinage ε
    # ----------------------------------------------------------

    def _neighbors(self, D: np.ndarray, idx: int) -> np.ndarray:
        """
        Retourne les indices des voisins de X[idx] dans Nε(idx).
        D est la matrice des distances euclidiennes complète.
        """
        return np.where(D[idx] <= self.eps)[0]

    # ----------------------------------------------------------
    # Expansion d'un cluster à partir d'un point cœur
    # ----------------------------------------------------------

    def _expand_cluster(self, D: np.ndarray, labels: np.ndarray,
                        visited: np.ndarray, seed: int,
                        neighbors: np.ndarray, cluster_id: int) -> None:
        """
        BFS depuis le point cœur `seed`.
        Propage l'étiquette cluster_id à tous les points
        densité-connexes (Définition 3.1).
        """
        labels[seed] = cluster_id
        queue = list(neighbors)

        while queue:
            q = queue.pop(0)
            if not visited[q]:
                visited[q] = True
                q_neighbors = self._neighbors(D, q)
                if len(q_neighbors) >= self.min_pts:
                    # q est un point cœur → expansion
                    queue.extend(q_neighbors.tolist())
            if labels[q] == -1:   # précédemment bruit → absorber
                labels[q] = cluster_id

    # ----------------------------------------------------------
    # Interface BaseClusterer
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Exécute DBSCAN (Algorithme 6 du document).

        Pour chaque point non visité :
          - si |Nε(x)| < MinPts → bruit
          - sinon → nouveau cluster par ExpandCluster
        """
        n = X.shape[0]
        D = euclidean(X, X)      # matrice n×n des distances

        labels = np.full(n, -1, dtype=int)    # −1 = bruit par défaut
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._neighbors(D, i)

            if len(neighbors) < self.min_pts:
                # Point bruit (peut être absorbé plus tard)
                labels[i] = -1
            else:
                # Point cœur → créer un nouveau cluster
                self._expand_cluster(D, labels, visited, i, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self.core_mask_ = np.array([
            len(self._neighbors(D, i)) >= self.min_pts
            for i in range(n)
        ])
        self.n_clusters_ = cluster_id
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN est inductif uniquement via les points cœurs."""
        self._check_fitted()
        return self.labels_

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """Met en valeur les points cœurs, bord et bruit."""
        self._check_fitted()
        labels = self.labels_
        unique_labels = np.unique(labels)
        colors = _palette(self.n_clusters_ + 1)
        is_3d = X.shape[1] >= 3
        fig = go.Figure()

        for idx, lbl in enumerate(unique_labels):
            mask = (labels == lbl) & self.core_mask_
            border_mask = (labels == lbl) & ~self.core_mask_

            if lbl == -1:
                color = "rgba(150,150,150,0.3)"
                name = "Bruit"
            else:
                color = colors[idx % len(colors)]
                name = f"Cluster {lbl}"

            # Points cœurs
            if mask.sum() > 0:
                if is_3d:
                    fig.add_trace(go.Scatter3d(
                        x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
                        mode="markers",
                        marker=dict(size=5, color=color, symbol="circle"),
                        name=name
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0], y=X[mask, 1],
                        mode="markers",
                        marker=dict(size=8, color=color,
                                    line=dict(width=1, color="black")),
                        name=name
                    ))

            # Points bord (plus petits)
            if border_mask.sum() > 0 and lbl != -1:
                if is_3d:
                    fig.add_trace(go.Scatter3d(
                        x=X[border_mask, 0], y=X[border_mask, 1], z=X[border_mask, 2],
                        mode="markers",
                        marker=dict(size=3, color=color, symbol="circle-open"),
                        name=f"Bord {lbl}", showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=X[border_mask, 0], y=X[border_mask, 1],
                        mode="markers",
                        marker=dict(size=5, color=color, symbol="circle-open"),
                        name=f"Bord {lbl}", showlegend=False
                    ))

        scene_kw = dict(scene=dict(xaxis_title="x₁", yaxis_title="x₂",
                                   zaxis_title="x₃")) if is_3d else {}
        fig.update_layout(
            title=dict(
                text=f"{self.name} {title_suffix}<br>"
                     f"<sup>ε={self.eps}, MinPts={self.min_pts}, "
                     f"Clusters={self.n_clusters_}</sup>",
                font=dict(size=15)
            ),
            template="plotly_white",
            margin=dict(l=40, r=20, t=80, b=40),
            **scene_kw
        )
        return fig