from init import *
from shared import _palette, euclidean_sq, euclidean
# ============================================================
# SECTION 0 — Classe de base abstraite
# ============================================================

class BaseClusterer(ABC):
    """
    Classe abstraite définissant l'interface commune
    à tous les algorithmes de clustering implémentés.

    Toute sous-classe doit implémenter fit() et predict().
    visualize() produit une figure Plotly.
    """

    def __init__(self, name: str):
        self.name = name
        self.labels_ = None          # étiquettes finales (ndarray int)
        self.n_clusters_ = None      # nombre de clusters trouvés
        self._is_fitted = False

    # ----------------------------------------------------------
    # Interface obligatoire
    # ----------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """
        Ajuste le modèle sur les données X (n_samples, n_features).
        Doit renseigner self.labels_ et self._is_fitted = True.
        Retourne self pour permettre le chaînage.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit l'étiquette de chaque point de X.
        Retourne un ndarray d'entiers (n_samples,).
        """

    # ----------------------------------------------------------
    # Méthodes communes (peuvent être surchargées)
    # ----------------------------------------------------------

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Raccourci fit + predict sur le même jeu."""
        return self.fit(X).predict(X)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} n'est pas encore ajusté. Appelez fit() d'abord.")

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """
        Génère une figure Plotly 2D ou 3D selon la dimension de X.
        Peut être surchargée pour un affichage spécifique à l'algorithme.
        """
        self._check_fitted()
        labels = self.labels_
        unique_labels = np.unique(labels)
        colors = _palette(len(unique_labels))

        is_3d = X.shape[1] >= 3
        fig = go.Figure()

        for idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            name = f"Bruit" if lbl == -1 else f"Cluster {lbl}"
            color = "rgba(150,150,150,0.4)" if lbl == -1 else colors[idx % len(colors)]

            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.85),
                    name=name
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=X[mask, 0], y=X[mask, 1],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.85),
                    name=name
                ))

        scene_kw = dict(
            scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃",
                bgcolor="rgba(0,0,0,0)"
            )
        ) if is_3d else {}

        fig.update_layout(
            title=dict(text=f"{self.name} {title_suffix}", font=dict(size=15)),
            legend=dict(itemsizing="constant"),
            margin=dict(l=40, r=20, t=60, b=40),
            template="plotly_white",
            **scene_kw
        )
        return fig