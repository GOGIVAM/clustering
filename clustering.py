"""
============================================================
CADEMI — Implémentation From Scratch : Théorie du Clustering
============================================================
M2R — Sciences des Données et Intelligence Artificielle
------------------------------------------------------------
Familles couvertes :
  1. Clustering Partitionnel   → KMeans (Lloyd + k-means++ init)
  2. Clustering Hiérarchique   → HAC (Ward linkage, Lance-Williams)
  3. Clustering Densité        → DBSCAN
  4. Modèles Probabilistes     → GMM-EM (Gaussian Mixture Model)
  5. Clustering Spectral       → SpectralClustering (Ng, Jordan & Weiss 2001)
  6. Deep Clustering           → DEC (Deep Embedded Clustering, Xie et al. 2016)

Dépendances : numpy, torch, plotly
Aucune fonction de clustering prédéfinie n'est utilisée.
------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from abc import ABC, abstractmethod


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


# ============================================================
# SECTION 2 — Clustering Hiérarchique : HAC (Ward)
# ============================================================

class HAC(BaseClusterer):
    """
    Hierarchical Agglomerative Clustering avec Lance-Williams.

    Liaisons disponibles : 'single', 'complete', 'average', 'ward'
    La liaison de Ward est l'implémentation par défaut (Proposition 2.1).

    Référence : Ward (1963) JASA ; Lance & Williams (1967)
    """

    LINKAGE_PARAMS = {
        # (alpha_a, alpha_b, beta, gamma) de Lance-Williams
        "single":   (0.5,  0.5,  0.0,  -0.5),
        "complete": (0.5,  0.5,  0.0,   0.5),
        "average":  (None, None, 0.0,   0.0),   # poids proportionnels à |C|
        "ward":     (None, None, None,  0.0),   # traitement spécial
    }

    def __init__(self, n_clusters: int = 2, linkage: str = "ward"):
        super().__init__(f"HAC ({linkage} linkage)")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.dendrogram_heights_ = []   # hauteurs de fusion
        self.merge_history_ = []        # liste (Ca, Cb) fusionnés

    # ----------------------------------------------------------
    # Calcul de la matrice de dissimilarité initiale
    # ----------------------------------------------------------

    def _init_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construit la matrice symétrique n×n des distances euclidiennes.
        Ward utilise les distances au carré (inertie).
        """
        if self.linkage == "ward":
            D = euclidean_sq(X, X).copy()
        else:
            D = euclidean(X, X).copy()
        np.fill_diagonal(D, np.inf)
        return D

    # ----------------------------------------------------------
    # Mise à jour Lance-Williams
    # ----------------------------------------------------------

    def _lance_williams(self, D: np.ndarray, sizes: dict,
                        a: int, b: int, ab: int) -> None:
        """
        Met à jour in-place la ligne/colonne `ab` dans D
        en appliquant la formule de Lance-Williams pour la liaison choisie.

        D[ab, r] = αa·D[a,r] + αb·D[b,r] + β·D[a,b] + γ|D[a,r]-D[b,r]|
        """
        n = D.shape[0]
        na, nb = sizes[a], sizes[b]

        for r in range(n):
            if r == a or r == b or r == ab:
                continue
            if D[a, r] == np.inf and D[b, r] == np.inf:
                continue

            if self.linkage == "single":
                alpha_a, alpha_b, beta, gamma = 0.5, 0.5, 0.0, -0.5
            elif self.linkage == "complete":
                alpha_a, alpha_b, beta, gamma = 0.5, 0.5, 0.0, 0.5
            elif self.linkage == "average":
                n_total = na + nb
                alpha_a = na / n_total
                alpha_b = nb / n_total
                beta, gamma = 0.0, 0.0
            elif self.linkage == "ward":
                # Formule Ward en distances au carré :
                # Δ(Cab, r) = (na+nr)/(na+nb+nr)·D[a,r]
                #           + (nb+nr)/(na+nb+nr)·D[b,r]
                #           - nr/(na+nb+nr)·D[a,b]
                nr = sizes.get(r, 1)
                n_total = na + nb + nr
                alpha_a = (na + nr) / n_total
                alpha_b = (nb + nr) / n_total
                beta    = -nr / n_total
                gamma   = 0.0
                D[ab, r] = D[r, ab] = (
                    alpha_a * D[a, r] +
                    alpha_b * D[b, r] +
                    beta    * D[a, b]
                )
                continue

            dab = D[a, b] if D[a, b] != np.inf else 0.0
            D[ab, r] = D[r, ab] = (
                alpha_a * D[a, r] +
                alpha_b * D[b, r] +
                beta * dab +
                gamma * abs(D[a, r] - D[b, r])
            )

    # ----------------------------------------------------------
    # Algorithme principal
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "HAC":
        """
        Exécute HAC et construit le dendrogramme.
        S'arrête quand le nombre de clusters cible est atteint.
        """
        n = X.shape[0]
        D = self._init_distance_matrix(X)

        # Chaque point est son propre cluster ; on mappe indice→liste de points
        clusters = {i: [i] for i in range(n)}
        sizes = {i: 1 for i in range(n)}
        labels = np.arange(n, dtype=int)    # étiquette courante par point
        active = set(range(n))
        next_id = n   # identifiant des nouveaux clusters fusionnés

        self.dendrogram_heights_ = []
        self.merge_history_ = []

        # La matrice D est de taille fixe n×n pour simplifier l'indexation.
        # On étend dynamiquement quand un nouveau cluster `ab` est créé.
        D_ext = np.full((2 * n, 2 * n), np.inf)
        D_ext[:n, :n] = D

        sizes_ext = sizes.copy()

        while len(active) > self.n_clusters:
            # Trouver les deux clusters actifs les plus proches
            active_list = list(active)
            best_dist = np.inf
            ca, cb = -1, -1
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    ii, jj = active_list[i], active_list[j]
                    if D_ext[ii, jj] < best_dist:
                        best_dist = D_ext[ii, jj]
                        ca, cb = ii, jj

            if ca == -1:
                break

            # Fusionner ca et cb en ab
            ab = next_id
            next_id += 1
            new_size = sizes_ext[ca] + sizes_ext[cb]
            sizes_ext[ab] = new_size

            # Étendre D_ext si nécessaire
            if ab >= D_ext.shape[0]:
                new_d = D_ext.shape[0] + n
                tmp = np.full((new_d, new_d), np.inf)
                tmp[:D_ext.shape[0], :D_ext.shape[0]] = D_ext
                D_ext = tmp

            D_ext[ab, :] = np.inf
            D_ext[:, ab] = np.inf

            # Copier les distances existantes de ca et cb vers ab
            for r in active:
                if r != ca and r != cb:
                    D_ext[ab, r] = D_ext[ca, r]
                    D_ext[r, ab] = D_ext[r, ca]

            # Mettre à jour par Lance-Williams
            self._lance_williams(D_ext, sizes_ext, ca, cb, ab)

            # Invalider ca et cb
            D_ext[ca, :] = np.inf
            D_ext[:, ca] = np.inf
            D_ext[cb, :] = np.inf
            D_ext[:, cb] = np.inf

            # Mettre à jour les étiquettes des points
            members_ca = clusters.get(ca, [ca])
            members_cb = clusters.get(cb, [cb])
            clusters[ab] = members_ca + members_cb

            self.merge_history_.append((ca, cb))
            self.dendrogram_heights_.append(float(best_dist))

            active.discard(ca)
            active.discard(cb)
            active.add(ab)

        # Construire les labels finaux
        self.labels_ = np.zeros(n, dtype=int)
        for cluster_id, label in zip(list(active), range(len(active))):
            for point_idx in clusters.get(cluster_id, [cluster_id]):
                self.labels_[point_idx] = label

        self.n_clusters_ = len(active)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Pour HAC, predict retourne les labels du fit (modèle inductif limité)."""
        self._check_fitted()
        return self.labels_

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """Affiche les clusters + historique des hauteurs de fusion."""
        scatter_fig = super().visualize(X, title_suffix)
        n_merges = len(self.dendrogram_heights_)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Clusters HAC", "Hauteurs de fusion"),
            column_widths=[0.65, 0.35]
        )

        # Panneau gauche : scatter des clusters
        for trace in scatter_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Panneau droit : courbe des hauteurs de fusion (dendrogramme simplifié)
        if n_merges > 0:
            fig.add_trace(go.Scatter(
                x=list(range(1, n_merges + 1)),
                y=self.dendrogram_heights_,
                mode="lines+markers",
                marker=dict(size=5, color="rgba(55,138,221,0.85)"),
                line=dict(width=2, color="rgba(55,138,221,0.85)"),
                name="Hauteur fusion"
            ), row=1, col=2)

        fig.update_xaxes(title_text="Étape de fusion", row=1, col=2)
        fig.update_yaxes(title_text="Distance (Ward)", row=1, col=2)
        fig.update_layout(
            title=dict(text=f"{self.name} {title_suffix}", font=dict(size=15)),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=20, t=80, b=40)
        )
        return fig


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


# ============================================================
# SECTION 4 — Modèles Probabilistes : GMM-EM
# ============================================================

class GaussianMixtureEM(BaseClusterer):
    """
    Modèle de Mélange Gaussien estimé par l'algorithme EM.

    Modèle génératif :
        p(x) = Σ_k π_k · N(x | µ_k, Σ_k)

    Algorithme EM :
        E-step : γ_ik = π_k N(xi|µk,Σk) / Σ_j π_j N(xi|µj,Σj)
        M-step : mise à jour de πk, µk, Σk

    Référence : Dempster, Laird & Rubin (1977) — JRSS-B
    """

    def __init__(self, k: int, max_iter: int = 200,
                 tol: float = 1e-6, reg_cov: float = 1e-6,
                 random_state: int = 42):
        super().__init__("GMM-EM")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.reg_cov = reg_cov        # régularisation numérique de Σ
        self.rng = np.random.default_rng(random_state)

        self.pi_ = None               # (k,) poids des composantes
        self.mu_ = None               # (k, d) moyennes
        self.sigma_ = None            # (k, d, d) matrices de covariance
        self.log_likelihood_ = None   # historique de la log-vraisemblance
        self.n_iter_ = None

    # ----------------------------------------------------------
    # Évaluation de la log-densité gaussienne multivariée
    # ----------------------------------------------------------

    def _log_gaussian(self, X: np.ndarray, mu: np.ndarray,
                      sigma: np.ndarray) -> np.ndarray:
        """
        Calcule log N(X | mu, sigma) pour chaque ligne de X.
        Retourne (n,).

        Formule : -d/2 · log(2π) - 1/2 · log|Σ| - 1/2 (x-µ)ᵀΣ⁻¹(x-µ)
        """
        d = X.shape[1]
        diff = X - mu                     # (n, d)

        try:
            L = np.linalg.cholesky(sigma + self.reg_cov * np.eye(d))
            # Résoudre L · v = diff.T → v = L⁻¹ · diff.T
            v = np.linalg.solve(L, diff.T)   # (d, n)
            mah_sq = np.sum(v ** 2, axis=0)  # (n,)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            sigma_reg = sigma + self.reg_cov * np.eye(d)
            sigma_inv = np.linalg.inv(sigma_reg)
            mah_sq = np.einsum("ni,ij,nj->n", diff, sigma_inv, diff)
            sign, log_det = np.linalg.slogdet(sigma_reg)
            if sign <= 0:
                log_det = 0.0

        return -0.5 * (d * np.log(2 * np.pi) + log_det + mah_sq)

    # ----------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------

    def _initialize(self, X: np.ndarray) -> None:
        """
        Initialise les paramètres θ⁽⁰⁾ par k-means (1 itération)
        pour garantir une bonne répartition initiale.
        """
        n, d = X.shape
        # Initialisation des µ par tirage aléatoire dans X
        idx = self.rng.choice(n, self.k, replace=False)
        self.mu_ = X[idx].copy()
        self.sigma_ = np.array([np.eye(d) for _ in range(self.k)])
        self.pi_ = np.ones(self.k) / self.k

    # ----------------------------------------------------------
    # E-step
    # ----------------------------------------------------------

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les responsabilités γ_ik (log-espace pour stabilité).

        γ_ik = π_k N(xi|µk,Σk) / Σ_j π_j N(xi|µj,Σj)

        Retourne : gamma de forme (n, k)
        """
        n = X.shape[0]
        log_resp = np.zeros((n, self.k))

        for k in range(self.k):
            log_resp[:, k] = np.log(self.pi_[k] + 1e-300) + \
                             self._log_gaussian(X, self.mu_[k], self.sigma_[k])

        # Normalisation log-sum-exp (stabilité numérique)
        log_resp_max = log_resp.max(axis=1, keepdims=True)
        log_resp -= log_resp_max
        resp = np.exp(log_resp)
        resp_sum = resp.sum(axis=1, keepdims=True)
        gamma = resp / np.maximum(resp_sum, 1e-300)
        return gamma

    # ----------------------------------------------------------
    # M-step
    # ----------------------------------------------------------

    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """
        Met à jour les paramètres {πk, µk, Σk} via les responsabilités.

        Nk = Σ_i γ_ik
        µk = (1/Nk) Σ_i γ_ik xi
        Σk = (1/Nk) Σ_i γ_ik (xi - µk)(xi - µk)ᵀ
        πk = Nk / n
        """
        n, d = X.shape
        Nk = gamma.sum(axis=0)                          # (k,)

        self.pi_ = Nk / n

        for k in range(self.k):
            Nk_k = max(Nk[k], 1e-300)
            self.mu_[k] = (gamma[:, k:k+1] * X).sum(axis=0) / Nk_k
            diff = X - self.mu_[k]                      # (n, d)
            self.sigma_[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk_k
            self.sigma_[k] += self.reg_cov * np.eye(d)  # régularisation

    # ----------------------------------------------------------
    # Log-vraisemblance incomplète
    # ----------------------------------------------------------

    def _log_likelihood(self, X: np.ndarray) -> float:
        """
        ℓ(θ) = Σ_i log Σ_k πk N(xi | µk, Σk)
        Calculé en log-espace pour la stabilité numérique.
        """
        n = X.shape[0]
        log_probs = np.zeros((n, self.k))
        for k in range(self.k):
            log_probs[:, k] = np.log(self.pi_[k] + 1e-300) + \
                              self._log_gaussian(X, self.mu_[k], self.sigma_[k])
        max_log = log_probs.max(axis=1)
        log_sum = max_log + np.log(np.exp(log_probs - max_log[:, None]).sum(axis=1))
        return float(log_sum.sum())

    # ----------------------------------------------------------
    # Interface BaseClusterer
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        """
        Exécute l'algorithme EM jusqu'à convergence de la log-vraisemblance.
        Monotonie garantie par le Théorème 4.1 (ELBO).
        """
        self._initialize(X)
        self.log_likelihood_ = []
        prev_ll = -np.inf

        for t in range(self.max_iter):
            gamma = self._e_step(X)       # E-step
            self._m_step(X, gamma)        # M-step
            ll = self._log_likelihood(X)
            self.log_likelihood_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                self.n_iter_ = t + 1
                break
            prev_ll = ll
        else:
            self.n_iter_ = self.max_iter

        # Assignation finale : argmax des responsabilités
        gamma_final = self._e_step(X)
        self.labels_ = gamma_final.argmax(axis=1)
        self.n_clusters_ = self.k
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigne chaque point à la composante de plus forte responsabilité."""
        self._check_fitted()
        gamma = self._e_step(X)
        return gamma.argmax(axis=1)

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """Affiche les clusters + ellipses de confiance (2D) + courbe EM."""
        fig_base = super().visualize(X, title_suffix)

        if X.shape[1] == 2:
            # Ellipses de confiance à 2 sigma pour chaque composante
            t = np.linspace(0, 2 * np.pi, 100)
            circle = np.column_stack([np.cos(t), np.sin(t)])
            colors = _palette(self.k)

            for k in range(self.k):
                try:
                    eigvals, eigvecs = np.linalg.eigh(self.sigma_[k])
                    eigvals = np.maximum(eigvals, 0)
                    scale = 2.0 * np.sqrt(eigvals)
                    ellipse = circle @ (eigvecs * scale).T + self.mu_[k]
                    fig_base.add_trace(go.Scatter(
                        x=ellipse[:, 0], y=ellipse[:, 1],
                        mode="lines",
                        line=dict(width=2, color=colors[k % len(colors)],
                                  dash="dash"),
                        name=f"σ₂ comp {k}",
                        showlegend=False
                    ))
                except Exception:
                    pass

        # Panneau log-vraisemblance
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Clusters GMM + ellipses 2σ", "Convergence EM"),
            column_widths=[0.65, 0.35]
        )
        for trace in fig_base.data:
            fig.add_trace(trace, row=1, col=1)

        if self.log_likelihood_:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(self.log_likelihood_) + 1)),
                y=self.log_likelihood_,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color="rgba(172,83,172,0.85)", width=2),
                name="Log-vraisemblance"
            ), row=1, col=2)
            fig.update_xaxes(title_text="Itération EM", row=1, col=2)
            fig.update_yaxes(title_text="ℓ(θ)", row=1, col=2)

        fig.update_layout(
            title=dict(text=f"{self.name} {title_suffix}<br>"
                            f"<sup>K={self.k}, itérations={self.n_iter_}</sup>",
                       font=dict(size=15)),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=20, t=80, b=40)
        )
        return fig


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
        """Affiche l'espace original et l'espace spectral côte à côte."""
        scatter_fig = super().visualize(X, title_suffix)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Espace original", "Espace spectral (2 premières dims)"),
            column_widths=[0.5, 0.5]
        )

        for trace in scatter_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Visualisation de l'espace spectral
        colors = _palette(self.k)
        U = self.embedding_
        for lbl in range(self.k):
            mask = self.labels_ == lbl
            if U.shape[1] >= 2:
                fig.add_trace(go.Scatter(
                    x=U[mask, 0], y=U[mask, 1],
                    mode="markers",
                    marker=dict(size=6, color=colors[lbl % len(colors)]),
                    name=f"Spectral {lbl}",
                    showlegend=False
                ), row=1, col=2)

        fig.update_xaxes(title_text="u₁", row=1, col=2)
        fig.update_yaxes(title_text="u₂", row=1, col=2)
        fig.update_layout(
            title=dict(
                text=f"{self.name} {title_suffix}<br>"
                     f"<sup>k={self.k}, σ={self.sigma}, "
                     f"affinité={self.affinity}</sup>",
                font=dict(size=15)
            ),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=20, t=80, b=40)
        )
        return fig


# ============================================================
# SECTION 6 — Deep Clustering : DEC (PyTorch)
# ============================================================

class Autoencoder(nn.Module):
    """
    Auto-encodeur entièrement connecté pour la pré-initialisation de DEC.

    Architecture : Encodeur → code latent → Décodeur
    Toutes les couches utilisent BatchNorm + ReLU sauf la sortie.
    """

    def __init__(self, input_dim: int, latent_dim: int = 10,
                 hidden_dims: list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # ---- Encodeur ------------------------------------------
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ]
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ---- Décodeur ------------------------------------------
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ]
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DEC(BaseClusterer):
    """
    Deep Embedded Clustering (Xie, Girshick & Farhadi, ICML 2016).

    Processus en deux phases :
      Phase 1 — Pré-entraînement de l'auto-encodeur (reconstruction L2)
      Phase 2 — Affinage conjoint des représentations et centres de clusters
                 via la divergence KL entre distribution douce q et cible p

    Distribution douce (Student-t, ν=1) :
        q_ik ∝ (1 + ||z_i - µ_k||² / ν)^{-(ν+1)/2}

    Distribution cible :
        p_ik = q_ik² / Σ_i q_ik  (normalisé)

    Perte : L_DEC = KL(P || Q) = Σ_i Σ_k p_ik log(p_ik / q_ik)
    """

    def __init__(self, k: int, latent_dim: int = 10,
                 hidden_dims: list = None,
                 pretrain_epochs: int = 50,
                 cluster_epochs: int = 100,
                 lr_pretrain: float = 1e-3,
                 lr_cluster: float = 1e-4,
                 batch_size: int = 256,
                 nu: float = 1.0,
                 random_state: int = 42):
        super().__init__("DEC (Deep Embedded Clustering)")
        self.k = k
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.pretrain_epochs = pretrain_epochs
        self.cluster_epochs = cluster_epochs
        self.lr_pretrain = lr_pretrain
        self.lr_cluster = lr_cluster
        self.batch_size = batch_size
        self.nu = nu                  # degré de liberté Student-t

        torch.manual_seed(random_state)
        self.device = torch.device("cpu")

        self.autoencoder = None
        self.cluster_centers_ = None  # (k, latent_dim) tensor
        self.pretrain_losses_ = []
        self.cluster_losses_ = []

    # ----------------------------------------------------------
    # Phase 1 : pré-entraînement de l'auto-encodeur
    # ----------------------------------------------------------

    def _pretrain(self, X_tensor: torch.Tensor) -> None:
        """
        Minimise la perte de reconstruction MSE :
        L_recon = ||xi - x̂i||²
        """
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr_pretrain)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        self.pretrain_losses_ = []

        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                z, x_hat = self.autoencoder(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            self.pretrain_losses_.append(epoch_loss / len(X_tensor))

    # ----------------------------------------------------------
    # Distribution douce d'assignation (Student-t)
    # ----------------------------------------------------------

    def _soft_assignment(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Calcule Q = q_ik (distribution Student-t).

        q_ik = (1 + ||z_i - µ_k||² / ν)^{-(ν+1)/2}
               --------------------------------
               Σ_j (1 + ||z_i - µ_j||² / ν)^{-(ν+1)/2}

        La distribution Student-t à queue lourde améliore la robustesse
        aux outliers par rapport à l'assignation gaussienne.

        Paramètres
        ----------
        Z : (n, latent_dim)
        Retourne : Q de forme (n, k)
        """
        # ||z_i - µ_k||² pour tout (i, k)
        diff = Z.unsqueeze(1) - self.cluster_centers_.unsqueeze(0)  # (n, k, d)
        dist_sq = (diff ** 2).sum(dim=-1)                           # (n, k)

        # Noyau Student-t
        numerator = (1.0 + dist_sq / self.nu) ** (-(self.nu + 1.0) / 2.0)
        Q = numerator / numerator.sum(dim=1, keepdim=True)
        return Q

    # ----------------------------------------------------------
    # Distribution cible
    # ----------------------------------------------------------

    def _target_distribution(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Distribution cible P amplifiant les assignations de haute confiance.

        p_ik = (q_ik² / f_k) / Σ_j (q_ij² / f_j)
        où f_k = Σ_i q_ik  (fréquences douces)

        Cette distribution « durcit » Q, renforçant les clusters
        bien séparés et pénalisant les assignations ambiguës.
        """
        weight = Q ** 2 / Q.sum(dim=0, keepdim=True)
        P = weight / weight.sum(dim=1, keepdim=True)
        return P

    # ----------------------------------------------------------
    # Phase 2 : affinage par minimisation de KL(P || Q)
    # ----------------------------------------------------------

    def _cluster_phase(self, X_tensor: torch.Tensor) -> None:
        """
        Optimise conjointement θ (auto-encodeur) et {µk} (centres).
        Perte : L_DEC = KL(P || Q) = Σ_i Σ_k p_ik log(p_ik / q_ik)
        """
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        optimizer = optim.Adam(
            list(self.autoencoder.parameters()) + [self.cluster_centers_],
            lr=self.lr_cluster
        )

        self.autoencoder.train()
        self.cluster_losses_ = []

        for epoch in range(self.cluster_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                Z = self.autoencoder.encode(batch)
                Q = self._soft_assignment(Z)
                P = self._target_distribution(Q).detach()

                # KL(P || Q) = Σ p log(p/q)
                loss = (P * (torch.log(P + 1e-10) - torch.log(Q + 1e-10))).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            self.cluster_losses_.append(epoch_loss / len(X_tensor))

    # ----------------------------------------------------------
    # Interface BaseClusterer
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DEC":
        """
        Exécute DEC en deux phases :
          1. Pré-entraînement de l'auto-encodeur
          2. Initialisation des centres par k-means sur l'espace latent
          3. Affinage par KL(P || Q)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        input_dim = X.shape[1]

        # Construire l'auto-encodeur
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Phase 1 : pré-entraînement
        self._pretrain(X_tensor)

        # Initialiser les centres par k-means sur les embeddings
        self.autoencoder.eval()
        with torch.no_grad():
            Z_init = self.autoencoder.encode(X_tensor).cpu().numpy()

        km = KMeans(k=self.k, n_init=5, random_state=42)
        km.fit(Z_init)
        centers_init = km.centroids_

        self.cluster_centers_ = nn.Parameter(
            torch.FloatTensor(centers_init).to(self.device)
        )

        # Phase 2 : affinage DEC
        self._cluster_phase(X_tensor)

        # Labels finaux
        self.autoencoder.eval()
        with torch.no_grad():
            Z_final = self.autoencoder.encode(X_tensor)
            Q_final = self._soft_assignment(Z_final)
            self.labels_ = Q_final.argmax(dim=1).cpu().numpy()

        self.n_clusters_ = self.k
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Encode X et assigne au cluster de plus forte responsabilité q."""
        self._check_fitted()
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.autoencoder.eval()
        with torch.no_grad():
            Z = self.autoencoder.encode(X_tensor)
            Q = self._soft_assignment(Z)
        return Q.argmax(dim=1).cpu().numpy()

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """Retourne l'espace latent Z = fθ(X)."""
        self._check_fitted()
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.autoencoder.eval()
        with torch.no_grad():
            Z = self.autoencoder.encode(X_tensor)
        return Z.cpu().numpy()

    def visualize(self, X: np.ndarray, title_suffix: str = "") -> go.Figure:
        """Affiche clusters en espace latent + courbes de perte."""
        self._check_fitted()
        Z = self.get_embedding(X)
        colors = _palette(self.k)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Espace original",
                "Espace latent (2D)",
                "Pertes d'entraînement"
            ),
            column_widths=[0.33, 0.33, 0.34]
        )

        # Panneau 1 : espace original (2 premières dims)
        for lbl in range(self.k):
            mask = self.labels_ == lbl
            color = colors[lbl % len(colors)]
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.7),
                name=f"Cluster {lbl}"
            ), row=1, col=1)

        # Panneau 2 : espace latent (2 premières dimensions)
        for lbl in range(self.k):
            mask = self.labels_ == lbl
            color = colors[lbl % len(colors)]
            z_x = Z[mask, 0] if Z.shape[1] > 0 else np.zeros(mask.sum())
            z_y = Z[mask, 1] if Z.shape[1] > 1 else np.zeros(mask.sum())
            fig.add_trace(go.Scatter(
                x=z_x, y=z_y,
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.7),
                name=f"Latent {lbl}",
                showlegend=False
            ), row=1, col=2)

        # Centres dans l'espace latent
        centers = self.cluster_centers_.detach().cpu().numpy()
        for lbl in range(self.k):
            fig.add_trace(go.Scatter(
                x=[centers[lbl, 0]],
                y=[centers[lbl, 1]] if centers.shape[1] > 1 else [0],
                mode="markers",
                marker=dict(symbol="star", size=14,
                            color=colors[lbl % len(colors)],
                            line=dict(width=1.5, color="black")),
                showlegend=False
            ), row=1, col=2)

        # Panneau 3 : courbes de perte
        if self.pretrain_losses_:
            fig.add_trace(go.Scatter(
                y=self.pretrain_losses_,
                mode="lines",
                line=dict(color="rgba(55,138,221,0.85)", width=2),
                name="MSE (pré-entraînement)"
            ), row=1, col=3)

        if self.cluster_losses_:
            fig.add_trace(go.Scatter(
                y=self.cluster_losses_,
                mode="lines",
                line=dict(color="rgba(172,83,172,0.85)", width=2),
                name="KL (DEC)"
            ), row=1, col=3)

        fig.update_xaxes(title_text="x₁", row=1, col=1)
        fig.update_yaxes(title_text="x₂", row=1, col=1)
        fig.update_xaxes(title_text="z₁", row=1, col=2)
        fig.update_yaxes(title_text="z₂", row=1, col=2)
        fig.update_xaxes(title_text="Époque", row=1, col=3)
        fig.update_yaxes(title_text="Perte", row=1, col=3)

        fig.update_layout(
            title=dict(
                text=f"{self.name} {title_suffix}<br>"
                     f"<sup>k={self.k}, latent_dim={self.latent_dim}</sup>",
                font=dict(size=15)
            ),
            template="plotly_white",
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=40, r=20, t=80, b=80)
        )
        return fig


# ============================================================
# SECTION 7 — Génération de données synthétiques
# ============================================================

def make_blobs_3d(n_samples: int = 300, n_clusters: int = 4,
                  std: float = 0.6, random_state: int = 42) -> np.ndarray:
    """Clusters gaussiens isotropes en 3D."""
    rng = np.random.default_rng(random_state)
    centers = rng.uniform(-5, 5, (n_clusters, 3))
    X = np.vstack([
        rng.normal(c, std, (n_samples // n_clusters, 3))
        for c in centers
    ])
    return X


def make_moons_3d(n_samples: int = 300, noise: float = 0.1,
                  random_state: int = 42) -> np.ndarray:
    """
    Deux croissants en 3D (extension de make_moons).
    Idéal pour démontrer DBSCAN face à k-means.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = rng.uniform(0, np.pi, n)
    X1 = np.column_stack([np.cos(theta), np.sin(theta),
                           rng.normal(0, noise, n)])
    X2 = np.column_stack([1 - np.cos(theta), 1 - np.sin(theta),
                           rng.normal(0, noise, n)])
    return np.vstack([X1, X2])


def make_rings_3d(n_samples: int = 400, noise: float = 0.05,
                  random_state: int = 42) -> np.ndarray:
    """
    Deux anneaux concentriques en 3D.
    Démonstre les limites de k-means et l'avantage du clustering spectral.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = rng.uniform(0, 2 * np.pi, n)

    r1 = 1.0 + rng.normal(0, noise, n)
    X1 = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta),
                           rng.normal(0, noise, n)])

    r2 = 3.0 + rng.normal(0, noise, n)
    X2 = np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta),
                           rng.normal(0, noise, n)])

    return np.vstack([X1, X2])


def make_gmm_3d(n_samples: int = 400, random_state: int = 42) -> np.ndarray:
    """
    Mélange gaussien 3D avec composantes de formes ellipsoïdales variées.
    Illustre les capacités du GMM-EM (covariances non sphériques).
    """
    rng = np.random.default_rng(random_state)
    components = [
        (np.array([0, 0, 0]),
         np.diag([1.5, 0.3, 0.3]), 100),
        (np.array([5, 5, 0]),
         np.diag([0.4, 1.5, 0.4]), 100),
        (np.array([0, 5, 5]),
         np.diag([0.3, 0.3, 1.5]), 100),
        (np.array([5, 0, 5]),
         np.array([[0.8, 0.5, 0.2],
                    [0.5, 0.8, 0.3],
                    [0.2, 0.3, 0.8]]), 100),
    ]
    Xs = []
    for mu, sigma, n in components:
        L = np.linalg.cholesky(sigma)
        Xs.append(rng.standard_normal((n, 3)) @ L.T + mu)
    return np.vstack(Xs)


# ============================================================
# SECTION 8 — Application principale
# ============================================================

def run_all():
    """
    Point d'entrée principal.
    Lance les 6 algorithmes sur des jeux de données adaptés
    et affiche les résultats dans des figures Plotly interactives.
    """

    print("=" * 65)
    print("CADEMI — Théorie du Clustering : Implémentation from scratch")
    print("=" * 65)

    # ----------------------------------------------------------
    # 1. K-MEANS sur blobs 3D
    # ----------------------------------------------------------
    print("\n[1/6] K-Means (Lloyd + k-means++) — Blobs 3D")
    X_blobs = make_blobs_3d(n_samples=400, n_clusters=4)
    kmeans = KMeans(k=4, n_init=10, random_state=0)
    kmeans.fit(X_blobs)
    fig1 = kmeans.visualize(X_blobs, "— Blobs 3D")
    print(f"    Inertie finale  : {kmeans.inertia_:.4f}")
    print(f"    Itérations (meilleure run) : {kmeans.n_iter_}")
    fig1.show()

    # ----------------------------------------------------------
    # 2. HAC (Ward) sur blobs 2D (dendrogramme en 2D suffit)
    # ----------------------------------------------------------
    print("\n[2/6] HAC Ward — Blobs 2D")
    rng = np.random.default_rng(0)
    X_hac = np.vstack([
        rng.normal([0, 0], 0.5, (80, 2)),
        rng.normal([4, 0], 0.5, (80, 2)),
        rng.normal([2, 3.5], 0.5, (80, 2)),
    ])
    hac = HAC(n_clusters=3, linkage="ward")
    hac.fit(X_hac)
    fig2 = hac.visualize(X_hac, "— Blobs 2D")
    print(f"    Clusters trouvés : {hac.n_clusters_}")
    print(f"    Fusions effectuées : {len(hac.merge_history_)}")
    fig2.show()

    # ----------------------------------------------------------
    # 3. DBSCAN sur croissants 3D (formes non-convexes)
    # ----------------------------------------------------------
    print("\n[3/6] DBSCAN — Croissants 3D (non-convexes)")
    X_moons = make_moons_3d(n_samples=400, noise=0.08)
    dbscan = DBSCAN(eps=0.3, min_pts=6)
    dbscan.fit(X_moons)
    fig3 = dbscan.visualize(X_moons, "— Croissants 3D")
    n_noise = (dbscan.labels_ == -1).sum()
    print(f"    Clusters détectés : {dbscan.n_clusters_}")
    print(f"    Points bruit      : {n_noise}")
    fig3.show()

    # ----------------------------------------------------------
    # 4. GMM-EM sur gaussiennes ellipsoïdales 3D
    # ----------------------------------------------------------
    print("\n[4/6] GMM-EM — Gaussiennes ellipsoïdales 3D")
    X_gmm = make_gmm_3d(n_samples=400)
    gmm = GaussianMixtureEM(k=4, max_iter=200, random_state=7)
    gmm.fit(X_gmm)
    fig4 = gmm.visualize(X_gmm, "— Gaussiennes 3D")
    print(f"    Itérations EM       : {gmm.n_iter_}")
    print(f"    Log-vraisemblance   : {gmm.log_likelihood_[-1]:.4f}")
    print(f"    Poids (πk)          : {np.round(gmm.pi_, 3)}")
    fig4.show()

    # ----------------------------------------------------------
    # 5. Clustering Spectral sur anneaux 3D (non-connexes)
    # ----------------------------------------------------------
    print("\n[5/6] Clustering Spectral (NJW) — Anneaux 3D")
    X_rings = make_rings_3d(n_samples=400, noise=0.05)
    spectral = SpectralClustering(k=2, sigma=0.5, affinity="rbf")
    spectral.fit(X_rings)
    fig5 = spectral.visualize(X_rings, "— Anneaux 3D")
    print(f"    Clusters trouvés      : {spectral.n_clusters_}")
    print(f"    Valeurs propres top-k : {np.round(spectral.eigenvalues_, 4)}")
    fig5.show()

    # ----------------------------------------------------------
    # 6. DEC sur blobs 2D (démo avec petit réseau)
    # ----------------------------------------------------------
    print("\n[6/6] DEC (Deep Embedded Clustering) — Blobs 2D")
    rng2 = np.random.default_rng(99)
    X_dec = np.vstack([
        rng2.normal([0, 0], 0.8, (150, 2)),
        rng2.normal([5, 0], 0.8, (150, 2)),
        rng2.normal([2.5, 4], 0.8, (150, 2)),
    ]).astype(np.float32)

    dec = DEC(
        k=3,
        latent_dim=8,
        hidden_dims=[64, 32],
        pretrain_epochs=80,
        cluster_epochs=80,
        lr_pretrain=1e-3,
        lr_cluster=1e-4,
        batch_size=64,
        random_state=42
    )
    dec.fit(X_dec)
    fig6 = dec.visualize(X_dec, "— Blobs 2D")
    print(f"    Clusters trouvés        : {dec.n_clusters_}")
    print(f"    Perte pré-entraînement  : {dec.pretrain_losses_[-1]:.6f}")
    print(f"    Perte DEC (KL) finale   : {dec.cluster_losses_[-1]:.6f}")
    fig6.show()

    print("\n" + "=" * 65)
    print("Toutes les visualisations ont été générées.")
    print("=" * 65)


if __name__ == "__main__":
    run_all()