from init import *
from shared import _palette, euclidean_sq, euclidean
from base_cluster import BaseClusterer
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
        """
        Panneau gauche  : scatter 2D ou 3D des clusters.
        Panneau droit   : courbe des hauteurs de fusion (dendrogramme simplifié).
        """
        self._check_fitted()
        labels = self.labels_
        unique_labels = np.unique(labels)
        colors = _palette(len(unique_labels))
        is_3d = X.shape[1] >= 3
        n_merges = len(self.dendrogram_heights_)

        if is_3d:
            specs = [[{"type": "scene"}, {"type": "xy"}]]
        else:
            specs = [[{"type": "xy"}, {"type": "xy"}]]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Clusters HAC", "Hauteurs de fusion"),
            column_widths=[0.65, 0.35],
            specs=specs
        )

        # ---- Panneau gauche --------------------------------------
        for idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            name = f"Cluster {lbl}"
            color = colors[idx % len(colors)]

            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.85),
                    name=name
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=X[mask, 0], y=X[mask, 1],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.85),
                    name=name
                ), row=1, col=1)

        # ---- Panneau droit : hauteurs de fusion ------------------
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

        if is_3d:
            fig.update_layout(scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃"
            ))

        fig.update_layout(
            title=dict(text=f"{self.name} {title_suffix}", font=dict(size=15)),
            template="plotly_white",
            showlegend=True,
            margin=dict(l=40, r=20, t=80, b=40)
        )
        return fig