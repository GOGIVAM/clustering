from init import *
from shared import euclidean




# ============================================================
# SECTION 9 — Évaluation comparative (indices internes)
# Chapitre 7 du document CADEMI
# ============================================================

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Indice de Silhouette global (Définition 7.1).

    Pour chaque point xi :
      a(i) = distance intra-cluster moyenne
      b(i) = plus petite distance moyenne vers un autre cluster
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Score global : s̄ = (1/n) Σ s(i)
    Plage : [-1, 1]. Plus proche de 1 = meilleur clustering.

    Complexité : O(n²) — calcul exact sans approximation.
    """
    unique = np.unique(labels)
    # Cas dégénérés
    if len(unique) < 2 or len(unique) >= len(X):
        return 0.0

    n = X.shape[0]
    D = euclidean(X, X)   # (n, n) matrice de distances
    s_vals = np.zeros(n)

    for i in range(n):
        lbl_i = labels[i]
        intra_mask = (labels == lbl_i)
        intra_mask[i] = False   # exclure le point lui-même

        if intra_mask.sum() == 0:
            s_vals[i] = 0.0
            continue

        a_i = D[i, intra_mask].mean()

        # Distance moyenne vers chaque autre cluster
        b_candidates = []
        for lbl_k in unique:
            if lbl_k == lbl_i or lbl_k == -1:
                continue
            other_mask = labels == lbl_k
            if other_mask.sum() == 0:
                continue
            b_candidates.append(D[i, other_mask].mean())

        if not b_candidates:
            s_vals[i] = 0.0
            continue

        b_i = min(b_candidates)
        denom = max(a_i, b_i)
        s_vals[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0

    return float(s_vals.mean())


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Indice de Davies-Bouldin (Définition 7.2).

    DB = (1/k) Σ_i max_{j≠i} (σ_i + σ_j) / d(µ_i, µ_j)

    où σ_i = dispersion intra-cluster i (distance moyenne au centroïde).
    À minimiser : plus proche de 0 = meilleur clustering.
    """
    unique = np.array([l for l in np.unique(labels) if l != -1])
    k = len(unique)
    if k < 2:
        return float("inf")

    # Centroïdes et dispersions
    centroids = np.array([X[labels == l].mean(axis=0) for l in unique])
    dispersions = np.array([
        euclidean(X[labels == l], centroids[i:i+1]).mean()
        for i, l in enumerate(unique)
    ])

    db_sum = 0.0
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            d_ij = float(euclidean(centroids[i:i+1], centroids[j:j+1])[0, 0])
            if d_ij < 1e-12:
                continue
            ratios.append((dispersions[i] + dispersions[j]) / d_ij)
        db_sum += max(ratios) if ratios else 0.0

    return db_sum / k


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Indice de Calinski-Harabász (Définition 7.3).

    CH = [tr(Bk) / (k-1)] / [tr(Wk) / (n-k)]

    Bk = dispersion inter-cluster, Wk = dispersion intra-cluster.
    À maximiser : valeur élevée = clusters bien séparés et compacts.
    """
    unique = np.array([l for l in np.unique(labels) if l != -1])
    k = len(unique)
    n = X.shape[0]
    if k < 2 or k >= n:
        return 0.0

    grand_mean = X.mean(axis=0)

    # Trace de Bk (inter-cluster)
    tr_Bk = 0.0
    for l in unique:
        mask = labels == l
        nk = mask.sum()
        mu_k = X[mask].mean(axis=0)
        diff = mu_k - grand_mean
        tr_Bk += nk * float(diff @ diff)

    # Trace de Wk (intra-cluster)
    tr_Wk = 0.0
    for l in unique:
        mask = labels == l
        mu_k = X[mask].mean(axis=0)
        diff = X[mask] - mu_k
        tr_Wk += float((diff ** 2).sum())

    if tr_Wk < 1e-12:
        return float("inf")

    return (tr_Bk / (k - 1)) / (tr_Wk / (n - k))


def evaluate_all(results: list) -> go.Figure:
    """
    Calcule et affiche les 3 indices internes pour chaque algorithme.

    Paramètres
    ----------
    results : liste de dicts avec clés :
        'name'   : str   — nom de l'algorithme
        'X'      : ndarray (n, d) — données
        'labels' : ndarray (n,)  — étiquettes produites

    Retourne une figure Plotly avec 3 barplots côte à côte.
    """
    print("\n" + "=" * 65)
    print("ÉVALUATION COMPARATIVE — Indices internes (Chapitre 7)")
    print("=" * 65)
    print(f"{'Algorithme':<35} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-H':>12}")
    print("-" * 75)

    names, sil_vals, db_vals, ch_vals = [], [], [], []

    for r in results:
        name  = r["name"]
        X     = r["X"]
        labels = r["labels"]

        # Exclure les points bruit (-1) du calcul des indices
        valid = labels != -1
        X_v, l_v = X[valid], labels[valid]

        if len(np.unique(l_v)) < 2:
            sil, db, ch = 0.0, float("inf"), 0.0
        else:
            sil = silhouette_score(X_v, l_v)
            db  = davies_bouldin_score(X_v, l_v)
            ch  = calinski_harabasz_score(X_v, l_v)

        names.append(name)
        sil_vals.append(round(sil, 4))
        db_vals.append(round(db, 4))
        ch_vals.append(round(ch, 2))

        print(f"{name:<35} {sil:>12.4f} {db:>16.4f} {ch:>12.2f}")

    print("-" * 75)
    print("  Silhouette  : ↑ max ([-1, 1])   — meilleur proche de 1")
    print("  Davies-B.   : ↓ min ([0, +∞))   — meilleur proche de 0")
    print("  Calinski-H  : ↑ max ([0, +∞))   — meilleur = valeur élevée")

    # ---- Figure Plotly : 3 barplots --------------------------------
    colors_bar = [
        "rgba(55,138,221,0.85)",    # bleu    — KMeans
        "rgba(29,158,117,0.85)",    # teal    — HAC
        "rgba(211,83,46,0.85)",     # corail  — DBSCAN
        "rgba(172,83,172,0.85)",    # violet  — GMM
        "rgba(230,160,40,0.85)",    # ambre   — Spectral
        "rgba(200,60,80,0.85)",     # rouge   — DEC
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Silhouette  ↑ (max = 1)",
            "Davies-Bouldin  ↓ (min = 0)",
            "Calinski-Harabász  ↑"
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )

    bar_kw = dict(
        x=names,
        marker_color=colors_bar[:len(names)],
        text=[str(v) for v in sil_vals],
        textposition="outside"
    )

    # Silhouette
    fig.add_trace(go.Bar(
        x=names, y=sil_vals,
        marker_color=colors_bar[:len(names)],
        text=[f"{v:.3f}" for v in sil_vals],
        textposition="outside",
        name="Silhouette"
    ), row=1, col=1)

    # Ligne de référence à 0 pour Silhouette
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  line_width=1, row=1, col=1)

    # Davies-Bouldin (inverser : plus petit = meilleur, on affiche direct)
    db_display = [min(v, 20.0) for v in db_vals]   # cap pour lisibilité
    fig.add_trace(go.Bar(
        x=names, y=db_display,
        marker_color=colors_bar[:len(names)],
        text=[f"{v:.3f}" for v in db_vals],
        textposition="outside",
        name="Davies-Bouldin"
    ), row=1, col=2)

    # Calinski-Harabász
    fig.add_trace(go.Bar(
        x=names, y=ch_vals,
        marker_color=colors_bar[:len(names)],
        text=[f"{v:.1f}" for v in ch_vals],
        textposition="outside",
        name="Calinski-Harabász"
    ), row=1, col=3)

    # Annotations direction
    for col, direction in zip([1, 2, 3], ["↑ meilleur", "↓ meilleur", "↑ meilleur"]):
        fig.update_yaxes(title_text=direction, row=1, col=col)

    fig.update_xaxes(tickangle=-30)
    fig.update_layout(
        title=dict(
            text="Évaluation comparative — Indices internes (sans vérité terrain)<br>"
                 "<sup>Chapitre 7 — CADEMI Théorie du Clustering</sup>",
            font=dict(size=15)
        ),
        template="plotly_white",
        showlegend=False,
        height=480,
        margin=dict(l=50, r=20, t=100, b=120)
    )
    return fig
