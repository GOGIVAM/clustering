from init import *
from partionnal_clustering import KMeans
from density_clustering import DBSCAN
from hierarchy_clustering import HAC
from spectral_clustering import SpectralClustering
from deep_clustering import DEC
from probabilistic_clustering import GaussianMixtureEM
from mock_data import make_blobs_3d, make_moons_3d, make_gmm_3d, make_rings_3d
from validate_clustering import evaluate_all




# ============================================================
# SECTION 8 — Application principale
# ============================================================

def run_all():
    """
    Point d'entrée principal.
    Lance les 6 algorithmes, affiche leurs figures individuelles,
    puis génère le tableau comparatif des indices internes.
    """
    print("=" * 65)
    print("CADEMI — Théorie du Clustering : Implémentation from scratch")
    print("=" * 65)

    results = []   # stockage pour l'évaluation finale

    # ----------------------------------------------------------
    # 1. K-MEANS — Blobs 3D
    # ----------------------------------------------------------
    print("\n[1/6] K-Means (Lloyd + k-means++) — Blobs 3D")
    X_blobs = make_blobs_3d(n_samples=400, n_clusters=4)
    kmeans = KMeans(k=4, n_init=10, random_state=0)
    kmeans.fit(X_blobs)
    fig1 = kmeans.visualize(X_blobs, "— Blobs 3D")
    print(f"    Inertie finale             : {kmeans.inertia_:.4f}")
    print(f"    Itérations (meilleure run) : {kmeans.n_iter_}")
    fig1.show()
    results.append({"name": "K-Means", "X": X_blobs, "labels": kmeans.labels_})

    # ----------------------------------------------------------
    # 2. HAC Ward — Blobs 2D
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
    print(f"    Clusters trouvés   : {hac.n_clusters_}")
    print(f"    Fusions effectuées : {len(hac.merge_history_)}")
    fig2.show()
    results.append({"name": "HAC (Ward)", "X": X_hac, "labels": hac.labels_})

    # ----------------------------------------------------------
    # 3. DBSCAN — Croissants 3D
    # ----------------------------------------------------------
    print("\n[3/6] DBSCAN — Croissants 3D (non-convexes)")
    X_moons = make_moons_3d(n_samples=400, noise=0.12)
    dbscan = DBSCAN(eps=0.3, min_pts=6)
    dbscan.fit(X_moons)
    fig3 = dbscan.visualize(X_moons, "— Croissants 3D")
    n_noise = (dbscan.labels_ == -1).sum()
    print(f"    Clusters détectés : {dbscan.n_clusters_}")
    print(f"    Points bruit      : {n_noise}")
    fig3.show()
    results.append({"name": "DBSCAN", "X": X_moons, "labels": dbscan.labels_})

    # ----------------------------------------------------------
    # 4. GMM-EM — Gaussiennes ellipsoïdales 3D
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
    results.append({"name": "GMM-EM", "X": X_gmm, "labels": gmm.labels_})

    # ----------------------------------------------------------
    # 5. Clustering Spectral — Anneaux 3D
    # ----------------------------------------------------------
    print("\n[5/6] Clustering Spectral (NJW) — Anneaux 3D")
    X_rings = make_rings_3d(n_samples=400, noise=0.05)
    spectral = SpectralClustering(k=2, sigma=0.5, affinity="rbf")
    spectral.fit(X_rings)
    fig5 = spectral.visualize(X_rings, "— Anneaux 3D")
    print(f"    Clusters trouvés      : {spectral.n_clusters_}")
    print(f"    Valeurs propres top-k : {np.round(spectral.eigenvalues_, 4)}")
    fig5.show()
    results.append({"name": "Spectral (NJW)", "X": X_rings, "labels": spectral.labels_})

    # ----------------------------------------------------------
    # 6. DEC — Blobs 2D
    # ----------------------------------------------------------
    print("\n[6/6] DEC (Deep Embedded Clustering) — Blobs 2D")
    rng2 = np.random.default_rng(99)
    X_dec = np.vstack([
        rng2.normal([0, 0], 0.8, (150, 2)),
        rng2.normal([5, 0], 0.8, (150, 2)),
        rng2.normal([2.5, 4], 0.8, (150, 2)),
    ]).astype(np.float32)
    dec = DEC(
        k=3, latent_dim=8, hidden_dims=[64, 32],
        pretrain_epochs=80, cluster_epochs=80,
        lr_pretrain=1e-3, lr_cluster=1e-4,
        batch_size=64, random_state=42
    )
    dec.fit(X_dec)
    fig6 = dec.visualize(X_dec, "— Blobs 2D")
    print(f"    Clusters trouvés        : {dec.n_clusters_}")
    print(f"    Perte pré-entraînement  : {dec.pretrain_losses_[-1]:.6f}")
    print(f"    Perte DEC (KL) finale   : {dec.cluster_losses_[-1]:.6f}")
    fig6.show()
    results.append({"name": "DEC", "X": X_dec, "labels": dec.labels_})

    # ----------------------------------------------------------
    # 7. Évaluation comparative finale
    # ----------------------------------------------------------
    fig_eval = evaluate_all(results)
    fig_eval.show()

    print("\n" + "=" * 65)
    print("Toutes les visualisations ont été générées.")
    print("=" * 65)


if __name__ == "__main__":
    run_all()