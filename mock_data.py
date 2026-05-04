from  init import *
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



def make_moons_3d(n_samples: int = 300, noise: float = 0.12,
                  random_state: int = 42) -> np.ndarray:
    """
    Deux croissants bien séparés en 3D.
    Les deux lunes sont décalées de façon à ce que les distances
    inter-cluster soient nettement > eps.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = rng.uniform(0, np.pi, n)

    # Lune 1 : arc supérieur, centré en (0,0)
    X1 = np.column_stack([
        np.cos(theta) + rng.normal(0, noise, n),
        np.sin(theta) + rng.normal(0, noise, n),
        rng.normal(0, noise, n)
    ])
    # Lune 2 : arc inférieur, décalée de (1, -0.5)
    X2 = np.column_stack([
        1 - np.cos(theta) + rng.normal(0, noise, n),
        0.3 - np.sin(theta) + rng.normal(0, noise, n),
        rng.normal(0, noise, n)
    ])
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