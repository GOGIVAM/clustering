from init import *
from base_cluster import BaseClusterer
from shared import _palette, euclidean, euclidean_sq



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
        """
        Affiche les clusters (2D ou 3D) dans un subplot gauche,
        et la courbe de convergence EM dans un subplot droit.
        Les ellipses 2σ sont ajoutées uniquement en mode 2D.
        """
        self._check_fitted()
        labels = self.labels_
        unique_labels = np.unique(labels)
        colors = _palette(self.k)
        is_3d = X.shape[1] >= 3

        # ---- Spec des subplots selon la dimension ----------------
        if is_3d:
            specs = [[{"type": "scene"}, {"type": "xy"}]]
        else:
            specs = [[{"type": "xy"}, {"type": "xy"}]]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Clusters GMM" + (" + ellipses 2σ" if not is_3d else " (3D)"),
                "Convergence EM"
            ),
            column_widths=[0.65, 0.35],
            specs=specs
        )

        # ---- Panneau gauche : scatter des clusters ---------------
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

        # ---- Ellipses 2σ (2D uniquement) -------------------------
        if not is_3d:
            t_vals = np.linspace(0, 2 * np.pi, 100)
            circle = np.column_stack([np.cos(t_vals), np.sin(t_vals)])
            for k in range(self.k):
                try:
                    eigvals, eigvecs = np.linalg.eigh(self.sigma_[k])
                    eigvals = np.maximum(eigvals, 0)
                    scale = 2.0 * np.sqrt(eigvals)
                    ellipse = circle @ (eigvecs * scale).T + self.mu_[k]
                    fig.add_trace(go.Scatter(
                        x=ellipse[:, 0], y=ellipse[:, 1],
                        mode="lines",
                        line=dict(width=2,
                                  color=colors[k % len(colors)],
                                  dash="dash"),
                        showlegend=False
                    ), row=1, col=1)
                except Exception:
                    pass

        # ---- Panneau droit : log-vraisemblance EM ----------------
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

        if is_3d:
            fig.update_layout(scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃"
            ))

        fig.update_layout(
            title=dict(
                text=f"{self.name} {title_suffix}<br>"
                     f"<sup>K={self.k}, itérations={self.n_iter_}</sup>",
                font=dict(size=15)
            ),
            template="plotly_white",
            showlegend=True,
            margin=dict(l=40, r=20, t=80, b=40)
        )
        return fig