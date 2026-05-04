from  init import *
from base_cluster import BaseClusterer
from shared import _palette, euclidean, euclidean_sq
from partionnal_clustering import KMeans
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