"""
Quantile Regression Neural Network (QRNN).

Single network with one output neuron per quantile (default: 5, 10, ..., 95).
Uses pinball loss. Exposes fit(X, y) and predict(X) returning (n_samples, n_quantiles).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


# Default quantile levels: 5 to 95 with step 5 (confidence levels in %)
DEFAULT_QUANTILES = np.arange(
    5, 100, 5, dtype=np.float64) / 100.0  # 0.05, 0.10, ..., 0.95


def _to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32):
    """Convert numpy array to torch tensor on given device."""
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device, dtype=dtype)


def _pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Sum of pinball losses over quantiles.
    loss = (y - pred) * (tau - 1_{y < pred}). Implemented as (mask_low - tau) * (pred - y).
    pred: (batch, n_quantiles), y: (batch, 1), quantiles: (n_quantiles,)
    """
    diff = pred - y  # (batch, n_quantiles)
    mask_low = (y < pred).float()  # (batch, n_quantiles)
    # (batch, n_quantiles) * (1, n_quantiles) -> (batch, n_quantiles)
    losses = (mask_low - quantiles.unsqueeze(0)) * diff  # (batch, n_quantiles)
    return losses.mean(dim=0).sum()  # mean over batch, sum over quantiles


class _QRNN(nn.Module):
    """MLP with shared backbone and one output per quantile."""

    def __init__(self, n_features: int, hidden_size: int, n_hidden_layers: int, n_quantiles: int):
        super().__init__()
        self.n_quantiles = n_quantiles
        layers = []
        in_dim = n_features
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


class QRNN:
    """
    Quantile Regression Neural Network.

    Output neurons for quantile levels from 5% to 95% with step 5% (19 quantiles).
    fit(X, y) and predict(X) -> (n_samples, 19) in order 0.05, 0.10, ..., 0.95.
    """

    def __init__(
        self,
        quantiles: np.ndarray | None = None,
        random_state: int = 42,
        hidden_size: int = 64,
        n_hidden_layers: int = 2,
        epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        if quantiles is None:
            quantiles = DEFAULT_QUANTILES
        self.quantiles = np.asarray(quantiles, dtype=np.float64).ravel()
        self.n_quantiles = len(self.quantiles)
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model: _QRNN | None = None
        self.n_features_: int | None = None
        self._device: torch.device | None = None

    def _set_seeds(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QRNN":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.n_features_ = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        y_2d = y[:, np.newaxis]  # (n, 1)
        self._set_seeds()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        model = _QRNN(
            n_features=self.n_features_,
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
            n_quantiles=self.n_quantiles,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        q_tensor = _to_tensor(self.quantiles, device).unsqueeze(
            0)  # (1, n_quantiles)
        n = len(X_scaled)
        for epoch in range(self.epochs):
            perm = np.random.RandomState(
                self.random_state + epoch).permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                batch_X = _to_tensor(X_scaled[idx], device)
                batch_y = _to_tensor(y_2d[idx], device)  # (batch, 1)
                optimizer.zero_grad()
                pred = model(batch_X)  # (batch, n_quantiles)
                loss = _pinball_loss(pred, batch_y, q_tensor.squeeze(0))
                loss.backward()
                optimizer.step()
        self.model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_quantiles) with columns in order self.quantiles."""
        if self.model is None:
            raise ValueError("QRNN not fitted")
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            t = _to_tensor(X_scaled, self._device)
            out = self.model(t)
        return out.cpu().numpy()

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        """Point prediction as median (0.5 quantile). Uses 50% output if present, else interpolate."""
        q_pred = self.predict(X)
        idx_50 = np.searchsorted(self.quantiles, 0.5)
        if idx_50 >= self.n_quantiles:
            return q_pred[:, -1]
        if idx_50 == 0:
            return q_pred[:, 0]
        if self.quantiles[idx_50] == 0.5:
            return q_pred[:, idx_50]
        # interpolate between idx_50-1 and idx_50
        w = (0.5 - self.quantiles[idx_50 - 1]) / \
            (self.quantiles[idx_50] - self.quantiles[idx_50 - 1])
        return (1 - w) * q_pred[:, idx_50 - 1] + w * q_pred[:, idx_50]
