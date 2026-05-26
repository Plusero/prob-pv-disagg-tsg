"""
Neural network regressors for point regression (AusGrid).
All models expose fit(X, y) and predict(X) returning 1D numpy arrays.
Input X: (n_samples, n_features). Scaling is applied internally.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


def _to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32):
    """Convert numpy array to torch tensor on given device."""
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device, dtype=dtype)


class FCNNRegressor:
    """
    Fully connected neural network for regression.
    Input: (n_samples, n_features). Output: (n_samples,) predictions.
    """

    def __init__(
        self,
        random_state: int = 42,
        hidden_size: int = 64,
        n_hidden_layers: int = 2,
        epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.n_features_ = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        self._set_seeds()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _FCNN(
            n_features=self.n_features_,
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        n = len(X_scaled)
        for epoch in range(self.epochs):
            perm = np.random.RandomState(self.random_state + epoch).permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch_X = _to_tensor(X_scaled[idx], device)
                batch_y = _to_tensor(y[idx], device).unsqueeze(1)
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        self.model = model
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            t = _to_tensor(X_scaled, self._device)
            out = self.model(t)
        return out.cpu().numpy().ravel()


class ResNetRegressor:
    """
    Residual network for regression (same I/O as FCNN).
    Input: (n_samples, n_features). Output: (n_samples,) predictions.
    """

    def __init__(
        self,
        random_state: int = 42,
        hidden_size: int = 64,
        n_blocks: int = 2,
        epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.n_features_ = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        self._set_seeds()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _ResNet(
            n_features=self.n_features_,
            hidden_size=self.hidden_size,
            n_blocks=self.n_blocks,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        n = len(X_scaled)
        for epoch in range(self.epochs):
            perm = np.random.RandomState(self.random_state + epoch).permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch_X = _to_tensor(X_scaled[idx], device)
                batch_y = _to_tensor(y[idx], device).unsqueeze(1)
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        self.model = model
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            t = _to_tensor(X_scaled, self._device)
            out = self.model(t)
        return out.cpu().numpy().ravel()


class LSTMRegressor:
    """
    LSTM for regression. Each row is one time step (sequence length 1).
    Input: (n_samples, n_features). Internally reshaped to (n_samples, 1, n_features).
    Output: (n_samples,) predictions.
    """

    def __init__(
        self,
        random_state: int = 42,
        hidden_size: int = 64,
        n_layers: int = 1,
        epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.n_features_ = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        self._set_seeds()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _LSTM(
            n_features=self.n_features_,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        n = len(X_scaled)
        for epoch in range(self.epochs):
            perm = np.random.RandomState(self.random_state + epoch).permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch_X = _to_tensor(X_scaled[idx], device)
                batch_y = _to_tensor(y[idx], device).unsqueeze(1)
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        self.model = model
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            t = _to_tensor(X_scaled, self._device)
            out = self.model(t)
        return out.cpu().numpy().ravel()

    def _set_seeds(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)


# --- PyTorch modules ---


class _FCNN(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return self.relu(out + x)


class _ResNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_blocks: int):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden_size)
        self.blocks = nn.ModuleList([_ResBlock(hidden_size) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.in_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


class _LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.linear(out)


# Attach _set_seeds to FCNN and ResNet (LSTM has its own)
def _set_seeds(self):
    np.random.seed(self.random_state)
    torch.manual_seed(self.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(self.random_state)


FCNNRegressor._set_seeds = _set_seeds
ResNetRegressor._set_seeds = _set_seeds
