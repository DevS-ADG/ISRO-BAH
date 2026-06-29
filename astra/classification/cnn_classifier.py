"""
ASTRA CNN Classifier — 1D Convolutional Neural Network for shape embedding.

Learns a 128-dimensional shape embedding from raw phase-folded light curve
profiles, capturing morphological features the 19-feature vector may miss.

Architecture:
  Conv1d(1,32,7) → ReLU → MaxPool → Conv1d(32,64,5) → ReLU → MaxPool →
  Conv1d(64,128,3) → ReLU → MaxPool → FC(128) → ReLU → Dropout(0.3) →
  FC(4) → Softmax

Input: 256-point resampled, normalized phase-folded flux.
"""

import numpy as np
from pathlib import Path

from astra.utils.logger import get_logger

logger = get_logger("astra.classification.cnn_classifier")


def _build_cnn_model(input_length: int = 256, n_classes: int = 4):
    """Build the PyTorch 1D CNN model.

    Args:
        input_length: Length of input sequence (default 256).
        n_classes: Number of output classes (default 4).

    Returns:
        PyTorch nn.Module model.
    """
    import torch
    import torch.nn as nn

    class TransitCNN(nn.Module):
        """1D CNN for transit shape classification.

        Architecture follows the specification exactly:
        3 convolutional layers with increasing filters (32→64→128),
        followed by FC layers producing a 128-dim embedding and
        4-class softmax output.
        """

        def __init__(self, input_len: int = 256, num_classes: int = 4):
            super().__init__()

            self.conv_layers = nn.Sequential(
                # Block 1: Conv1d(1, 32, kernel_size=7, padding=3) → ReLU → MaxPool
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Block 2: Conv1d(32, 64, kernel_size=5, padding=2) → ReLU → MaxPool
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Block 3: Conv1d(64, 128, kernel_size=3, padding=1) → ReLU → MaxPool
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

            # Compute flattened size: input_len / 2^3 = input_len / 8
            flat_size = 128 * (input_len // 8)

            self.embedding = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, 1, input_len).

            Returns:
                Class logits of shape (batch, num_classes).
            """
            x = self.conv_layers(x)
            x = self.embedding(x)
            x = self.classifier(x)
            return x

        def get_embedding(self, x):
            """Extract 128-dim embedding without classification head.

            Args:
                x: Input tensor of shape (batch, 1, input_len).

            Returns:
                Embedding tensor of shape (batch, 128).
            """
            x = self.conv_layers(x)
            x = self.embedding(x)
            return x

    return TransitCNN(input_len=input_length, num_classes=n_classes)


class CNNClassifier:
    """1D CNN classifier for transit shape classification.

    Uses phase-folded light curves resampled to 256 points.
    Produces both class probabilities and 128-dim shape embeddings.

    Args:
        model_dir: Directory for saved model weights.
        input_length: Input sequence length (default 256).
        n_classes: Number of classes (default 4).
    """

    def __init__(
        self,
        model_dir: str = "models/cnn_phase2/",
        input_length: int = 256,
        n_classes: int = 4,
    ):
        self.model_dir = Path(model_dir)
        self.input_length = input_length
        self.n_classes = n_classes
        self.model = None
        self.device = "cpu"  # CPU only as per spec

    def build(self) -> None:
        """Build the CNN model."""
        self.model = _build_cnn_model(self.input_length, self.n_classes)
        self.model.to(self.device)
        logger.info(f"CNN model built ({self.device})")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities for phase-folded light curves.

        Args:
            X: Array of shape (n_candidates, 256) with resampled phase-folded flux.

        Returns:
            Tuple of (class_labels, probability_matrix).
        """
        import torch
        import torch.nn.functional as F

        if self.model is None:
            raise RuntimeError("CNN model not loaded. Call load() or train first.")

        self.model.eval()

        # Reshape: (n, 256) → (n, 1, 256)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = F.softmax(logits, dim=1).cpu().numpy()

        labels = np.argmax(proba, axis=1)
        return labels, proba

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """Extract 128-dim shape embeddings.

        Args:
            X: Array of shape (n_candidates, 256).

        Returns:
            Embedding array of shape (n_candidates, 128).
        """
        import torch

        if self.model is None:
            raise RuntimeError("CNN model not loaded.")

        self.model.eval()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_embedding(X_tensor).cpu().numpy()

        return embeddings

    def save(self) -> None:
        """Save model weights."""
        import torch

        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / "cnn_weights.pt")
        logger.info(f"CNN model saved to {self.model_dir}")

    def load(self) -> bool:
        """Load model weights.

        Returns:
            True if loaded successfully.
        """
        import torch

        weights_path = self.model_dir / "cnn_weights.pt"
        if not weights_path.exists():
            logger.warning(f"CNN weights not found at {weights_path}")
            return False

        if self.model is None:
            self.build()

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        logger.info(f"CNN model loaded from {weights_path}")
        return True
