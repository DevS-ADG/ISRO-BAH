"""
ASTRA Training — 1D CNN Shape Classifier Training.

Trains the CNN on phase-folded light curve arrays with early stopping.
Handles class imbalance with SMOTE oversampling.

Uses Adam optimizer, lr=1e-3, batch size 64, patience 10 epochs.
"""

import numpy as np
from pathlib import Path

from astra.utils.logger import get_logger

logger = get_logger("astra.training.train_cnn")


def train_cnn_classifier(
    X_curves: np.ndarray,
    labels: np.ndarray,
    model_dir: str = "models/cnn_phase2/",
    report_dir: str = "models/training_reports/",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    input_length: int = 256,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_epochs: int = 200,
    patience: int = 10,
    use_smote: bool = True,
    random_seed: int = 42,
) -> dict:
    """Train the 1D CNN classifier on phase-folded light curves.

    Args:
        X_curves: Array of shape (n_samples, input_length) with resampled
                  phase-folded flux arrays.
        labels: Multi-class labels (0=PLANET, 1=EB, 2=BLEND, 3=OTHER).
        model_dir: Directory to save trained weights.
        report_dir: Directory for training reports.
        train_fraction: Training set fraction.
        val_fraction: Validation set fraction.
        test_fraction: Test set fraction.
        input_length: Input sequence length.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        use_smote: Whether to apply SMOTE oversampling.
        random_seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    from astra.classification.cnn_classifier import CNNClassifier

    logger.info(f"Training CNN: {len(labels)} samples, input_length={input_length}")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_curves, labels, test_size=(val_fraction + test_fraction),
        stratify=labels, random_state=random_seed,
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        stratify=y_temp, random_state=random_seed,
    )

    # SMOTE oversampling on training set
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_seed)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied: training set expanded to {len(y_train)} samples")
        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping SMOTE")

    logger.info(
        f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    )

    # Create PyTorch datasets
    # Shape: (n, 1, input_length) — 1 channel
    train_X = torch.FloatTensor(X_train).unsqueeze(1)
    train_y = torch.LongTensor(y_train.astype(int))
    val_X = torch.FloatTensor(X_val).unsqueeze(1)
    val_y = torch.LongTensor(y_val.astype(int))
    test_X = torch.FloatTensor(X_test).unsqueeze(1)
    test_y = torch.LongTensor(y_test.astype(int))

    train_ds = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_ds = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size)
    test_ds = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size)

    # Build model
    cnn = CNNClassifier(model_dir=model_dir, input_length=input_length)
    cnn.build()
    model = cnn.model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_ds:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        n_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_ds:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                n_val_batches += 1
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_loss /= max(n_val_batches, 1)
        val_acc = accuracy_score(val_true, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        for batch_X, batch_y in test_ds:
            outputs = model(batch_X)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())

    from astra.classification.xgb_classifier import CLASS_NAMES

    test_acc = accuracy_score(test_true, test_preds)
    test_report = classification_report(test_true, test_preds, target_names=CLASS_NAMES)

    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"\n{test_report}")

    # Save model
    cnn.model = model
    cnn.save()

    # Save report
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / "cnn_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ASTRA — CNN Phase 2 Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input length: {input_length}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Epochs trained: {len(history['train_loss'])}\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report (Test Set):\n")
        f.write(test_report + "\n")

    logger.info(f"CNN training report saved to {report_path}")

    return {
        "test_accuracy": test_acc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }
