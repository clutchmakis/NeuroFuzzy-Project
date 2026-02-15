"""Evaluation, metrics reporting and visualisation helpers."""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)


def evaluate_model(model, X_test, y_test, encoder, model_name: str = "Model"):
    """Predict on *X_test*, decode labels and print a classification report.

    Parameters
    ----------
    model : keras.Model
        A trained Keras model.
    X_test : np.ndarray
        Preprocessed (padded) test sequences.
    y_test : np.ndarray
        One-hot encoded true labels.
    encoder : LabelEncoder
        Fitted ``sklearn.preprocessing.LabelEncoder`` used for label mapping.
    model_name : str
        Name shown in printed output.

    Returns
    -------
    tuple
        (predicted_labels_text, true_labels_text, accuracy, precision, recall, fscore)
    """
    predictions = model.predict(X_test)
    predicted_indices = predictions.argmax(axis=1)
    true_indices = y_test.argmax(axis=1)

    predicted_labels = encoder.inverse_transform(predicted_indices)
    true_labels = encoder.inverse_transform(true_indices)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, fscore, support = precision_recall_fscore_support(
        true_labels, predicted_labels
    )

    print(f"\n{'='*60}")
    print(f"  {model_name} – Test-set Evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n  Per-class metrics:")
    print(f"  {'Precision':>12s} {'Recall':>10s} {'F-score':>10s} {'Support':>10s}")
    for p, r, f, s in zip(precision, recall, fscore, support):
        print(f"  {p:12.4f} {r:10.4f} {f:10.4f} {s:10d}")

    print(f"\n  Macro avg  – P: {precision.mean():.4f}  R: {recall.mean():.4f}  F: {fscore.mean():.4f}")
    print(f"{'='*60}\n")

    return predicted_labels, true_labels, accuracy, precision, recall, fscore


def measure_inference_time(model, X_test, n_samples: int = 20) -> float:
    """Measure average single-sample inference time in milliseconds."""
    times = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(X_test))
        sample = X_test[idx : idx + 1]
        start = time.time()
        model.predict(sample, verbose=0)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_ms = np.mean(times)
    print(f"Average inference time ({n_samples} samples): {avg_ms:.2f} ms")
    return avg_ms


def plot_training_history(history, title_prefix: str = ""):
    """Plot loss and accuracy curves from a Keras ``History`` object."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    prefix = f"{title_prefix} – " if title_prefix else ""

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title(f"{prefix}Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    acc_key = "categorical_accuracy" if "categorical_accuracy" in history.history else "accuracy"
    val_acc_key = f"val_{acc_key}"
    axes[1].plot(history.history[acc_key], label="Train Accuracy")
    axes[1].plot(history.history[val_acc_key], label="Validation Accuracy")
    axes[1].set_title(f"{prefix}Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion(true_labels, predicted_labels, title: str = "Confusion Matrix"):
    """Display a confusion matrix for the given labels."""
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
