import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay, roc_curve

import torch

import json
import os

def log_model(train_losses, val_losses, log_dir):
    
    train_epochs, train_vals = zip(*train_losses)
    val_epochs, val_vals = zip(*val_losses)

    # save the training and validation losses
    with open(os.path.join(log_dir, 'losses.json'), 'w') as f:
        json.dump(
            {
                'train_epochs': train_epochs,
                'train_losses': train_vals,
                'val_epochs': val_epochs,
                'val_losses': val_vals,
            },
        f)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(train_epochs, train_vals, label="Train Loss", linewidth=2)
    ax.plot(val_epochs, val_vals, label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14)

    # Grid, legend, and tight layout
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    # Save to file without showing
    fig.savefig(os.path.join(log_dir, 'losses.png'), dpi=300)  # high resolution
    plt.close(fig)  # close the figure to free memory

def get_roc_curve(model, test_loader, encoder, log_dir, device='cpu'):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            if encoder is not None:
                data = encoder.encode(data)
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
    
            y_true.append(target.numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    ax.plot(fpr, tpr, label='ROC Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    fig.savefig(os.path.join(log_dir, 'test_roc.png'), dpi=300)  # high resolution
    plt.show()