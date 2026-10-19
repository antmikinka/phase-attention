"""Closed-form ridge probes for decoding (e.g. suppressed) content."""
from __future__ import annotations

import torch


def ridge_probe(Xtr, ytr, Xte, yte, n_classes: int, lam: float = 1e-2) -> float:
    mu = Xtr.mean(0, keepdim=True)
    Xtr, Xte = Xtr - mu, Xte - mu
    Y = torch.nn.functional.one_hot(ytr, n_classes).float()
    D = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * Xtr.shape[0] * torch.eye(D)
    W = torch.linalg.solve(A, Xtr.T @ (Y - Y.mean(0, keepdim=True)))
    pred = (Xte @ W).argmax(-1)
    return (pred == yte).float().mean().item()
