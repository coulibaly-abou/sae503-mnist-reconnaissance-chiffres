"""Réseau convolutif (PyTorch) utilisé dans la partie 8 et par l'application Streamlit.

On garde volontairement une architecture courte, proche de LeNet : deux blocs
de convolutions, puis une couche dense. Elle s'entraîne en quelques minutes
sur un processeur ordinaire.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PetitCNN(nn.Module):
    def __init__(self, dropout=0.25, filtres=32):
        super().__init__()
        f = filtres
        self.bloc1 = nn.Sequential(
            nn.Conv2d(1, f, 3, padding=1), nn.ReLU(),
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
        )
        self.bloc2 = nn.Sequential(
            nn.Conv2d(f, 2 * f, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2 * f, 2 * f, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
        )
        self.tete = nn.Sequential(
            nn.Flatten(), nn.Linear(2 * f * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.tete(self.bloc2(self.bloc1(x)))


def augmenter(x, rotation=10, echelle=0.10, decalage=2):
    """Petites déformations aléatoires (rotation, zoom, translation) d'un lot d'images."""
    n = x.shape[0]
    a = torch.empty(n).uniform_(-rotation, rotation) * math.pi / 180
    s = torch.empty(n).uniform_(1 - echelle, 1 + echelle)
    t = torch.empty(n, 2).uniform_(-decalage, decalage) * 2 / 28
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = torch.cos(a) / s
    theta[:, 0, 1] = -torch.sin(a) / s
    theta[:, 1, 0] = torch.sin(a) / s
    theta[:, 1, 1] = torch.cos(a) / s
    theta[:, :, 2] = t
    grille = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grille, align_corners=False, padding_mode="zeros")


def vers_tenseur(images):
    """(n, 28, 28) uint8 -> tenseur (n, 1, 28, 28) float dans [0, 1]."""
    return torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1)


def entrainer(images, etiquettes, epoques=6, lr=1e-3, dropout=0.25, augmentation=False,
              taille_lot=128, graine=0, journal=None):
    torch.manual_seed(graine)
    np.random.seed(graine)
    x = vers_tenseur(images)
    y = torch.from_numpy(np.asarray(etiquettes, dtype=np.int64))
    modele = PetitCNN(dropout=dropout)
    opt = torch.optim.Adam(modele.parameters(), lr=lr)
    planif = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoques)
    n = len(y)
    for e in range(epoques):
        modele.train()
        ordre = torch.randperm(n)
        perte_tot = 0.0
        for i in range(0, n, taille_lot):
            idx = ordre[i:i + taille_lot]
            xb = x[idx]
            if augmentation:
                xb = augmenter(xb)
            opt.zero_grad()
            perte = F.cross_entropy(modele(xb), y[idx])
            perte.backward()
            opt.step()
            perte_tot += perte.item() * len(idx)
        planif.step()
        if journal:
            journal(f"    époque {e + 1}/{epoques} : perte {perte_tot / n:.4f}")
    modele.eval()
    return modele


@torch.no_grad()
def probabilites(modele, images, taille_lot=1000):
    modele.eval()
    x = vers_tenseur(images)
    sorties = [F.softmax(modele(x[i:i + taille_lot]), dim=1) for i in range(0, len(x), taille_lot)]
    return torch.cat(sorties).numpy()
