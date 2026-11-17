"""Chargement des fichiers MNIST au format IDX (avec contrôle de l'en-tête)."""
import gzip
from pathlib import Path

import numpy as np

FICHIERS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def lire_idx(chemin):
    """Lit un fichier IDX gzippé et renvoie (nombre magique, dimensions, tableau numpy)."""
    with gzip.open(chemin, "rb") as f:
        brut = f.read()
    magique = int.from_bytes(brut[:4], "big")
    assert brut[0] == 0 and brut[1] == 0, "en-tête IDX invalide"
    assert brut[2] == 0x08, "type de donnée attendu : unsigned byte"
    ndim = brut[3]
    dims = [int.from_bytes(brut[4 + 4*i: 8 + 4*i], "big") for i in range(ndim)]
    valeurs = np.frombuffer(brut, dtype=np.uint8, offset=4 + 4*ndim)
    assert valeurs.size == np.prod(dims), "taille déclarée ≠ taille lue"
    return magique, dims, valeurs.reshape(dims)


def charger_mnist(dossier):
    """Renvoie X_train (60000, 28, 28), y_train, X_test (10000, 28, 28), y_test en uint8."""
    dossier = Path(dossier)
    a = {cle: lire_idx(dossier / nom)[2] for cle, nom in FICHIERS.items()}
    return a["train_images"], a["train_labels"], a["test_images"], a["test_labels"]
