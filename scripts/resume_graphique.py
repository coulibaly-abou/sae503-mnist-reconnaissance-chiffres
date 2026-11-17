"""Résumé graphique du rapport -> figures/fig0_resume_graphique.png (après modelisation.py)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
from mnist_io import charger_mnist  # noqa: E402

c78 = json.loads((RACINE / "resultats" / "chiffres_cles_p7_p8.json").read_text(encoding="utf-8"))
c6 = json.loads((RACINE / "resultats" / "chiffres_cles_p6.json").read_text(encoding="utf-8"))
Xtr, ytr, _, _ = charger_mnist(RACINE / "data")


def v(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


cnn = c78["test"]["Réseau convolutif ajusté"]
svm = c78["test"]["SVM RBF ajusté"]
b = c78["bootstrap"]["Accuracy Réseau convolutif ajusté"]
ETAPES = [
    ("Données", "#dbe9f6", "#1f4e79", "70 000 chiffres manuscrits\n28 x 28 pixels\n60 000 apprentissage\n10 000 test"),
    ("Qualité et EDA", "#e2f0d9", "#38761d", "0 manquant, 0 doublon\nclasses équilibrées\n67 pixels toujours vides\n90 % de variance : 87 CP"),
    ("Anomalies", "#fff2cc", "#b45f06", f"{c6['consensus_n']} images atypiques\n(conservées)\n{c6['suspects_n']} étiquettes fausses\n(retirées)"),
    ("7 modèles", "#f4dce6", "#85200c", "VC à 5 plis\nACP à 50 composantes\ngrid search\nSVM et CNN"),
    ("Résultat", "#e8e0f3", "#4c2a85", f"CNN : {v(cnn['err'], 2)} % d'erreur\nIC 95 % : [{v(100 - b[2], 2)} ; {v(100 - b[1], 2)}]\nSVM : {v(svm['err'], 2)} %\nRana et al. : 2,08 %"),
]

plt.rcParams.update({"font.family": "Times New Roman"})
fig = plt.figure(figsize=(10, 3.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10); ax.set_ylim(0, 3.3); ax.axis("off")
largeur, ecart = 1.72, 0.3
for i, (titre, fond, bord, texte) in enumerate(ETAPES):
    x = 0.12 + i * (largeur + ecart)
    ax.add_patch(FancyBboxPatch((x, 0.15), largeur, 2.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fond, ec=bord, lw=1.5))
    ax.text(x + largeur / 2, 2.12, titre, ha="center", va="center", fontsize=12, fontweight="bold", color=bord)
    ax.text(x + largeur / 2, 1.1, texte, ha="center", va="center", fontsize=9.5, linespacing=1.5)
    if i < len(ETAPES) - 1:
        ax.add_patch(FancyArrowPatch((x + largeur + 0.02, 1.25), (x + largeur + ecart - 0.02, 1.25),
                                     arrowstyle="-|>", mutation_scale=14, color="#595959"))
# bandeau de chiffres en haut
rng = np.random.default_rng(3)
for c in range(10):
    i = rng.choice(np.where(ytr == c)[0])
    a = fig.add_axes([0.2 + c * 0.062, 0.79, 0.05, 0.17])
    a.imshow(Xtr[i], cmap="gray_r"); a.axis("off")
fig.savefig(RACINE / "figures" / "fig0_resume_graphique.png", dpi=200, bbox_inches="tight")
print("figures/fig0_resume_graphique.png")
