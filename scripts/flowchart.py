"""Diagramme flowchart des tâches de la SAE 5.03 -> figures/fig1_flowchart.png

Usage : python scripts/flowchart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig1_flowchart.png"

PHASES = {
    "cadrage":   ("Cadrage", "#dbe9f6", "#1f4e79"),
    "donnees":   ("Compréhension des données", "#e2f0d9", "#38761d"),
    "prepa":     ("Préparation / inférence", "#fff2cc", "#b45f06"),
    "modele":    ("Modélisation", "#f4dce6", "#85200c"),
    "restit":    ("Restitution", "#e8e0f3", "#4c2a85"),
}
W, H = 3.6, 0.95          # taille des boîtes
XL, XR = 3.0, 7.9         # abscisses des deux colonnes

fig, ax = plt.subplots(figsize=(10, 12.5))
ax.set_xlim(0, 10.2); ax.set_ylim(1.2, 16.2); ax.axis("off")


def boite(x, y, titre, detail, phase, w=W, h=H):
    _, fond, bord = PHASES[phase]
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fond, ec=bord, lw=1.4, zorder=2))
    ax.text(x, y + 0.17, titre, ha="center", va="center", fontsize=9.5, weight="bold", color=bord, zorder=3)
    ax.text(x, y - 0.2, detail, ha="center", va="center", fontsize=7.8, color="#333333", zorder=3)


def terminal(x, y, texte, w=3.0):
    ax.add_patch(FancyBboxPatch((x - w/2, y - 0.33), w, 0.66, boxstyle="round,pad=0.02,rounding_size=0.33",
                                fc="#404040", ec="#404040", zorder=2))
    ax.text(x, y, texte, ha="center", va="center", fontsize=9, color="white", weight="bold", zorder=3)


def losange(x, y, texte, phase, dx=1.25, dy=0.62):
    _, fond, bord = PHASES[phase]
    ax.add_patch(Polygon([(x, y + dy), (x + dx, y), (x, y - dy), (x - dx, y)], closed=True,
                         fc="white", ec=bord, lw=1.4, zorder=2))
    ax.text(x, y, texte, ha="center", va="center", fontsize=8.2, weight="bold", color=bord, zorder=3)


def fleche(points, texte=None, pos=None, couleur="#555555"):
    xs, ys = zip(*points)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=couleur, lw=1.2, zorder=1)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=couleur, lw=1.2, mutation_scale=12), zorder=1)
    if texte:
        ax.text(*pos, texte, fontsize=8, style="italic", color=couleur, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.5))


# ---------------------------------------------------------------- colonne gauche
yL = [15.6, 14.3, 12.95, 11.6, 10.25, 8.9, 7.55, 5.85, 4.15]
terminal(XL, yL[0], "Début de la SAE")
boite(XL, yL[1], "1. Gestion de projet", "Gantt · dépôt commun · rôles · flowchart", "cadrage")
boite(XL, yL[2], "2. Bibliographie", "résumé de l'article · recherche d'études similaires", "cadrage")
boite(XL, yL[3], "3. Présentation des données", "lecture IDX · source · période · typologie", "donnees")
boite(XL, yL[4], "4. Qualité des données", "intégrité · doublons · équilibre · pixels constants", "donnees")
boite(XL, yL[5], "5. Exploration (EDA)", "descripteurs · visualisations · dataviz · carte", "donnees")
boite(XL, yL[6], "6. Détection d'anomalies", "méthodes univariées et multivariées", "prepa")
losange(XL, yL[7], "Modifier le\njeu de données ?", "prepa")
boite(XL, yL[8], "7. Estimation bootstrap", "intervalles de confiance sur des paramètres", "prepa")
boite(0.85, yL[7], "Nettoyage", "exclusion\njustifiée", "prepa", w=1.4, h=1.0)

for a, b in zip(yL[:6], yL[1:7]):
    fleche([(XL, a - (0.33 if a == yL[0] else H/2)), (XL, b + H/2)])
fleche([(XL, yL[6] - H/2), (XL, yL[7] + 0.62)])
fleche([(XL, yL[7] - 0.62), (XL, yL[8] + H/2)], "non", (XL + 0.25, (yL[7] + yL[8]) / 2 - 0.05))
fleche([(XL - 1.25, yL[7]), (0.85 + 0.7, yL[7])], "oui", (1.65, yL[7] + 0.2))
fleche([(0.85, yL[7] - 0.5), (0.85, yL[8]), (XL - W/2, yL[8])])

# ---------------------------------------------------------------- colonne droite
yR = [14.3, 12.95, 11.6, 10.25, 8.9, 7.3, 5.7, 4.15, 2.6]
boite(XR, yR[0], "8.i Sélection des prédicteurs", "ACP · variance · importance des variables", "modele")
boite(XR, yR[1], "8.ii Construire ≥ 5 modèles", "algorithmes différents (KNN, SVM, RF, MLP…)", "modele")
boite(XR, yR[2], "8.iii Comparer les modèles", "métriques (accuracy, F1…) en validation croisée", "modele")
boite(XR, yR[3], "8.iv Choisir le meilleur modèle", "compromis performance / coût de calcul", "modele")
boite(XR, yR[4], "8.v Ajuster les hyperparamètres", "grid search + validation croisée", "modele")
losange(XR, yR[5], "Performance\nsatisfaisante ?", "modele")
boite(XR, yR[6], "8.vi Prédire sur le test", "10 000 images jamais vues à l'entraînement", "modele")
boite(XR, yR[7], "Rédaction du rapport", "≤ 15 p. · résumé · listes · annexes · code", "restit")
terminal(XR, yR[8], "Dépôt Moodle (rapport + badges)", w=3.6)

# liaison bootstrap -> modélisation
xm = (XL + W/2 + XR - W/2) / 2
fleche([(XL + W/2, yL[8]), (xm, yL[8]), (xm, yR[0]), (XR - W/2, yR[0])])
for a, b in zip(yR[:4], yR[1:5]):
    fleche([(XR, a - H/2), (XR, b + H/2)])
fleche([(XR, yR[4] - H/2), (XR, yR[5] + 0.62)])
fleche([(XR, yR[5] - 0.62), (XR, yR[6] + H/2)], "oui", (XR + 0.25, (yR[5] + yR[6]) / 2 - 0.05))
fleche([(XR + 1.25, yR[5]), (9.95, yR[5]), (9.95, yR[0]), (XR + W/2, yR[0])], "non", (9.95, (yR[5] + yR[4]) / 2))
fleche([(XR, yR[6] - H/2), (XR, yR[7] + H/2)])
fleche([(XR, yR[7] - H/2), (XR, yR[8] + 0.33)])


leg = [Patch(fc=f, ec=b, label=n) for n, f, b in PHASES.values()]
leg += [Line2D([], [], color="#555555", marker=">", label="enchaînement")]
ax.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))
ax.set_title("Flowchart des tâches – SAE 5.03 (MNIST)", fontsize=13, weight="bold", pad=4)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("écrit :", OUT)
