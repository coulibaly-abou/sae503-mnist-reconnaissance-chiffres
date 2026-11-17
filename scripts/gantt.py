"""Annexe 1 : diagramme de Gantt (lu dans Gantt_SAE503_MNIST_simple .xlsx) -> figures/figA1_gantt.png"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
from matplotlib.patches import Patch

RACINE = Path(__file__).resolve().parents[1]
wb = openpyxl.load_workbook(RACINE / "Gantt_SAE503_MNIST_simple .xlsx", data_only=True)
ws = wb["Gantt détaillé"]
LIGNE_ENTETE = next(r for r in range(1, 10) if str(ws.cell(r, 3).value or "").startswith("Semaine"))
semaines = [ws.cell(LIGNE_ENTETE, c).value.replace("\n", " ") for c in range(3, 14)]
PHASES = {"FFA9D18E": ("Préparation", "#a9d18e"), "FFFFC000": ("Données", "#ffc000"),
          "FFF4A582": ("Modélisation et livrables", "#f4a582"), "FFE03C31": ("Semaine d'évaluation", "#e03c31")}

taches = []
for r in range(LIGNE_ENTETE + 1, LIGNE_ENTETE + 40):
    nom = ws.cell(r, 2).value
    if not nom or nom.startswith("Heures"):
        if nom and nom.startswith("Heures"):
            break
        continue
    cases = [(i, ws.cell(r, c).fill.fgColor.rgb) for i, c in enumerate(range(3, 14))]
    cases = [(i, rgb) for i, rgb in cases if rgb in PHASES]
    taches.append((nom, re.match(r"^\d\. ", nom) is not None, cases))

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
fig, ax = plt.subplots(figsize=(9, 7.5))
for y, (nom, titre, cases) in enumerate(taches):
    if titre:
        ax.axhspan(y - 0.5, y + 0.5, color="#f2f2f2", zorder=0)
    for i, rgb in cases:
        ax.barh(y, 0.92, left=i + 0.04, color=PHASES[rgb][1], height=0.7, zorder=2)
ax.set_yticks(range(len(taches)))
ax.set_yticklabels([("" if t else "    ") + n for n, t, _ in taches])
for lab, (_, t, _) in zip(ax.get_yticklabels(), taches):
    if t:
        lab.set_fontweight("bold")
ax.set_ylim(len(taches) - 0.5, -0.5)
ax.set_xlim(0, len(semaines))
ax.set_xticks([i + 0.5 for i in range(len(semaines))])
ax.set_xticklabels([s.replace(" ", "\n", 1) for s in semaines], fontsize=8)
ax.xaxis.tick_top()
ax.grid(axis="x", color="#d9d9d9")
ax.set_axisbelow(True)
for s in ("right", "bottom"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(color=c, label=l) for l, c in PHASES.values()], loc="lower center",
          bbox_to_anchor=(0.45, -0.07), ncol=4, frameon=False)
fig.savefig(RACINE / "figures" / "figA1_gantt.png", dpi=200, bbox_inches="tight")
print("figures/figA1_gantt.png")
