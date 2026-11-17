"""Le modèle reconnaît-il des chiffres qui ne viennent pas de MNIST ?

On écrit les chiffres 0 à 9 avec 12 polices installées sous Windows (8 de style
manuscrit, 4 d'imprimerie), on les passe dans la même chaîne de prétraitement
que l'application, puis on demande au réseau convolutif de les reconnaître.

Usage : python scripts/test_chiffres_externes.py   (après scripts/modelisation.py)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
from cnn import PetitCNN, probabilites  # noqa: E402
from pretraitement import vers_mnist  # noqa: E402

POLICES = {  # nom affiché : (fichier, style)
    "Segoe Print": ("segoepr.ttf", "manuscrit"), "Segoe Script": ("segoesc.ttf", "manuscrit"),
    "Ink Free": ("Inkfree.ttf", "manuscrit"), "Lucida Handwriting": ("LHANDW.TTF", "manuscrit"),
    "Bradley Hand": ("BRADHITC.TTF", "manuscrit"), "Kristen ITC": ("ITCKRIST.TTF", "manuscrit"),
    "MV Boli": ("mvboli.ttf", "manuscrit"), "Comic Sans": ("comic.ttf", "manuscrit"),
    "Arial": ("arial.ttf", "imprimerie"), "Times New Roman": ("times.ttf", "imprimerie"),
    "Georgia": ("georgia.ttf", "imprimerie"), "Verdana": ("verdana.ttf", "imprimerie"),
}

sauvegarde = torch.load(RACINE / "app" / "modele_cnn.pt", weights_only=True)
modele = PetitCNN(dropout=sauvegarde["dropout"])
modele.load_state_dict(sauvegarde["etat"])

lignes, images = [], {}
for nom, (fichier, style) in POLICES.items():
    chemin = Path("C:/Windows/Fonts") / fichier
    if not chemin.exists():
        continue
    police = ImageFont.truetype(str(chemin), 200)
    for c in range(10):
        img = Image.new("L", (300, 300), 0)
        ImageDraw.Draw(img).text((150, 150), str(c), fill=255, font=police, anchor="mm")
        x = vers_mnist(img, inverser=False)
        p = probabilites(modele, x[None])[0]
        images[(nom, c)] = x
        lignes.append({"police": nom, "style": style, "chiffre": c, "predit": int(p.argmax()),
                       "confiance_pct": 100 * float(p.max())})
df = pd.DataFrame(lignes)
df["correct"] = df.chiffre == df.predit
df.to_csv(RACINE / "resultats" / "t8_7_chiffres_externes.csv", index=False, float_format="%.1f")

par_style = df.groupby("style").correct.mean().mul(100).round(1).to_dict()
par_chiffre = df.groupby("chiffre").correct.mean().mul(100).round(0).to_dict()
erreurs = df[~df.correct]
resume = {"n_images": int(len(df)), "n_polices": int(df.police.nunique()),
          "accuracy_pct": round(100 * df.correct.mean(), 1), "par_style": par_style,
          "par_chiffre": {str(k): v for k, v in par_chiffre.items()},
          "erreurs": [f"{r.chiffre} lu {r.predit} ({r.police})" for r in erreurs.itertuples()]}
(RACINE / "resultats" / "chiffres_cles_test_externe.json").write_text(
    json.dumps(resume, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(resume, ensure_ascii=False, indent=1))

# Figure : une ligne par police, en rouge les erreurs
plt.rcParams.update({"font.family": "Times New Roman"})
noms = list(dict.fromkeys(df.police))
fig, axes = plt.subplots(len(noms), 10, figsize=(7.2, 0.78 * len(noms)))
for i, nom in enumerate(noms):
    for c in range(10):
        ax = axes[i, c]
        r = df[(df.police == nom) & (df.chiffre == c)].iloc[0]
        ax.imshow(images[(nom, c)], cmap="gray_r")
        ax.set_xticks([]); ax.set_yticks([])
        couleur = "#2e7d32" if r.correct else "#c62828"
        for s in ax.spines.values():
            s.set_color(couleur); s.set_linewidth(1.6)
        ax.set_title(str(r.predit), fontsize=8, color=couleur, pad=1)
        if c == 0:
            ax.set_ylabel(nom, rotation=0, ha="right", va="center", fontsize=8)
fig.subplots_adjust(wspace=0.15, hspace=0.45)
fig.savefig(RACINE / "figures" / "fig8_6_chiffres_externes.png", dpi=200, bbox_inches="tight")
