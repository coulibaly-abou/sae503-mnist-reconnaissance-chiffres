"""Annexe 4 : une photo de téléphone avant et après la correction du prétraitement,
puis un test sur 120 fausses photos (chiffres en police, trait fin, papier gris, ombre, tache).

Usage : python scripts/figure_photo.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
from cnn import PetitCNN, probabilites  # noqa: E402
from pretraitement import photo_vers_mnist, vers_mnist  # noqa: E402


def charger(nom, sorties):
    d = torch.load(RACINE / "app" / nom, map_location="cpu", weights_only=True)
    m = PetitCNN(dropout=d["dropout"], sorties=sorties)
    m.load_state_dict(d["etat"])
    return m.eval()


modele, detecteur = charger("modele_cnn.pt", 10), charger("detecteur.pt", 2)
photo = Image.open(RACINE / "data" / "photos" / "sept_barre.jpg")
resultats = {}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
axes[0].imshow(photo.convert("RGB"))
axes[0].set_title("Photo prise au téléphone", fontsize=11)
for ax, (cle, titre, x) in zip(axes[1:], [("ancien", "Ancien traitement", vers_mnist(photo)),
                                          ("nouveau", "Nouveau traitement", photo_vers_mnist(photo))]):
    p = probabilites(modele, x[None])[0]
    q = probabilites(detecteur, x[None])[0]
    resultats[cle] = {"predit": int(p.argmax()), "confiance_pct": round(100 * float(p.max()), 1),
                      "detecteur_chiffre_pct": round(100 * float(q[1]), 1)}
    ax.imshow(x, cmap="gray")
    verdict = (f"lu « {p.argmax()} » à {100 * p.max():.0f} %" if q[1] >= 0.5
               else "refusé : « pas un chiffre »")
    ax.set_title(f"{titre}\n{verdict}".replace(".", ","), fontsize=11)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(RACINE / "figures" / "figA4_photo.png", dpi=200)

# Fausses photos : les 12 polices du test externe, trait fin sur papier gris ombré.
POLICES = ["segoepr.ttf", "segoesc.ttf", "Inkfree.ttf", "LHANDW.TTF", "BRADHITC.TTF", "ITCKRIST.TTF",
           "mvboli.ttf", "comic.ttf", "arial.ttf", "times.ttf", "georgia.ttf", "verdana.ttf"]
rng = np.random.default_rng(503)
W, H = 1200, 1600
yy, xx = np.mgrid[0:H, 0:W]
justes, erreurs = 0, []
for fichier in POLICES:
    police = ImageFont.truetype("C:/Windows/Fonts/" + fichier, 300)
    for c in range(10):
        papier = 175 - 40 * (yy / H) * (xx / W) + rng.normal(0, 3, (H, W))   # ombre dans un coin + grain
        trace = Image.new("L", (W, H), 0)
        dessin = ImageDraw.Draw(trace)
        dessin.text((W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-200, 200)), str(c), fill=255,
                    font=police, anchor="mm")
        dessin.ellipse((200, 300, 206, 306), fill=255)                         # une tache
        gris = np.clip(papier * (1 - 0.65 * np.asarray(trace) / 255), 0, 255).astype(np.uint8)
        p = probabilites(modele, photo_vers_mnist(Image.fromarray(gris))[None])[0]
        justes += int(p.argmax() == c)
        if p.argmax() != c:
            erreurs.append({"police": fichier, "chiffre": c, "predit": int(p.argmax())})
resultats["fausses_photos"] = {"n": len(POLICES) * 10, "justes": justes, "erreurs": erreurs}
(RACINE / "resultats" / "chiffres_cles_photo.json").write_text(json.dumps(resultats, indent=2), encoding="utf-8")
print(resultats)
