"""Détecteur « chiffre / pas un chiffre » utilisé comme filtre par l'application.

Le classifieur principal (10 classes) choisit toujours un chiffre, même devant un
gribouillis, et parfois avec 100 % de confiance. Ce second réseau, de même
architecture mais à 2 sorties, apprend à reconnaître ce qui n'est pas un chiffre :

- classe 1 « chiffre »      : images MNIST d'apprentissage ;
- classe 0 « pas un chiffre » : dessins générés au hasard (gribouillis, étoiles,
  zigzags, formes pleines, hachures, lettres, deux chiffres côte à côte),
  passés par la même mise au format MNIST que les dessins de l'application.

Usage : python scripts/detecteur_non_chiffre.py   (environ 10 min)
Sortie : app/detecteur.pt, resultats/chiffres_cles_detecteur.json, figures/fig8_7_non_chiffres.png
"""
import json
import math
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
import cnn  # noqa: E402
from mnist_io import charger_mnist  # noqa: E402
from pretraitement import vers_mnist  # noqa: E402

GRAINE = 503
rng = np.random.default_rng(GRAINE)
torch.set_num_threads(12)
TAILLE = 280
LETTRES = "ACEFHKMNPRTUVWXYacefhkmnrwxy#@&%*"   # sans les lettres qui ressemblent à des chiffres (O, I, S, Z, B, g, q…)
# Polices d'APPRENTISSAGE (lettres négatives et chiffres positifs). Les 12 polices de l'évaluation
# (scripts/test_chiffres_externes.py) en sont exclues pour que le test reste honnête.
POLICES = [p for p in ["calibri.ttf", "tahoma.ttf", "trebuc.ttf", "consola.ttf", "cour.ttf", "pala.ttf",
                       "seguisb.ttf", "segoeui.ttf", "Gabriola.ttf", "FRSCRIPT.TTF", "lucon.ttf"]
           if (Path("C:/Windows/Fonts") / p).exists()]


def trait(d, pts, w):
    d.line(pts, fill=255, width=w, joint="curve")
    for x, y in pts:
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)


def point():
    return tuple(rng.uniform(20, TAILLE - 20, 2))


def non_chiffre(X_mnist):
    """Un dessin qui n'est pas un chiffre, au format 28 x 28 (ou None si vide)."""
    img = Image.new("L", (TAILLE, TAILLE), 0)
    d = ImageDraw.Draw(img)
    w = int(rng.integers(8, 32))
    genre = rng.choice(["gribouillis", "etoile", "zigzag", "plein", "hachures", "lettre", "deux_chiffres",
                        "spirale", "multi_traits"],
                       p=[0.2, 0.12, 0.1, 0.1, 0.1, 0.13, 0.12, 0.06, 0.07])
    if genre == "gribouillis":
        trait(d, [point() for _ in range(rng.integers(7, 16))], w)
    elif genre == "etoile":
        c, r, n = np.array(point()) * 0.4 + TAILLE * 0.3, rng.uniform(70, 130), rng.integers(3, 7)
        for k in range(n):
            a = math.pi * k / n + rng.uniform(-0.15, 0.15)
            trait(d, [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)),
                      (c[0] - r * math.cos(a), c[1] - r * math.sin(a))], w)
    elif genre == "zigzag":
        n = rng.integers(5, 10)
        ys = np.linspace(30, TAILLE - 30, n)
        trait(d, [(rng.choice([30, TAILLE - 30]) + rng.uniform(-15, 15), y) for y in ys], w)
    elif genre == "plein":
        x0, y0 = rng.uniform(5, 100, 2)
        x1, y1 = rng.uniform(180, 275, 2)
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=255)
    elif genre == "hachures":
        for y in np.arange(rng.uniform(15, 40), TAILLE - 15, rng.uniform(w * 1.2, w * 3)):
            trait(d, [(rng.uniform(10, 60), y), (rng.uniform(220, 270), y + rng.uniform(-20, 20))], w)
    elif genre == "lettre":
        police = ImageFont.truetype(str(Path("C:/Windows/Fonts") / rng.choice(POLICES)), int(rng.integers(150, 230)))
        d.text((TAILLE / 2, TAILLE / 2), str(rng.choice(list(LETTRES))), fill=255, font=police, anchor="mm")
    elif genre == "deux_chiffres":
        a, b = X_mnist[rng.integers(len(X_mnist), size=2)]
        img = Image.fromarray(np.hstack([a, b])).resize((TAILLE, TAILLE // 2)).convert("L")
        fond = Image.new("L", (TAILLE, TAILLE), 0)
        fond.paste(img, (0, TAILLE // 4))
        img = fond
    elif genre == "spirale":
        c = np.array([TAILLE / 2, TAILLE / 2]) + rng.uniform(-30, 30, 2)
        t = np.linspace(0, rng.uniform(4, 7) * math.pi, 60)
        r = np.linspace(5, rng.uniform(90, 125), 60)
        trait(d, [(c[0] + ri * math.cos(ti), c[1] + ri * math.sin(ti)) for ri, ti in zip(r, t)], w)
    else:  # plusieurs traits séparés
        for _ in range(rng.integers(4, 9)):
            p = np.array(point())
            trait(d, [tuple(p), tuple(p + rng.uniform(-70, 70, 2))], w)
    return vers_mnist(img, inverser=False)


Xtr, ytr, Xte, yte = charger_mnist(RACINE / "data")


def chiffre_style_trait(x):
    """Chiffre MNIST redessiné avec un trait épais et régulier, comme à la souris."""
    img = Image.fromarray(((x > 60) * 255).astype(np.uint8)).resize((TAILLE, TAILLE), Image.NEAREST)
    from PIL import ImageFilter
    k = int(rng.choice([3, 5, 7, 9]))
    img = img.filter(ImageFilter.MaxFilter(k)) if rng.random() < 0.7 else img.filter(ImageFilter.MinFilter(k))
    return vers_mnist(img, inverser=False)


def chiffre_police():
    pol = ImageFont.truetype(str(Path("C:/Windows/Fonts") / rng.choice(POLICES)), int(rng.integers(150, 230)))
    img = Image.new("L", (TAILLE, TAILLE), 0)
    ImageDraw.Draw(img).text((TAILLE / 2, TAILLE / 2), str(rng.integers(10)), fill=255, font=pol, anchor="mm")
    return vers_mnist(img.rotate(rng.uniform(-12, 12)), inverser=False)
N = 30000


def generer(n):
    sortie = []
    while len(sortie) < n:
        x = non_chiffre(Xtr)
        if x is not None:
            sortie.append(x)
    return np.stack(sortie)


print("génération des non-chiffres…", flush=True)
neg_app, neg_val = generer(N), generer(3000)
idx = rng.choice(len(Xtr), N, replace=False)
pos_mnist = Xtr[idx[:20000]]
pos_trait = np.stack([z for z in (chiffre_style_trait(x) for x in Xtr[idx[20000:]]) if z is not None])
pos_pol = np.stack([z for z in (chiffre_police() for _ in range(4000)) if z is not None])
pos_app = np.concatenate([pos_mnist, pos_trait, pos_pol])
print(f"positifs : {len(pos_mnist)} MNIST, {len(pos_trait)} au trait épais, {len(pos_pol)} en police", flush=True)
X = np.concatenate([pos_app, neg_app])
y = np.concatenate([np.ones(len(pos_app), int), np.zeros(N, int)])

print("apprentissage…", flush=True)
modele = cnn.entrainer(X, y, epoques=4, lr=3e-3, dropout=0.25, augmentation=True, graine=GRAINE,
                       sorties=2, journal=print)
torch.save({"etat": modele.state_dict(), "dropout": 0.25, "sorties": 2}, RACINE / "app" / "detecteur.pt")

# Validation : vrais chiffres jamais vus (test MNIST, polices) et non-chiffres jamais vus
p_test = cnn.probabilites(modele, Xte)[:, 1]
p_neg = cnn.probabilites(modele, neg_val)[:, 1]
polices = []
for f in ["segoepr.ttf", "segoesc.ttf", "Inkfree.ttf", "LHANDW.TTF", "BRADHITC.TTF", "ITCKRIST.TTF", "mvboli.ttf",
          "comic.ttf", "arial.ttf", "times.ttf", "georgia.ttf", "verdana.ttf"]:
    if (Path("C:/Windows/Fonts") / f).exists():
        pol = ImageFont.truetype(str(Path("C:/Windows/Fonts") / f), 200)
        for c in range(10):
            img = Image.new("L", (300, 300), 0)
            ImageDraw.Draw(img).text((150, 150), str(c), fill=255, font=pol, anchor="mm")
            polices.append(vers_mnist(img, inverser=False))
p_pol = cnn.probabilites(modele, np.stack(polices))[:, 1]
SEUIL = 0.5
res = {"seuil": SEUIL,
       "chiffres_test_acceptes_pct": round(100 * float((p_test >= SEUIL).mean()), 2),
       "chiffres_polices_acceptes_pct": round(100 * float((p_pol >= SEUIL).mean()), 1),
       "non_chiffres_rejetes_pct": round(100 * float((p_neg < SEUIL).mean()), 2),
       "n_test": int(len(p_test)), "n_polices": int(len(p_pol)), "n_non_chiffres_validation": int(len(p_neg))}
(RACINE / "resultats" / "chiffres_cles_detecteur.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))

fig, axes = plt.subplots(3, 12, figsize=(9, 2.6))
for ax, x, p in zip(axes.flat, neg_val[:36], p_neg[:36]):
    ax.imshow(x, cmap="gray_r"); ax.axis("off")
    ax.set_title(f"{100 * (1 - p):.0f} %", fontsize=7, color="#2e7d32" if p < SEUIL else "#c62828")
fig.suptitle("Exemples de « non-chiffres » générés (probabilité de ne pas être un chiffre)", fontsize=9)
fig.savefig(RACINE / "figures" / "fig8_7_non_chiffres.png", dpi=200, bbox_inches="tight")
