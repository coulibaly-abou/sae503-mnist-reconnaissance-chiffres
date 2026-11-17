"""Mise au format MNIST d'un chiffre dessiné ou photographié.

On reproduit la chaîne de construction de MNIST (partie 3.2) : le chiffre est
recadré, réduit pour tenir dans une boîte de 20 x 20 pixels sans être déformé,
puis centré selon son centre de masse dans une grille de 28 x 28.
"""
import numpy as np
from PIL import Image, ImageOps


def vers_mnist(image, inverser=None):
    """Image PIL (ou tableau) -> tableau (28, 28) uint8, encre claire sur fond noir.

    Renvoie None si l'image est vide.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    if image.mode in ("RGBA", "LA"):
        fond = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(fond, image.convert("RGBA"))
    gris = np.asarray(image.convert("L"), dtype=np.float32)

    # Une photo de chiffre noir sur papier blanc doit être inversée.
    if inverser is None:
        inverser = np.median(gris) > 127
    if inverser:
        gris = 255 - gris

    # On efface le bruit de fond léger (papier, compression).
    gris[gris < 0.25 * gris.max()] = 0
    if gris.max() == 0:
        return None
    lignes = np.where(gris.max(axis=1) > 0)[0]
    colonnes = np.where(gris.max(axis=0) > 0)[0]
    chiffre = gris[lignes[0]:lignes[-1] + 1, colonnes[0]:colonnes[-1] + 1]

    # Réduction dans une boîte de 20 x 20 en gardant les proportions.
    h, w = chiffre.shape
    k = 20.0 / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    petit = Image.fromarray(chiffre.astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    petit = np.asarray(petit, dtype=np.float32)

    # Centrage par le centre de masse dans la grille 28 x 28.
    grille = np.zeros((28, 28), dtype=np.float32)
    haut, gauche = (28 - nh) // 2, (28 - nw) // 2
    grille[haut:haut + nh, gauche:gauche + nw] = petit
    total = grille.sum()
    yy, xx = np.indices(grille.shape)
    cy, cx = (yy * grille).sum() / total, (xx * grille).sum() / total
    dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
    grille = np.roll(np.roll(grille, dy, axis=0), dx, axis=1)

    grille = grille / grille.max() * 255
    return np.clip(grille, 0, 255).astype(np.uint8)
