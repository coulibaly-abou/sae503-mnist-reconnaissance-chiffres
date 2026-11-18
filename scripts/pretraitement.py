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


# Épaisseur relative du trait dans MNIST (2 x aire / périmètre, rapportée à la
# hauteur du chiffre) : médiane 0,13 mesurée sur 2 000 images du jeu de test.
EPAISSEUR_MNIST = 0.13


def _otsu(valeurs):
    hist, bords = np.histogram(valeurs, bins=256)
    centres = (bords[:-1] + bords[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def photo_vers_mnist(image):
    """Photo d'un chiffre (téléphone, scan) -> tableau (28, 28) uint8, ou None.

    Une photo n'est pas un dessin propre : le papier est gris, l'éclairage n'est
    pas uniforme, le trait du stylo est très fin par rapport à l'image et il y a
    des petites taches. On (1) soustrait le fond estimé localement, (2) sépare
    l'encre du papier par le seuil d'Otsu, (3) ne garde que les gros morceaux de
    tracé, (4) épaissit le trait jusqu'à l'épaisseur typique de MNIST, puis on
    applique la même mise au format que pour un dessin.
    """
    from PIL import ImageFilter
    from scipy import ndimage as ndi

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA"):
        fond = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(fond, image.convert("RGBA"))
    gris = image.convert("L")
    gris.thumbnail((500, 500))

    # 1. Fond local : le papier est la partie claire, on l'estime par un filtre max
    #    (qui efface le trait) puis un flou ; l'encre est ce qui est plus sombre.
    clair = np.median(np.asarray(gris)) > 127
    filtre = ImageFilter.MaxFilter(15) if clair else ImageFilter.MinFilter(15)
    fond = np.asarray(gris.filter(filtre).filter(ImageFilter.GaussianBlur(10)), dtype=np.float32)
    g = np.asarray(gris, dtype=np.float32)
    encre = np.clip(fond - g if clair else g - fond, 0, None)

    # 2. Seuil d'Otsu, avec un minimum pour ne pas transformer le grain du papier en encre.
    if encre.max() < 20:
        return None
    masque = encre > max(_otsu(encre[encre > 2]), 20)

    # 3. On retire les taches : on garde les morceaux d'au moins 10 % du plus gros.
    etiq, n = ndi.label(masque, structure=np.ones((3, 3)))
    if n == 0:
        return None
    aires = ndi.sum(masque, etiq, range(1, n + 1))
    masque = np.isin(etiq, 1 + np.where(aires >= 0.10 * aires.max())[0])

    # 4. Épaississement du trait jusqu'à l'épaisseur relative de MNIST.
    ys, xs = np.where(masque)
    masque = masque[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    masque = np.pad(masque, 20)
    taille = max(np.ptp(ys), np.ptp(xs)) + 1
    perimetre = (masque & ~ndi.binary_erosion(masque)).sum()
    epaisseur = 2 * masque.sum() / max(perimetre, 1)
    manque = EPAISSEUR_MNIST * taille - epaisseur
    if manque > 0:
        masque = ndi.distance_transform_edt(~masque) <= manque / 2
    trace = ndi.gaussian_filter(masque.astype(np.float32), sigma=max(1.0, 0.02 * taille)) * 255
    return vers_mnist(np.clip(trace, 0, 255).astype(np.uint8), inverser=False)
