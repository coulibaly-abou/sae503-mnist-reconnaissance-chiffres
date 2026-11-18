"""Application Streamlit : dessinez un chiffre, le réseau convolutif le reconnaît.

Lancement local : streamlit run app/app.py
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "scripts"))
from cnn import PetitCNN, probabilites  # noqa: E402
from pretraitement import photo_vers_mnist, vers_mnist  # noqa: E402

st.set_page_config(page_title="Reconnaissance de chiffres manuscrits", page_icon=":material/draw:",
                   layout="wide")


@st.cache_resource
def charger_modele():
    sauvegarde = torch.load(ICI / "modele_cnn.pt", map_location="cpu", weights_only=True)
    modele = PetitCNN(dropout=sauvegarde["dropout"])
    modele.load_state_dict(sauvegarde["etat"])
    modele.eval()
    return modele, sauvegarde["accuracy_test_pct"]


@st.cache_resource
def charger_detecteur():
    """Second réseau à 2 sorties : 0 = pas un chiffre, 1 = chiffre (scripts/detecteur_non_chiffre.py)."""
    chemin = ICI / "detecteur.pt"
    if not chemin.exists():
        return None
    sauvegarde = torch.load(chemin, map_location="cpu", weights_only=True)
    det = PetitCNN(dropout=sauvegarde["dropout"], sorties=2)
    det.load_state_dict(sauvegarde["etat"])
    det.eval()
    return det


@st.cache_data
def charger_exemples():
    d = np.load(ICI / "exemples_test.npz")
    return d["images"], d["etiquettes"], d["indices"]


@st.cache_data
def charger_resultats():
    return json.loads((ICI / "resultats.json").read_text(encoding="utf-8"))


modele, acc_test = charger_modele()
detecteur = charger_detecteur()


# Garde-fous : le modèle choisit toujours entre 0 et 9, il ne sait pas dire « ce n'est pas un chiffre ».
# - Aucun chiffre de MNIST n'a plus de 44,8 % de pixels encrés (moyenne 19 %) : au-delà de 45 %,
#   le dessin est trop rempli pour être un chiffre.
# - Moins de 0,4 % des vrais chiffres du test sont prédits avec moins de 70 % de confiance.
ENCRE_MAX_PCT = 45
CONFIANCE_MIN = 0.70


def afficher_prediction(image28, colonne_image=None, verite=None):
    """Affiche l'image vue par le modèle, la classe prédite et les probabilités."""
    p = probabilites(modele, image28[None])[0]
    classe = int(p.argmax())
    encre_pct = 100 * (image28 > 0).mean()
    g, d = st.columns([1, 2])
    with g:
        st.image(Image.fromarray(image28).resize((168, 168), Image.NEAREST),
                 caption="Ce que voit le modèle (28 x 28)")
    with d:
        if verite is None and encre_pct > ENCRE_MAX_PCT:
            st.markdown("## Ce n'est pas un chiffre")
            st.error(f"Le dessin remplit {encre_pct:.0f} % de l'image, alors qu'un chiffre manuscrit en occupe "
                     "environ 20 % (jamais plus de 45 % dans MNIST). Effacez et écrivez un seul chiffre.",
                     icon=":material/block:")
            st.caption(f"Pour information, la forme la plus proche serait un {classe} "
                       f"({100 * p[classe]:.0f} %), ce qui n'a pas de sens ici.")
            return
        p_chiffre = float(probabilites(detecteur, image28[None])[0, 1]) if detecteur is not None else 1.0
        if verite is None and p_chiffre < 0.5:
            st.markdown("## Ce n'est pas un chiffre")
            st.error(f"Le détecteur estime à {100 * (1 - p_chiffre):.0f} % que ce dessin n'est pas un chiffre "
                     "(gribouillis, lettre, plusieurs chiffres…). Effacez et écrivez un seul chiffre.",
                     icon=":material/block:")
            st.caption(f"Pour information, le classifieur aurait répondu {classe} ({100 * p[classe]:.0f} %) : "
                       "il choisit toujours entre 0 et 9, d'où ce filtre.")
            return
        if verite is None and p[classe] < CONFIANCE_MIN:
            st.markdown("## Je ne reconnais pas de chiffre")
            st.warning(f"Le modèle n'est sûr qu'à {100 * p[classe]:.0f} % (seuil : 70 %). Le chiffre le plus "
                       f"proche serait un **{classe}**. Écrivez le chiffre plus grand, bien au centre, "
                       "d'un seul tracé net.", icon=":material/help:")
        else:
            texte = f"## Chiffre reconnu : **{classe}**"
            if verite is not None:
                texte += "  :green[(correct)]" if classe == verite else f"  :red[(la bonne réponse était {verite})]"
            st.markdown(texte)
            st.markdown(f"Confiance du modèle : **{100 * p[classe]:.1f} %**".replace(".", ","))
        df = pd.DataFrame({"chiffre": [str(i) for i in range(10)], "probabilité (%)": 100 * p})
        st.bar_chart(df, x="chiffre", y="probabilité (%)", height=220)


# ---------------------------------------------------------------------------
# Canevas de dessin (composant Streamlit v2 écrit à la main)
# ---------------------------------------------------------------------------
HTML = """
<div class="zone">
  <canvas width="280" height="280"></canvas>
  <div class="boutons"><button type="button" class="effacer">Effacer</button></div>
</div>
"""
CSS = """
.zone { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
canvas { background: #000; border-radius: 8px; touch-action: none; cursor: crosshair;
         border: 2px solid var(--st-primary-color); }
button { font: inherit; padding: 4px 14px; border-radius: 6px; cursor: pointer;
         border: 1px solid var(--st-border-color, #ccc); background: var(--st-secondary-background-color);
         color: var(--st-text-color); }
"""
JS = """
export default function(component) {
  const { parentElement, setStateValue, data } = component;
  const canvas = parentElement.querySelector('canvas');
  if (canvas.dataset.pret) { return; }
  canvas.dataset.pret = '1';
  const ctx = canvas.getContext('2d');
  const epaisseur = (data && data.epaisseur) || 18;
  const vider = () => { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height); };
  vider();
  let dessine = false, x0 = 0, y0 = 0;
  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
  };
  canvas.addEventListener('pointerdown', (e) => {
    dessine = true; [x0, y0] = pos(e); canvas.setPointerCapture(e.pointerId);
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x0, y0, epaisseur / 2, 0, 2 * Math.PI); ctx.fill();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dessine) return;
    const [x, y] = pos(e);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = epaisseur; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x, y); ctx.stroke();
    x0 = x; y0 = y;
  });
  const fin = () => {
    if (!dessine) return;
    dessine = false;
    setStateValue('image', canvas.toDataURL('image/png'));
  };
  canvas.addEventListener('pointerup', fin);
  canvas.addEventListener('pointerleave', fin);
  parentElement.querySelector('.effacer').addEventListener('click', () => {
    vider(); setStateValue('image', null);
  });
}
"""
canevas = st.components.v2.component("canevas_chiffre", html=HTML, css=CSS, js=JS)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Reconnaissance de chiffres manuscrits")
st.caption("SAÉ 5.03 · Ayi AMAVIGAN et Aboubakar COULIBALY · réseau convolutif entraîné sur MNIST, "
           + f"{acc_test:.2f}".replace(".", ",") + " % de bonnes réponses sur les 10 000 images de test")

onglet_dessin, onglet_mnist, onglet_photo, onglet_modele = st.tabs(
    [":material/draw: Dessiner", ":material/grid_view: Images MNIST", ":material/photo_camera: Importer une image",
     ":material/analytics: Le modèle"])

with onglet_dessin:
    st.markdown("Dessinez **un seul chiffre** dans le cadre noir, à la souris ou au doigt. "
                "La prédiction apparaît dès que vous relâchez. Le modèle a appris à choisir entre 0 et 9 : "
                "si le dessin ne ressemble à aucun chiffre (cadre rempli, gribouillis), l'application le signale "
                "au lieu d'inventer une réponse.")
    g, d = st.columns([1, 1.6])
    with g:
        epaisseur = st.slider("Épaisseur du trait", 10, 30, 18)
        res = canevas(key="dessin", data={"epaisseur": epaisseur}, default={"image": None},
                      on_image_change=lambda: None)
    with d:
        url = res.image if res is not None else None
        if url:
            image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            image28 = vers_mnist(image, inverser=False)
            if image28 is None:
                st.info("Le cadre est vide.")
            else:
                afficher_prediction(image28)
        else:
            st.info("En attente d'un dessin.", icon=":material/gesture:")

with onglet_mnist:
    images, etiquettes, indices = charger_exemples()
    st.markdown("Ces images viennent du **jeu de test** : le modèle ne les a jamais vues pendant l'apprentissage.")
    g, d = st.columns([1, 1])
    with g:
        chiffre = st.selectbox("Chiffre à tirer", ["au hasard"] + [str(i) for i in range(10)])
    with d:
        st.write("")
        st.write("")
        tirer = st.button("Tirer une image", icon=":material/casino:")
    if tirer or "tirage" not in st.session_state:
        candidats = np.arange(len(etiquettes)) if chiffre == "au hasard" else np.where(etiquettes == int(chiffre))[0]
        st.session_state.tirage = int(np.random.choice(candidats))
    i = st.session_state.tirage
    st.caption(f"Image n° {indices[i]} du jeu de test")
    afficher_prediction(images[i], verite=int(etiquettes[i]))

with onglet_photo:
    st.markdown("Photographiez un chiffre écrit sur une feuille (stylo ou feutre). Le fond gris, les ombres "
                "et le trait fin sont corrigés automatiquement ; évitez seulement d'avoir d'autres traits dans l'image.")
    fichier = st.file_uploader("Image (PNG ou JPG)", type=["png", "jpg", "jpeg"])
    if fichier:
        image28 = photo_vers_mnist(Image.open(fichier))
        if image28 is None:
            st.error("Aucun tracé détecté dans l'image.")
        else:
            afficher_prediction(image28)

with onglet_modele:
    r = charger_resultats()
    st.subheader("Résultats sur les 10 000 images de test")
    tab = pd.DataFrame([{"Modèle": k, "Accuracy (%)": v["acc"], "Erreur (%)": v["err"], "F1 macro (%)": v["f1"],
                         "Apprentissage (s)": v["fit_s"]} for k, v in r["test"].items()])
    st.dataframe(tab, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tab.columns[1:4]}
                 | {"Apprentissage (s)": st.column_config.NumberColumn(format="%.0f")})
    b = r["bootstrap_cnn"]
    st.markdown(f"Intervalle de confiance bootstrap à 95 % de l'accuracy du réseau convolutif : "
                f"**[{b[1]:.2f} % ; {b[2]:.2f} %]**.".replace(".", ","))
    st.subheader("Taux d'erreur du réseau convolutif par chiffre")
    st.bar_chart(pd.DataFrame({"chiffre": list(r["cnn_erreur_par_classe"].keys()),
                               "erreur (%)": list(r["cnn_erreur_par_classe"].values())}),
                 x="chiffre", y="erreur (%)", height=240)
    st.markdown(
        "**Architecture.** Deux blocs de deux convolutions 3 x 3 (32 puis 64 filtres) suivis d'un max-pooling, "
        "puis une couche dense de 128 neurones. Hyperparamètres choisis par grid search en validation croisée "
        f"(taux d'apprentissage {r['cnn_meilleurs_params']['lr']:g}, dropout {r['cnn_meilleurs_params']['dropout']}). "
        "Le modèle final est entraîné sur les 59 922 images conservées après la détection d'anomalies, "
        "avec de petites rotations et translations aléatoires pour mieux supporter les dessins à main levée.")
