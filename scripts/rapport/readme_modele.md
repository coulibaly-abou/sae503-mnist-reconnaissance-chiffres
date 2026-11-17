# SAÉ 5.03 – Data mining sur MNIST : reconnaissance de chiffres manuscrits

[![Ouvrir l'application](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mnist-reconnaissance-chiffres.streamlit.app/)

**Application en ligne : https://mnist-reconnaissance-chiffres.streamlit.app/** : dessinez un chiffre, le modèle le reconnaît.

Projet de BUT Science des Données (3e année, 2026-2027, IUT de Carcassonne, enseignant : Sébastien Pinel).
Auteurs : **Ayi AMAVIGAN** et **Aboubakar COULIBALY**.

On part des 70 000 images de chiffres manuscrits de MNIST pour suivre tout le processus de data mining :
présentation et qualité des données, exploration, détection d'anomalies, bootstrap, puis comparaison de
sept modèles de classification. Le meilleur modèle, un réseau convolutif, est utilisable dans une
**application Streamlit** : vous dessinez un chiffre et il le reconnaît.

## Résultats

| Modèle (appris sur 59 922 images) | Erreur sur les 10 000 images de test |
|---|---|
{TABLEAU}

- Intervalle de confiance bootstrap à 95 % de l'erreur du réseau convolutif : **{IC}**.
- Article de référence (Rana *et al.*, 2023) : 2,08 % avec un SVM réglé directement sur le jeu de test.
- Chiffres écrits avec 12 polices de Windows : {EXT} % reconnus. Chiffres dessinés à la souris dans l'application : {DESSIN}.

Le rapport complet est dans [`rapport/AMAVIGAN_COULIBALY_rapport_SAE503.pdf`](rapport/AMAVIGAN_COULIBALY_rapport_SAE503.pdf).

## Tester l'application

En ligne : **https://mnist-reconnaissance-chiffres.streamlit.app/**

Ou en local :

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Trois onglets : **Dessiner** (souris ou doigt), **Images MNIST** (images du test tirées au hasard) et
**Importer une image** (photo d'un chiffre écrit au feutre). L'onglet **Le modèle** reprend les résultats.

**Et si on dessine n'importe quoi ?** Le classifieur a appris à choisir entre 0 et 9 : face à un
gribouillis, il répond quand même un chiffre, parfois avec 100 % de confiance. L'application filtre donc
chaque dessin avant de répondre :

1. un dessin qui remplit plus de 45 % de l'image est refusé (aucun chiffre de MNIST ne dépasse 44,8 %) ;
2. un second réseau, le **détecteur « chiffre / pas un chiffre »** (`scripts/detecteur_non_chiffre.py`),
   entraîné sur MNIST et sur 30 000 dessins générés (gribouillis, étoiles, zigzags, formes pleines,
   lettres, deux chiffres côte à côte), refuse ce qui n'est pas un chiffre. Validation : {DET_OK} % des
   chiffres du test MNIST et {DET_POL} % des chiffres en police acceptés, {DET_NEG} % des non-chiffres refusés ;
3. une prédiction sous 70 % de confiance est signalée comme incertaine.

Déploiement sur Streamlit Community Cloud : choisir ce dépôt, la branche `main` et le fichier `app/app.py`
(les dépendances sont lues dans `app/requirements.txt`).

## Organisation

| Dossier / fichier | Contenu |
|---|---|
| `docs/` | Sujet et grille de notation |
| `data/` | Fichiers MNIST officiels (IDX gzippés) ; `data/external/` : fonds Natural Earth pour la carte |
| `notebooks/SAE503_MNIST_exploration.ipynb` | Parties 3 à 5 : présentation, qualité, exploration |
| `notebooks/SAE503_MNIST_anomalies.ipynb` | Partie 6 : détection d'anomalies et étiquettes suspectes |
| `scripts/modelisation.py` | Parties 7 et 8 : bootstrap, sélection ACP, 7 modèles, grid search, test |
| `scripts/cnn.py`, `scripts/pretraitement.py` | Réseau convolutif (PyTorch) et mise au format MNIST d'un dessin |
| `scripts/test_chiffres_externes.py` | Test du modèle sur des chiffres écrits avec des polices Windows |
| `scripts/detecteur_non_chiffre.py` | Détecteur « chiffre / pas un chiffre » utilisé comme filtre par l'application |
| `scripts/gantt.py`, `scripts/resume_graphique.py`, `scripts/flowchart.py` | Figures du rapport |
| `scripts/rapport/` | Construction du rapport Word à partir des résultats (python-docx, Word pour le PDF) |
| `app/` | Application Streamlit, modèle entraîné (`modele_cnn.pt`), détecteur (`detecteur.pt`) et 1 000 images de test |
| `figures/`, `resultats/` | Figures (PNG 200 dpi), tableaux CSV et chiffres clés JSON |
| `rapport/` | Rapport final (docx et pdf) et version de travail du 16 octobre |

## Reproduire

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/jupyter lab notebooks/SAE503_MNIST_exploration.ipynb   # parties 3-5, "Run all"
.venv/Scripts/jupyter lab notebooks/SAE503_MNIST_anomalies.ipynb     # partie 6
.venv/Scripts/python scripts/modelisation.py                          # parties 7-8 (~1 h sur 12 cœurs)
.venv/Scripts/python scripts/test_chiffres_externes.py
```

Graines aléatoires fixées (503). Le PDF de l'article de référence n'est pas inclus (droits d'auteur) :
Rana, Kabir et Sobur (2023), DOI [10.5281/zenodo.8010602](https://doi.org/10.5281/zenodo.8010602).
