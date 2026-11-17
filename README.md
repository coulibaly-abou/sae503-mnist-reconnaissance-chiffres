# SAE 5.03 – Data mining sur MNIST : reconnaissance de chiffres manuscrits

Projet de BUT Science des Données (3e année, 2026-2027) : exploration, qualité, détection d'anomalies et classification du jeu de données MNIST.

Auteurs : Ayi AMAVIGAN, Aboubakar COULIBALY.

Article de référence : Rana *et al.* (2023), *Comparison of the Error Rates of MNIST Datasets Using Different Type of Machine Learning Model*, DOI [10.5281/zenodo.8010602](https://doi.org/10.5281/zenodo.8010602) (PDF non inclus dans le dépôt).

| Dossier | Contenu |
|---|---|
| `docs/` | Sujet, grille de notation, article de référence (Rana *et al.*, 2023) |
| `data/` | Fichiers MNIST officiels (IDX gzippés) ; `data/external/` = fonds Natural Earth téléchargés par le notebook |
| `notebooks/SAE503_MNIST_exploration.ipynb` | Parties 3 (présentation), 4 (qualité) et 5 (exploration) |
| `notebooks/SAE503_MNIST_anomalies.ipynb` | Partie 6 (détection d'anomalies, étiquettes suspectes, décision) |
| `scripts/mnist_io.py` | Lecture des fichiers IDX (utilisé par le notebook de la partie 6) |
| `scripts/flowchart.py` | Génère le flowchart des tâches (`figures/fig1_flowchart.png`) |
| `figures/` | Toutes les figures du rapport (PNG 200 dpi) |
| `resultats/` | Tableaux CSV + `chiffres_cles.json` (chiffres cités dans le rapport) |
| `rapport/` | Rapport Word `Nom1_Nom2_Nom3_rapport_SAE503.docx` |

## Reproduire

```bash
python -m venv .venv            # de préférence hors de Google Drive
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/jupyter lab notebooks/SAE503_MNIST_exploration.ipynb   # "Run all"
.venv/Scripts/python scripts/flowchart.py
```

Le notebook peut être lancé depuis `notebooks/` ou depuis la racine ; il régénère `figures/` et `resultats/`.
La première exécution télécharge les fonds de carte Natural Earth (~1 Mo) pour la carte (partie 5.iv). Durée totale : ~1 à 2 min (t-SNE inclus).

## Avancement

- [x] Flowchart des tâches · [ ] Gantt (annexe 1) · [ ] dépôt commun / rôles
- [x] 2. Bibliographie (résumé de l'article + recherche)
- [x] 3. Présentation des données · [x] 4. Qualité · [x] 5. Exploration (stats, visualisations, dataviz radiale, carte)
- [x] 6. Anomalies (78 étiquettes suspectes retirées de l'apprentissage : `resultats/t6_3_indices_train_retires.csv`) · [ ] 7. Bootstrap · [ ] 8. Classification (≥ 5 modèles, grid search CV)
- [ ] Résumé, résumé graphique, recommandations, conclusion

Dans le rapport, les passages **surlignés en jaune** sont à compléter.
