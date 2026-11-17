"""Parties 7 (bootstrap) et 8 (classification) de la SAÉ 5.03.

Usage : python scripts/modelisation.py          (environ 40 min sur un portable, 12 cœurs)
        python scripts/modelisation.py --rapide (petits échantillons, pour vérifier le code)

Entrées  : data/*.gz, resultats/t6_3_indices_train_retires.csv (partie 6)
Sorties  : resultats/t7_*.csv, resultats/t8_*.csv, resultats/chiffres_cles_p7_p8.json,
           figures/fig7_*.png, figures/fig8_*.png, app/modele_cnn.pt, app/exemples_test.npz
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import cnn  # noqa: E402
from mnist_io import charger_mnist  # noqa: E402

RAPIDE = "--rapide" in sys.argv
FIG, RES, APP = RACINE / "figures", RACINE / "resultats", RACINE / "app"
for d in (FIG, RES, APP):
    d.mkdir(exist_ok=True)
GRAINE = 503
torch.set_num_threads(12)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "figure.dpi": 100,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
BLEU, ORANGE, GRIS = "#1f4e79", "#c55a11", "#7f7f7f"
CHIFFRES = {}
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Données : on retire les 78 étiquettes suspectes (décision de la partie 6)
# ---------------------------------------------------------------------------
Xtr_img, ytr, Xte_img, yte = charger_mnist(RACINE / "data")
retires = pd.read_csv(RES / "t6_3_indices_train_retires.csv")["indice"].to_numpy()
garde = np.ones(len(ytr), bool)
garde[retires] = False
Xtr_brut_img, ytr_brut = Xtr_img, ytr          # version sans nettoyage (pour comparer)
Xtr_img, ytr = Xtr_img[garde], ytr[garde]
log(f"apprentissage : {len(ytr)} images (après retrait de {len(retires)}), test : {len(yte)}")

utiles = Xtr_brut_img.reshape(len(Xtr_brut_img), -1).max(axis=0) > 0   # 717 pixels non constants
def plat(images):
    return images.reshape(len(images), -1)[:, utiles].astype(np.float32) / 255.0

Xtr, Xte = plat(Xtr_img), plat(Xte_img)
CHIFFRES["n_train"] = int(len(ytr))
CHIFFRES["n_pixels_utiles"] = int(utiles.sum())

# Sous-échantillon stratifié pour les comparaisons en validation croisée
N_SOUS = 2000 if RAPIDE else 12000
rng = np.random.default_rng(GRAINE)
sous = np.sort(np.concatenate([rng.choice(np.where(ytr == c)[0], N_SOUS // 10, replace=False)
                               for c in range(10)]))
Xs, ys, Xs_img = Xtr[sous], ytr[sous], Xtr_img[sous]
CV5 = StratifiedKFold(5, shuffle=True, random_state=GRAINE)
CV3 = StratifiedKFold(3, shuffle=True, random_state=GRAINE)
CHIFFRES["n_sous_echantillon"] = int(len(ys))

# ---------------------------------------------------------------------------
# 8.i  Sélection des prédicteurs : combien de composantes principales garder ?
# ---------------------------------------------------------------------------
log("8.i  sélection du nombre de composantes (KNN et SVM en validation croisée)")
lignes = []
for k in [10, 20, 35, 50, 87, 150, None]:
    for nom, clf in [("KNN", KNeighborsClassifier(3, weights="distance", n_jobs=-1)),
                     ("SVM RBF", SVC())]:
        etapes = [PCA(k, random_state=GRAINE)] if k else []
        pipe = make_pipeline(*etapes, clf)
        scores = []
        for a, v in CV5.split(Xs, ys):
            pipe.fit(Xs[a], ys[a])
            scores.append(accuracy_score(ys[v], pipe.predict(Xs[v])))
        lignes.append({"composantes": k or int(utiles.sum()), "modele": nom,
                       "accuracy_pct": 100 * np.mean(scores), "ecart_type_pct": 100 * np.std(scores)})
sel = pd.DataFrame(lignes)
sel.to_csv(RES / "t8_1_selection_acp.csv", index=False, float_format="%.2f")
meilleur_k = {m: int(g.loc[g.accuracy_pct.idxmax(), "composantes"]) for m, g in sel.groupby("modele")}
log(f"     meilleur nombre de composantes : {meilleur_k}")
K_ACP = 50   # retenu : au niveau du meilleur score, avec 14 fois moins de variables
CHIFFRES["acp_k_retenu"] = K_ACP
CHIFFRES["selection"] = {f"{r.modele}_{r.composantes}": round(r.accuracy_pct, 2) for r in sel.itertuples()}

fig, ax = plt.subplots(figsize=(6.5, 3.2))
for (nom, g), c in zip(sel.groupby("modele"), [BLEU, ORANGE]):
    ax.errorbar(g.composantes, g.accuracy_pct, yerr=g.ecart_type_pct, marker="o", color=c,
                capsize=3, label=nom)
ax.axvline(K_ACP, ls="--", color=GRIS, lw=1)
ax.text(K_ACP * 1.05, ax.get_ylim()[0] + 0.3, "50 composantes retenues", color=GRIS, fontsize=9)
ax.set_xscale("log")
ax.set_xticks(sel.composantes.unique())
ax.set_xticklabels([str(v) for v in sel.composantes.unique()])
ax.set_xlabel("Nombre de composantes principales (717 = tous les pixels utiles)")
ax.set_ylabel("Accuracy en VC 5 plis (%)")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.savefig(FIG / "fig8_1_selection_acp.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8.ii-iii  Sept modèles de familles différentes, comparés en VC 5 plis
# ---------------------------------------------------------------------------
def acp(clf):
    return make_pipeline(PCA(K_ACP, random_state=GRAINE), clf)

MODELES = {
    "Bayésien naïf gaussien": acp(GaussianNB()),
    "Régression logistique": LogisticRegression(C=0.1, max_iter=300),
    "Forêt aléatoire": RandomForestClassifier(300, n_jobs=-1, random_state=GRAINE),
    "KNN (k = 3)": acp(KNeighborsClassifier(3, weights="distance", n_jobs=-1)),
    "SVM RBF": acp(SVC()),
    "Perceptron multicouche": MLPClassifier((256,), alpha=1e-4, max_iter=60, early_stopping=True,
                                            random_state=GRAINE),
}
EPOQUES_CV = 2 if RAPIDE else 6

log("8.ii comparaison des modèles en validation croisée à 5 plis")
lignes = []
for nom, modele in list(MODELES.items()) + [("Réseau convolutif (CNN)", None)]:
    acc, f1, t_fit, t_pred = [], [], [], []
    for a, v in CV5.split(Xs, ys):
        t = time.time()
        if modele is None:
            m = cnn.entrainer(Xs_img[a], ys[a], epoques=EPOQUES_CV, graine=GRAINE)
            t_fit.append(time.time() - t); t = time.time()
            pred = cnn.probabilites(m, Xs_img[v]).argmax(1)
        else:
            modele.fit(Xs[a], ys[a])
            t_fit.append(time.time() - t); t = time.time()
            pred = modele.predict(Xs[v])
        t_pred.append(time.time() - t)
        acc.append(accuracy_score(ys[v], pred))
        f1.append(f1_score(ys[v], pred, average="macro"))
    lignes.append({"modele": nom, "accuracy_pct": 100 * np.mean(acc), "ecart_type_pct": 100 * np.std(acc),
                   "f1_macro_pct": 100 * np.mean(f1), "temps_apprentissage_s": np.mean(t_fit),
                   "temps_prediction_s": np.mean(t_pred)})
    log(f"     {nom:28s} accuracy {lignes[-1]['accuracy_pct']:.2f} %")
comp = pd.DataFrame(lignes).sort_values("accuracy_pct", ascending=False)
comp.to_csv(RES / "t8_2_comparaison_vc.csv", index=False, float_format="%.2f")
CHIFFRES["comparaison_vc"] = {r.modele: {"acc": round(r.accuracy_pct, 2), "sd": round(r.ecart_type_pct, 2),
                                         "f1": round(r.f1_macro_pct, 2), "fit_s": round(r.temps_apprentissage_s, 1)}
                              for r in comp.itertuples()}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
c = comp.iloc[::-1]
couleurs = [ORANGE if "CNN" in n else (BLEU if "SVM" in n else GRIS) for n in c.modele]
a1.barh(c.modele, c.accuracy_pct, xerr=c.ecart_type_pct, color=couleurs, capsize=2)
for y, v in enumerate(c.accuracy_pct):
    a1.text(v + 0.6, y, f"{v:.1f}".replace(".", ","), va="center", fontsize=9)
a1.set_xlim(max(0, c.accuracy_pct.min() - 10), 103)
a1.set_xlabel("Accuracy en VC 5 plis (%)")
a2.barh(c.modele, c.temps_apprentissage_s, color=couleurs)
a2.set_xscale("log")
a2.set_yticks([])
a2.set_xlabel("Temps d'apprentissage par pli (s, échelle log)")
for a in (a1, a2):
    a.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig8_2_comparaison_modeles.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8.v  Ajustement des hyperparamètres (grid search + VC 3 plis)
# ---------------------------------------------------------------------------
log("8.v  grid search du SVM RBF")
grille_svm = {"svc__C": [1, 5, 10, 50], "svc__gamma": ["scale", 0.01, 0.02, 0.04, 0.06, 0.08]}
gs = GridSearchCV(acp(SVC()), grille_svm, cv=CV3, n_jobs=-1, scoring="accuracy")
gs.fit(Xs, ys)
res_gs = pd.DataFrame(gs.cv_results_)
tab = res_gs.pivot_table(index="param_svc__C", columns=res_gs.param_svc__gamma.astype(str),
                         values="mean_test_score") * 100
tab.to_csv(RES / "t8_3_gridsearch_svm.csv", float_format="%.2f")
svm_params = {k.replace("svc__", ""): v for k, v in gs.best_params_.items()}
CHIFFRES["svm_meilleurs_params"] = {k: (v if isinstance(v, str) else float(v)) for k, v in svm_params.items()}
CHIFFRES["svm_gs_acc"] = round(100 * gs.best_score_, 2)
CHIFFRES["svm_defaut_acc_cv3"] = round(float(tab.loc[1, "scale"]), 2)
log(f"     meilleurs paramètres SVM : {svm_params} ({100 * gs.best_score_:.2f} %)")

log("8.v  grid search du CNN (VC 3 plis)")
lignes = []
for lr in [1e-3, 3e-3]:
    for dropout in [0.1, 0.25, 0.4]:
        scores = []
        for a, v in CV3.split(Xs, ys):
            m = cnn.entrainer(Xs_img[a], ys[a], epoques=EPOQUES_CV, lr=lr, dropout=dropout, graine=GRAINE)
            scores.append(accuracy_score(ys[v], cnn.probabilites(m, Xs_img[v]).argmax(1)))
        lignes.append({"lr": lr, "dropout": dropout, "accuracy_pct": 100 * np.mean(scores),
                       "ecart_type_pct": 100 * np.std(scores)})
        log(f"     lr={lr:g} dropout={dropout} : {lignes[-1]['accuracy_pct']:.2f} %")
gs_cnn = pd.DataFrame(lignes)
gs_cnn.round({"accuracy_pct": 2, "ecart_type_pct": 2}).to_csv(RES / "t8_4_gridsearch_cnn.csv", index=False)
best = gs_cnn.loc[gs_cnn.accuracy_pct.idxmax()]
cnn_params = {"lr": float(best.lr), "dropout": float(best.dropout)}
CHIFFRES["cnn_meilleurs_params"] = cnn_params
CHIFFRES["cnn_gs_acc"] = round(float(best.accuracy_pct), 2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
im = a1.imshow(tab.values, cmap="Blues", aspect="auto")
a1.set_xticks(range(tab.shape[1]), tab.columns)
a1.set_yticks(range(tab.shape[0]), tab.index)
a1.set_xlabel("gamma")
a1.set_ylabel("C")
a1.set_title("SVM RBF (ACP 50)", fontsize=11)
for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
        v = tab.values[i, j]
        a1.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=9,
                color="white" if v > np.nanpercentile(tab.values, 60) else "black")
pv = gs_cnn.pivot(index="lr", columns="dropout", values="accuracy_pct")
a2.imshow(pv.values, cmap="Oranges", aspect="auto")
a2.set_xticks(range(pv.shape[1]), pv.columns)
a2.set_yticks(range(pv.shape[0]), [f"{v:g}" for v in pv.index])
a2.set_xlabel("taux de dropout")
a2.set_ylabel("taux d'apprentissage")
a2.set_title("Réseau convolutif", fontsize=11)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        a2.text(j, i, f"{pv.values[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9)
fig.suptitle("Accuracy moyenne en validation croisée à 3 plis (%)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig8_3_gridsearch.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8.vi  Apprentissage sur les 59 922 images, prédiction sur les 10 000 du test
# ---------------------------------------------------------------------------
log("8.vi apprentissage complet et prédiction sur le test")
if RAPIDE:
    Xf, yf, Xf_img = Xs, ys, Xs_img
else:
    Xf, yf, Xf_img = Xtr, ytr, Xtr_img
lignes, PRED = [], {}
for nom, modele in MODELES.items():
    t = time.time()
    modele.fit(Xf, yf)
    t_fit = time.time() - t
    PRED[nom] = modele.predict(Xte)
    lignes.append({"modele": nom, "parametres": "par défaut", "temps_apprentissage_s": t_fit})
    log(f"     {nom:28s} test {100 * accuracy_score(yte, PRED[nom]):.2f} %")

t = time.time()
svm_final = acp(SVC(C=svm_params["C"], gamma=svm_params["gamma"]))
svm_final.fit(Xf, yf)
PRED["SVM RBF ajusté"] = svm_final.predict(Xte)
lignes.append({"modele": "SVM RBF ajusté", "parametres": f"C = {svm_params['C']}, gamma = {svm_params['gamma']}",
               "temps_apprentissage_s": time.time() - t})

EPOQUES_FINAL = 2 if RAPIDE else 12
t = time.time()
cnn_final = cnn.entrainer(Xf_img, yf, epoques=EPOQUES_FINAL, augmentation=True, graine=GRAINE, journal=log,
                          **cnn_params)
P_CNN = cnn.probabilites(cnn_final, Xte_img)
PRED["Réseau convolutif ajusté"] = P_CNN.argmax(1)
lignes.append({"modele": "Réseau convolutif ajusté",
               "parametres": f"lr = {cnn_params['lr']:g}, dropout = {cnn_params['dropout']}, "
                             f"{EPOQUES_FINAL} époques, augmentation",
               "temps_apprentissage_s": time.time() - t})
test = pd.DataFrame(lignes)
test["accuracy_pct"] = [100 * accuracy_score(yte, PRED[n]) for n in test.modele]
test["erreur_pct"] = 100 - test.accuracy_pct
test["f1_macro_pct"] = [100 * f1_score(yte, PRED[n], average="macro") for n in test.modele]
test = test.sort_values("accuracy_pct", ascending=False)
test.to_csv(RES / "t8_5_resultats_test.csv", index=False, float_format="%.2f")
CHIFFRES["test"] = {r.modele: {"acc": round(r.accuracy_pct, 2), "err": round(r.erreur_pct, 2),
                               "f1": round(r.f1_macro_pct, 2), "fit_s": round(r.temps_apprentissage_s, 1)}
                    for r in test.itertuples()}
log(f"     SVM ajusté {CHIFFRES['test']['SVM RBF ajusté']['acc']} % | "
    f"CNN ajusté {CHIFFRES['test']['Réseau convolutif ajusté']['acc']} %")

torch.save({"etat": cnn_final.state_dict(), "dropout": cnn_params["dropout"],
            "accuracy_test_pct": CHIFFRES["test"]["Réseau convolutif ajusté"]["acc"]}, APP / "modele_cnn.pt")

# Effet du retrait des 78 étiquettes suspectes (même SVM, même test)
if not RAPIDE:
    log("     effet du nettoyage (SVM ajusté avec les 60 000 images d'origine)")
    svm_brut = acp(SVC(C=svm_params["C"], gamma=svm_params["gamma"]))
    svm_brut.fit(plat(Xtr_brut_img), ytr_brut)
    p_brut = svm_brut.predict(Xte)
    CHIFFRES["nettoyage"] = {"acc_sans_retrait": round(100 * accuracy_score(yte, p_brut), 2),
                             "acc_avec_retrait": CHIFFRES["test"]["SVM RBF ajusté"]["acc"],
                             "n_erreurs_sans": int((p_brut != yte).sum()),
                             "n_erreurs_avec": int((PRED["SVM RBF ajusté"] != yte).sum())}
    log(f"     {CHIFFRES['nettoyage']}")

# Matrice de confusion du CNN (en % de la classe réelle) et erreurs
mc = confusion_matrix(yte, PRED["Réseau convolutif ajusté"])
mc_pct = 100 * mc / mc.sum(axis=1, keepdims=True)
pd.DataFrame(mc_pct).to_csv(RES / "t8_6_confusion_cnn_pct.csv", float_format="%.2f")
hors_diag = mc.copy()
np.fill_diagonal(hors_diag, 0)
paires = sorted(((hors_diag[i, j], i, j) for i in range(10) for j in range(10) if hors_diag[i, j]), reverse=True)[:4]
CHIFFRES["cnn_confusions"] = [f"{i} lu comme {j} ({n})" for n, i, j in paires]
CHIFFRES["cnn_n_erreurs"] = int(hors_diag.sum())
err_classe = 100 - np.diag(mc_pct)
CHIFFRES["cnn_erreur_par_classe"] = {str(c): round(float(e), 2) for c, e in enumerate(err_classe)}

fig, ax = plt.subplots(figsize=(5.2, 4.4))
masque = np.where(np.eye(10, dtype=bool), np.nan, mc_pct)
im = ax.imshow(masque, cmap="Reds", vmin=0)
for i in range(10):
    for j in range(10):
        v = mc_pct[i, j]
        if i == j:
            ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=7.5, color=BLEU,
                    fontweight="bold")
        elif v >= 0.1:
            ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=7.5)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
fig.colorbar(im, fraction=0.046, label="% de la classe réelle (hors diagonale)")
fig.savefig(FIG / "fig8_4_confusion_cnn.png")
plt.close(fig)

idx_err = np.where(PRED["Réseau convolutif ajusté"] != yte)[0]
conf = P_CNN.max(1)
idx_err = idx_err[np.argsort(-conf[idx_err])][:16]
fig, axes = plt.subplots(2, 8, figsize=(9, 2.8))
for ax, i in zip(axes.flat, idx_err):
    ax.imshow(Xte_img[i], cmap="gray_r")
    ax.set_title(f"vrai {yte[i]} → {PRED['Réseau convolutif ajusté'][i]}\n({100 * conf[i]:.0f} %)", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "fig8_5_erreurs_cnn.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7.  Bootstrap : intervalles de confiance à 95 %
# ---------------------------------------------------------------------------
log("7.   bootstrap")
B = 300 if RAPIDE else 5000
rng = np.random.default_rng(GRAINE)
ok_cnn = (PRED["Réseau convolutif ajusté"] == yte).astype(float)
ok_svm = (PRED["SVM RBF ajusté"] == yte).astype(float)
n = len(yte)
tirages = rng.integers(0, n, size=(B, n))
boot_cnn = 100 * ok_cnn[tirages].mean(1)
boot_svm = 100 * ok_svm[tirages].mean(1)
boot_diff = boot_cnn - boot_svm

def ic(v):
    return np.percentile(v, [2.5, 97.5])

def ic_normal(p):
    e = 1.96 * np.sqrt(p * (1 - p) / n)
    return 100 * (p - e), 100 * (p + e)

lignes = []
for nom, ok, bt in [("Réseau convolutif ajusté", ok_cnn, boot_cnn), ("SVM RBF ajusté", ok_svm, boot_svm)]:
    b, h = ic(bt)
    nb, nh = ic_normal(ok.mean())
    lignes.append({"parametre": f"Accuracy {nom}", "estimation": 100 * ok.mean(), "ic_bas": b, "ic_haut": h,
                   "ic_normal_bas": nb, "ic_normal_haut": nh})
b, h = ic(boot_diff)
lignes.append({"parametre": "Écart d'accuracy CNN - SVM (points)", "estimation": 100 * (ok_cnn.mean() - ok_svm.mean()),
               "ic_bas": b, "ic_haut": h})
CHIFFRES["boot_diff_part_positive_pct"] = round(100 * float((boot_diff > 0).mean()), 2)

# Taux d'erreur du CNN par classe (bootstrap stratifié dans chaque classe)
err_ic = []
for c in range(10):
    oc = 1 - ok_cnn[yte == c]
    bt = 100 * oc[rng.integers(0, len(oc), size=(B, len(oc)))].mean(1)
    b, h = ic(bt)
    err_ic.append((c, 100 * oc.mean(), b, h))
    lignes.append({"parametre": f"Erreur du CNN sur le chiffre {c}", "estimation": 100 * oc.mean(), "ic_bas": b, "ic_haut": h})

# Quantité d'encre moyenne par classe (descripteur de la partie 5)
encre = Xtr_img.reshape(len(Xtr_img), -1).sum(1) / 255.0
encre_ic = []
for c in range(10):
    e = encre[ytr == c]
    bt = e[rng.integers(0, len(e), size=(min(B, 2000), len(e)))].mean(1)
    b, h = ic(bt)
    encre_ic.append((c, e.mean(), b, h))
    lignes.append({"parametre": f"Encre moyenne du chiffre {c} (pixels pleins)", "estimation": e.mean(),
                   "ic_bas": b, "ic_haut": h})
boot = pd.DataFrame(lignes)
boot.to_csv(RES / "t7_1_intervalles_bootstrap.csv", index=False, float_format="%.3f")
CHIFFRES["bootstrap_B"] = B
CHIFFRES["bootstrap"] = {r.parametre: [round(r.estimation, 2), round(r.ic_bas, 2), round(r.ic_haut, 2)]
                         for r in boot.itertuples()}
CHIFFRES["ic_normal_cnn"] = [round(v, 2) for v in ic_normal(ok_cnn.mean())]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
a1.hist(boot_svm, bins=40, color=BLEU, alpha=0.75, label="SVM RBF ajusté")
a1.hist(boot_cnn, bins=40, color=ORANGE, alpha=0.75, label="Réseau convolutif")
for bt, col in [(boot_svm, BLEU), (boot_cnn, ORANGE)]:
    for v in ic(bt):
        a1.axvline(v, color=col, ls="--", lw=1)
a1.set_xlabel(f"Accuracy sur le test rééchantillonné (%), B = {B}")
a1.set_ylabel("Nombre de rééchantillons")
a1.legend(frameon=False, fontsize=9)
err = np.array(err_ic)
a2.errorbar(err[:, 0], err[:, 1], yerr=[err[:, 1] - err[:, 2], err[:, 3] - err[:, 1]], fmt="o", color=ORANGE,
            capsize=3)
a2.set_xticks(range(10))
a2.set_xlabel("Chiffre réel")
a2.set_ylabel("Taux d'erreur du CNN (%)")
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig7_1_bootstrap.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Exemples du test pour l'application (100 par classe, ~0,5 Mo)
# ---------------------------------------------------------------------------
idx_app = np.sort(np.concatenate([np.where(yte == c)[0][:100] for c in range(10)]))
np.savez_compressed(APP / "exemples_test.npz", images=Xte_img[idx_app], etiquettes=yte[idx_app], indices=idx_app)
(APP / "resultats.json").write_text(json.dumps({
    "test": CHIFFRES["test"], "comparaison_vc": CHIFFRES["comparaison_vc"],
    "cnn_meilleurs_params": cnn_params, "svm_meilleurs_params": CHIFFRES["svm_meilleurs_params"],
    "bootstrap_cnn": CHIFFRES["bootstrap"]["Accuracy Réseau convolutif ajusté"],
    "cnn_erreur_par_classe": CHIFFRES["cnn_erreur_par_classe"],
}, ensure_ascii=False, indent=1), encoding="utf-8")

CHIFFRES["date_execution"] = time.strftime("%Y-%m-%d")
CHIFFRES["duree_min"] = round((time.time() - T0) / 60, 1)
nom_json = "chiffres_cles_p7_p8_rapide.json" if RAPIDE else "chiffres_cles_p7_p8.json"
(RES / nom_json).write_text(json.dumps(CHIFFRES, ensure_ascii=False, indent=1), encoding="utf-8")
log("terminé")
