"""Complète le rapport SAÉ 5.03 : parties 7-8, résumé, recommandations, conclusion, annexes.

Part de rapport/Nom1_Nom2_Nom3_rapport_SAE503.docx (version du 16/10, jamais modifiée)
et écrit rapport/AMAVIGAN_COULIBALY_rapport_SAE503.docx.
"""
import json
import sys
from pathlib import Path

import docx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import outils_docx  # noqa: E402
from outils_docx import Curseur, remplacer_paragraphe, supprimer, times_partout  # noqa: E402

P = Path(__file__).resolve().parents[2]
FIG, RES = P / "figures", P / "resultats"
C = json.loads((RES / "chiffres_cles_p7_p8.json").read_text(encoding="utf-8"))
C6 = json.loads((RES / "chiffres_cles_p6.json").read_text(encoding="utf-8"))
EXT = json.loads((RES / "chiffres_cles_test_externe.json").read_text(encoding="utf-8"))
DESSIN = json.loads((RES / "chiffres_cles_dessins.json").read_text(encoding="utf-8"))
DET = json.loads((RES / "chiffres_cles_detecteur.json").read_text(encoding="utf-8"))


def v(x, d=1):
    """Nombre à la française."""
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace(" ", "\u202f")


doc = docx.Document(P / "rapport" / "Nom1_Nom2_Nom3_rapport_SAE503.docx")
paras = doc.paragraphs
FIGS = {}
ANCIENNES = [q for q in doc.paragraphs if q.style.name == "Legende figure" and "SEQ Figure" in q._p.xml]
NOUVEAUX_TEXTES = []
ANCIENS_P = {q._p for q in doc.paragraphs}


def trouver(debut, style=None):
    for p in doc.paragraphs:
        if p.text.startswith(debut) and (style is None or p.style.name == style):
            return p
    raise KeyError(debut)


# ---------------------------------------------------------------------------
# Chiffres utilisés dans le texte
# ---------------------------------------------------------------------------
T = C["test"]
cnn, svm = T["Réseau convolutif ajusté"], T["SVM RBF ajusté"]
cv = C["comparaison_vc"]
ordre_cv = list(cv.keys())                     # déjà trié par accuracy décroissante
B = C["bootstrap"]
b_cnn, b_svm = B["Accuracy Réseau convolutif ajusté"], B["Accuracy SVM RBF ajusté"]
b_diff = B["Écart d'accuracy CNN - SVM (points)"]
err_cl = {int(k): val for k, val in C["cnn_erreur_par_classe"].items()}
pire = sorted(err_cl, key=err_cl.get, reverse=True)
meilleur = sorted(err_cl, key=err_cl.get)
ic_err = {c: B[f"Erreur du CNN sur le chiffre {c}"] for c in range(10)}
encre = {c: B[f"Encre moyenne du chiffre {c} (pixels pleins)"] for c in range(10)}
net = C.get("nettoyage", {})
sp = C["svm_meilleurs_params"]
cp = C["cnn_meilleurs_params"]
sel = C["selection"]
gs_svm = pd.read_csv(RES / "t8_3_gridsearch_svm.csv", index_col=0)
gs_cnn = pd.read_csv(RES / "t8_4_gridsearch_cnn.csv")
n_err_cnn = C["cnn_n_erreurs"]
conf = C["cnn_confusions"]
B_N = C["bootstrap_B"]

# ---------------------------------------------------------------------------
# Page de garde
# ---------------------------------------------------------------------------
remplacer_paragraphe(trouver("Version de travail"), "Décembre 2026")
remplacer_paragraphe(trouver("IUT de Perpignan"), "IUT de Carcassonne – Année universitaire 2026-2027")
p = trouver("Exploration, qualité et modélisation")
remplacer_paragraphe(p, "Exploration, qualité et modélisation du jeu de données MNIST")

# ---------------------------------------------------------------------------
# Résumé et résumé graphique
# ---------------------------------------------------------------------------
resume = (
    "Reconnaître un chiffre écrit à la main paraît simple pour un humain, beaucoup moins pour une machine. "
    "Nous avons étudié ce problème sur MNIST, le jeu de données de référence du domaine : 70 000 images de "
    "chiffres de 28 x 28 pixels, écrits par des employés du Census Bureau et des lycéens américains. "
    "Notre point de départ est l'article de Rana et al. (2023), qui obtient 2,08 % d'erreur avec un SVM "
    "mais règle ses modèles directement sur le jeu de test. "
    "Les contrôles de qualité n'ont révélé ni valeur manquante, ni doublon, ni déséquilibre entre classes. "
    "L'exploration montre une information très redondante : 87 composantes principales suffisent pour 90 % "
    "de la variance des 784 pixels. "
    f"La détection d'anomalies, qui combine quatre méthodes, a isolé {C6['consensus_n']} images atypiques, "
    f"que nous avons gardées, et {C6['suspects_n']} étiquettes probablement fausses, que nous avons retirées "
    "de l'apprentissage. "
    f"Nous avons ensuite comparé sept modèles de familles différentes en validation croisée à 5 plis. "
    f"Le réseau convolutif ({v(cv['Réseau convolutif (CNN)']['acc'], 1)} % de bonnes réponses) et le SVM à "
    f"noyau gaussien ({v(cv['SVM RBF']['acc'], 1)} %) dominent nettement, loin devant le bayésien naïf "
    f"({v(cv['Bayésien naïf gaussien']['acc'], 1)} %). Après un réglage par grid search, le réseau convolutif "
    f"ne se trompe que sur {v(cnn['err'], 2)} % des 10 000 images de test, contre {v(svm['err'], 2)} % pour "
    f"le SVM. Le bootstrap donne un intervalle de confiance à 95 % de [{v(100 - b_cnn[2], 2)} % ; "
    f"{v(100 - b_cnn[1], 2)} %] pour cette erreur et confirme l'avantage du réseau convolutif sur le SVM. "
    f"Nous divisons ainsi par {v(2.08 / cnn['err'], 0)} l'erreur de l'article, avec une évaluation plus "
    "honnête. "
    "Nous recommandons d'utiliser un réseau convolutif entraîné avec de légères déformations des images, et "
    "de vérifier ses performances sur de vraies écritures avant tout usage. C'est ce que permet l'application "
    "web que nous avons développée : on y dessine un chiffre et le modèle le reconnaît en direct."
)
remplacer_paragraphe(trouver("À compléter – résumé d'environ"), resume)

p = trouver("À compléter – visuel de synthèse")
cur = Curseur(p)
cur.image(FIG / "fig0_resume_graphique.png", 17)
leg = cur.texte("*Résumé graphique : de la donnée brute au modèle retenu.*")
leg.alignment = 1
supprimer(p)

# ---------------------------------------------------------------------------
# Introduction et partie 1 (réécriture plus directe)
# ---------------------------------------------------------------------------
remplacer_paragraphe(trouver("Ce rapport met en œuvre un processus complet"), (
    "Ce rapport applique un processus complet de data mining à MNIST. Il part de l'article de Rana, Kabir et "
    "Sobur (2023), qui compare le taux d'erreur de trois familles de modèles : SVM, KNN et réseau de neurones. "
    "Nous avions deux objectifs. Le premier était de bien connaître les données : les décrire, contrôler leur "
    "qualité et repérer les images douteuses. Le second était de construire plusieurs modèles de "
    "classification, de les régler par validation croisée, puis de vérifier sur des images jamais vues qu'ils "
    "reconnaissent bien le chiffre écrit."))
remplacer_paragraphe(trouver("Après la présentation de l'organisation"), (
    "Nous présentons d'abord l'organisation du projet (partie 1) et la bibliographie (partie 2). Les parties "
    "3 à 5 décrivent les données, leur qualité et leur exploration. La partie 6 traite des anomalies, la "
    "partie 7 du bootstrap et la partie 8 de la classification. Nous terminons par des recommandations et une "
    "conclusion. Une petite application web, dans laquelle on dessine un chiffre que le modèle doit "
    "reconnaître, accompagne le rapport."))
remplacer_paragraphe(trouver("Le diagramme de Gantt figure en annexe 1"), (
    "Le diagramme de Gantt se trouve en annexe 1. Nous y avons découpé la SAÉ en neuf étapes sur onze "
    "semaines, du 28 septembre au 11 décembre 2026, pour 29 h 45 de travail planifié : 17 h 15 en autonomie, "
    "10 h 30 encadrées et 2 h d'évaluation. Les livrables (rapport de groupe et badges individuels) sont "
    "déposés sur Moodle."))
remplacer_paragraphe(trouver("À compléter – préciser l'outil de dépôt commun"), (
    "Tout le projet est rangé dans un dossier Google Drive partagé, SAE_DATAMING, organisé comme un projet de "
    "code : docs/ (sujet, article), data/ (fichiers MNIST), notebooks/ et scripts/ (code), figures/ et "
    "resultats/ (sorties) et rapport/. Une copie versionnée, qui contient aussi l'application, est publiée sur "
    "GitHub : github.com/coulibaly-abou/sae503-mnist-reconnaissance-chiffres. Ayi AMAVIGAN a surtout pris en "
    "charge la présentation, la qualité et l'exploration des données ainsi que la détection d'anomalies "
    "(parties 3 à 6). Aboubakar COULIBALY s'est occupé du bootstrap, de la classification, de l'application et "
    "de la mise en forme du rapport (parties 7 et 8). La bibliographie, les choix de méthode et la relecture "
    "ont été faits ensemble."))
remplacer_paragraphe(trouver("La figure 1 décrit l'enchaînement des tâches"), (
    "La figure 1 montre l'enchaînement des tâches. Le projet passe par cinq phases : cadrage, compréhension "
    "des données, préparation, modélisation et restitution. Deux questions rythment le travail. Faut-il "
    "modifier le jeu de données après la détection d'anomalies ? La performance obtenue en validation croisée "
    "est-elle suffisante, ou faut-il revenir aux modèles ? Chaque étape produit des figures et des tableaux "
    "qui alimentent directement le rapport."))
remplacer_paragraphe(trouver("Regard critique."), (
    "**Regard critique.** La démarche est simple et lisible, mais l'étude a plusieurs faiblesses. Les "
    "hyperparamètres sont choisis directement sur le jeu de test, sans validation croisée, ce qui rend les "
    "erreurs annoncées un peu trop optimistes. Les grilles testées sont très petites, et les 9 % d'erreur du "
    "réseau ReLU laissent penser que son apprentissage n'a pas convergé. Les 27 % d'erreur du SVM linéaire "
    "surprennent aussi, puisque LeCun et al. (1998) obtenaient 8 à 12 % avec des classifieurs linéaires. "
    "Aucun réseau convolutif n'est testé, alors que c'est la référence pour ce type d'images. Enfin, le texte "
    "contient des maladresses : « MIST » au lieu de MNIST, un découpage 70/30 cité en conclusion alors que le "
    "découpage officiel 60 000/10 000 est utilisé, et des références sans lien avec le sujet. Ces limites ont "
    "guidé notre démarche : régler les modèles par validation croisée et garder le jeu de test pour la seule "
    "évaluation finale."))
remplacer_paragraphe(trouver("Jeu de test : inchangé."), (
    "**Jeu de test : inchangé.** La partie 8 mesure l'effet de ce retrait en comparant un même modèle appris "
    "avec et sans les images retirées."))

# ---------------------------------------------------------------------------
# Partie 7 : bootstrap
# ---------------------------------------------------------------------------
p7 = trouver("À compléter – intervalles de confiance bootstrap")
cur = Curseur(p7)
cur.texte(
    "Un taux d'erreur mesuré sur 10 000 images reste une estimation : avec d'autres images de test, on aurait "
    "obtenu un chiffre un peu différent. Le bootstrap (Efron et Tibshirani, 1993) permet de mesurer cette "
    "incertitude sans hypothèse sur la loi des données. Le principe est simple : on tire au hasard, avec "
    f"remise, 10 000 images parmi les 10 000 du test, on recalcule l'indicateur, et on recommence {v(B_N, 0)} "
    "fois. Les quantiles 2,5 % et 97,5 % des valeurs obtenues forment l'intervalle de confiance à 95 % "
    "(méthode des percentiles).")
cur.texte(
    "Nous l'avons appliqué à trois paramètres : l'accuracy des deux meilleurs modèles de la partie 8, l'écart "
    "entre les deux, calculé sur les mêmes tirages pour tenir compte du fait que les modèles sont jugés sur "
    "les mêmes images, et le taux d'erreur du réseau convolutif chiffre par chiffre. Nous l'avons aussi "
    "appliqué à la quantité d'encre moyenne de chaque classe, le descripteur de la partie 5.")
cur.legende("Tableau", f"Intervalles de confiance bootstrap à 95 % (B = {v(B_N, 0)} tirages, jeu de test)")
lignes = [
    ["Accuracy du réseau convolutif (%)", v(b_cnn[0], 2), f"[{v(b_cnn[1], 2)} ; {v(b_cnn[2], 2)}]"],
    ["Accuracy du SVM RBF ajusté (%)", v(b_svm[0], 2), f"[{v(b_svm[1], 2)} ; {v(b_svm[2], 2)}]"],
    ["Écart CNN moins SVM (points)", v(b_diff[0], 2), f"[{v(b_diff[1], 2)} ; {v(b_diff[2], 2)}]"],
    [f"Erreur du CNN sur le chiffre {pire[0]} (%)", v(ic_err[pire[0]][0], 2),
     f"[{v(ic_err[pire[0]][1], 2)} ; {v(ic_err[pire[0]][2], 2)}]"],
    [f"Erreur du CNN sur le chiffre {meilleur[0]} (%)", v(ic_err[meilleur[0]][0], 2),
     f"[{v(ic_err[meilleur[0]][1], 2)} ; {v(ic_err[meilleur[0]][2], 2)}]"],
    ["Encre moyenne du chiffre 1 (pixels pleins)", v(encre[1][0], 1), f"[{v(encre[1][1], 1)} ; {v(encre[1][2], 1)}]"],
    ["Encre moyenne du chiffre 0 (pixels pleins)", v(encre[0][0], 1), f"[{v(encre[0][1], 1)} ; {v(encre[0][2], 1)}]"],
]
cur.tableau(["Paramètre", "Estimation", "IC à 95 %"], lignes, [8.5, 3, 4.5], modele=doc.tables[2])
cur.image(FIG / "fig7_1_bootstrap.png", 16.5)
FIGS["boot"] = cur.legende("Figure", "Distribution bootstrap de l'accuracy des deux meilleurs modèles (gauche) et taux d'erreur "
                      "du réseau convolutif par chiffre avec son IC à 95 % (droite)")
cur.texte(
    f"L'accuracy du réseau convolutif est estimée à {v(b_cnn[0], 2)} %, avec un intervalle de "
    f"[{v(b_cnn[1], 2)} % ; {v(b_cnn[2], 2)} %]. L'intervalle est étroit parce que le test est grand et que "
    f"les erreurs sont rares. Il est presque identique à celui de l'approximation normale "
    f"([{v(C['ic_normal_cnn'][0], 2)} % ; {v(C['ic_normal_cnn'][1], 2)} %]), ce qui est attendu avec "
    "10 000 observations. Le point le plus utile est l'écart entre les deux modèles : son intervalle "
    f"[{v(b_diff[1], 2)} ; {v(b_diff[2], 2)}] ne contient pas zéro, et le réseau convolutif fait mieux que "
    f"le SVM dans {v(C['boot_diff_part_positive_pct'], 1)} % des tirages. Sa supériorité ne tient donc pas "
    "au hasard du découpage.")
cur.texte(
    f"Par chiffre, l'erreur va de {v(ic_err[meilleur[0]][0], 2)} % (chiffre {meilleur[0]}) à "
    f"{v(ic_err[pire[0]][0], 2)} % (chiffre {pire[0]}). Comme chaque classe ne compte qu'environ 1 000 images, "
    "les intervalles se chevauchent largement (§fig:boot§, à droite), et même ceux des deux extrêmes se "
    "recouvrent un peu : avec si peu d'erreurs, on ne peut pas affirmer qu'un chiffre est plus difficile qu'un "
    "autre. Enfin, les intervalles de l'encre moyenne sont très "
    f"courts (moins de {v(max(encre[c][2] - encre[c][1] for c in range(10)), 1)} pixel de large) : avec près "
    "de 6 000 images par classe, les écarts entre chiffres décrits en partie 5 sont solides.")
supprimer(p7)

# ---------------------------------------------------------------------------
# Partie 8 : classification
# ---------------------------------------------------------------------------
p8 = trouver("À compléter – sélection des prédicteurs")
cur = Curseur(p8)
cur.texte(
    "Le problème est une classification supervisée à dix classes : à partir des pixels d'une image, prédire "
    "le chiffre écrit. Tous les modèles apprennent sur les 59 922 images conservées après la partie 6 et sont "
    "évalués à la fin, une seule fois, sur les 10 000 images du test. Pour les comparaisons et les réglages, "
    f"nous avons travaillé sur un sous-échantillon stratifié de {v(C['n_sous_echantillon'], 0)} images "
    "d'apprentissage, ce qui garde des temps de calcul raisonnables sans changer le classement des modèles. "
    "Les métriques sont l'accuracy, adaptée puisque les classes sont équilibrées (partie 4), et le F1-score "
    "macro, qui donne le même poids à chaque chiffre.")
cur.titre("8.1 Sélection des prédicteurs")
k = C["acp_k_retenu"]
cur.texte(
    "Nous avons d'abord écarté les 67 pixels toujours vides, ce qui laisse 717 prédicteurs. C'est encore "
    "beaucoup, et ces pixels sont très corrélés entre voisins. Pour les modèles sensibles à la dimension (KNN, "
    "SVM, bayésien naïf), nous les avons résumés par une ACP. Le nombre de composantes a été choisi par "
    "validation croisée à 5 plis (§fig:acp§) : l'accuracy du KNN est la meilleure entre 35 et 50 "
    f"composantes ({v(sel['KNN_50'], 1)} % dans les deux cas), puis baisse quand on en ajoute, "
    "car le bruit des petites composantes perturbe le calcul des distances. Le SVM culmine à 50 composantes "
    f"({v(sel['SVM RBF_50'], 1)} %, contre {v(sel['SVM RBF_717'], 1)} % avec tous les pixels). Nous avons "
    f"retenu {k} composantes, qui gardent 82,5 % de la variance avec quatorze fois moins de variables. Les "
    "autres modèles (régression logistique, forêt aléatoire, perceptron, réseau convolutif) reçoivent les "
    "pixels bruts, car ils font eux-mêmes leur sélection : régularisation, choix des variables à chaque "
    "nœud, ou filtres appris.")
cur.image(FIG / "fig8_1_selection_acp.png", 11.5)
FIGS["acp"] = cur.legende("Figure", "Accuracy en validation croisée selon le nombre de composantes principales conservées")

cur.titre("8.2 Sept modèles comparés en validation croisée")
cur.texte(
    "Nous avons construit sept modèles qui couvrent les grandes familles de la classification : un modèle "
    "bayésien (bayésien naïf gaussien), un modèle linéaire (régression logistique multinomiale), un modèle à "
    "base d'arbres (forêt aléatoire de 300 arbres ; Breiman, 2001), un modèle fondé sur les instances (KNN, "
    "k = 3 pondéré par la distance, comme dans l'article), un SVM à noyau gaussien (Cortes et Vapnik, 1995), "
    "un perceptron multicouche (256 neurones cachés) et un réseau convolutif proche de LeNet (LeCun et al., "
    "1998), avec deux blocs de convolutions 3 x 3, du max-pooling et du dropout, entraîné avec Adam "
    "(Kingma et Ba, 2015). Ils sont comparés avec leurs réglages par défaut, sur les mêmes 5 plis.")
cur.legende("Tableau", "Comparaison des sept modèles en validation croisée à 5 plis (sous-échantillon "
                       "d'apprentissage)")
lignes = [[nom, v(r["acc"], 1), v(r["sd"], 1), v(r["f1"], 1), v(r["fit_s"], 1) if r["fit_s"] < 10 else v(r["fit_s"], 0)]
          for nom, r in cv.items()]
cur.tableau(["Modèle", "Accuracy (%)", "Écart-type (points)", "F1 macro (%)", "Apprentissage (s)"], lignes,
            [5.6, 2.6, 3, 2.6, 2.7], modele=doc.tables[2])
premier, second = ordre_cv[0], ordre_cv[1]
cur.texte(
    f"Le classement est net (tableau ci-dessus et §fig:comp§ en annexe 3). Le réseau convolutif arrive en tête "
    f"avec {v(cv['Réseau convolutif (CNN)']['acc'], 1)} % de bonnes réponses, suivi du SVM "
    f"({v(cv['SVM RBF']['acc'], 1)} %) puis du KNN ({v(cv['KNN (k = 3)']['acc'], 1)} %). Les écarts-types entre plis restent faibles (moins d'un demi-point pour "
    "la plupart des modèles) : le classement ne dépend pas du découpage. Le bayésien naïf ferme la marche, "
    "ce qui s'explique : il suppose les variables indépendantes, ce qui n'est pas le cas des pixels, même "
    "après l'ACP. La régression logistique, qui ne trace que des frontières linéaires, plafonne vers 91 %, "
    "un niveau proche des 88 % de LeCun et al. (1998). Le F1 macro suit exactement l'accuracy, signe "
    "qu'aucun chiffre n'est sacrifié. Le temps de calcul compte aussi : le réseau convolutif est de loin le "
    "plus long à entraîner, alors que le KNN n'apprend rien mais doit comparer chaque image à toutes les "
    "autres au moment de prédire.")

cur.titre("8.3 Choix du modèle et ajustement des hyperparamètres")
best_svm = gs_svm.values.max()
cur.texte(
    "Deux modèles se détachent : le réseau convolutif, le plus précis, et le SVM, le meilleur des modèles "
    "« classiques », celui que retient l'article. Nous les avons ajustés tous les deux par grid search (§fig:gs§) en "
    "validation croisée à 3 plis, sur le même sous-échantillon, pour voir si le réglage pouvait combler "
    f"l'écart. Pour le SVM, nous avons croisé quatre valeurs de C (1 à 50) et six valeurs de gamma (la valeur « scale » de scikit-learn, puis de 0,01 à 0,08). Le "
    f"meilleur couple est C = {v(sp['C'], 0)} et gamma = {sp['gamma'] if isinstance(sp['gamma'], str) else v(sp['gamma'], 2)}, "
    f"avec {v(C['svm_gs_acc'], 2)} % contre {v(C['svm_defaut_acc_cv3'], 2)} % pour le réglage par défaut. "
    "Pour le réseau convolutif, nous avons croisé deux taux d'apprentissage et trois taux de dropout. "
    f"Le meilleur réglage (taux d'apprentissage {v(cp['lr'], 3)}, dropout {v(cp['dropout'], 2)}) atteint "
    f"{v(C['cnn_gs_acc'], 2)} %. Le taux d'apprentissage compte plus que le dropout : passer de 0,001 à 0,003 "
    "fait gagner environ 0,6 point, alors que le dropout ne change le score que de quelques dixièmes.")
cur.image(FIG / "fig8_3_gridsearch.png", 15)
FIGS["gs"] = cur.legende("Figure", "Grid search : accuracy moyenne en validation croisée à 3 plis pour chaque combinaison "
                      "d'hyperparamètres")
cur.texte(
    "Le réglage fait gagner près d'un point au SVM et environ 0,7 point au réseau convolutif : l'écart entre les "
    "deux ne se resserre pas. Nous retenons donc le réseau convolutif. Pour l'entraînement final sur les 59 922 images, nous l'avons laissé apprendre "
    "plus longtemps (12 époques) et nous avons ajouté une augmentation de données : à chaque passage, les "
    "images sont légèrement tournées (± 10°), agrandies ou réduites (± 10 %) et décalées (± 2 pixels). Cette "
    "technique, classique sur MNIST (Simard et al., 2003), apprend au modèle que ces petites variations "
    "ne changent pas le chiffre. Elle est surtout utile pour l'application, où les chiffres dessinés sont "
    "moins bien centrés que ceux de MNIST.")

cur.titre("8.4 Prédiction sur le jeu de test")
cur.legende("Tableau", "Résultats sur les 10 000 images de test (modèles appris sur 59 922 images)")
lignes = [[nom, r_["param"], v(r_["acc"], 2), v(r_["err"], 2), v(r_["f1"], 2)] for nom, r_ in [
    (n, {**T[n], "param": "réglage choisi" if "ajusté" in n else "par défaut"}) for n in T]]
cur.tableau(["Modèle", "Paramètres", "Accuracy (%)", "Erreur (%)", "F1 macro (%)"], lignes,
            [5.6, 3.2, 2.6, 2.4, 2.6], modele=doc.tables[2])
cur.texte(
    f"Sur le test, le réseau convolutif ajusté ne se trompe que sur {n_err_cnn} images sur 10 000, soit "
    f"{v(cnn['err'], 2)} % d'erreur. Le SVM ajusté fait {v(svm['err'], 2)} %, mieux que les 2,08 % de "
    "l'article alors que nous n'avons jamais utilisé le test pour le régler. Tous les scores sont meilleurs "
    "qu'en validation croisée, car les modèles disposent cette fois de cinq fois plus d'images pour apprendre. "
    "L'ordre change peu : seul le perceptron multicouche remonte, de "
    f"{v(cv['Perceptron multicouche']['acc'], 1)} % à {v(T['Perceptron multicouche']['acc'], 1)} %, "
    "car c'est le modèle qui profite le plus d'un grand volume de données. Notre réseau convolutif fait mieux "
    "que LeNet-5 (0,95 %) et s'approche des meilleurs résultats publiés (environ 0,2 %), obtenus avec des "
    "réseaux plus profonds ou des comités de plusieurs réseaux.")
if net:
    cur.texte(
        f"Le retrait des {C6['suspects_n']} étiquettes suspectes (partie 6) a un effet très faible : appris sur "
        f"les 60 000 images d'origine, le même SVM fait {net['n_erreurs_sans']} erreurs sur le test, contre "
        f"{net['n_erreurs_avec']} après nettoyage. Ce résultat est cohérent : 78 images sur 60 000 pèsent peu, "
        "et un modèle bien régularisé tolère quelques erreurs d'étiquetage. Le nettoyage reste justifié, car il "
        "ne coûte rien et évite d'apprendre des associations fausses.")
cur.image(FIG / "fig8_4_confusion_cnn.png", 8.5)
FIGS["mc"] = cur.legende("Figure", "Matrice de confusion du réseau convolutif sur le test, en % de chaque classe réelle")
cur.texte(
    "La matrice de confusion (§fig:mc§) montre que les erreurs restantes sont dispersées. Les plus fréquentes sont "
    f"{conf[0].replace(')', ' images)')}, {conf[1].replace(')', ' images)')} et {conf[2].replace(')', ' images)')} : on retrouve des chiffres de forme proche, déjà repérés lors de "
    "l'exploration (corrélation des images moyennes, t-SNE). Les images mal classées (§fig:err§, annexe 3) sont "
    "souvent difficiles à lire, même pour nous.")

cur.titre("8.5 Test sur nos propres chiffres")
cur.texte(
    "Un bon score sur MNIST ne garantit pas que le modèle reconnaîtra n'importe quelle écriture. Nous l'avons "
    f"donc testé sur des chiffres extérieurs. D'abord, {EXT['n_images']} chiffres écrits avec "
    f"{EXT['n_polices']} polices de Windows, dont 8 imitent l'écriture manuscrite. Ils passent par la même "
    "chaîne de préparation que MNIST : recadrage, réduction dans une boîte de 20 x 20 pixels, centrage selon "
    f"le centre de masse. Le modèle en reconnaît {round(EXT['accuracy_pct'] * EXT['n_images'] / 100)} sur {EXT['n_images']} "
    f"({v(EXT['par_style']['manuscrit'], 0)} % pour les polices manuscrites, "
    f"{v(EXT['par_style']['imprimerie'], 0)} % pour les polices d'imprimerie ; §fig:ext§ en annexe 3). "
    "La seule erreur est un 1 de la police MV Boli, lu comme un 7 : son trait de tête très marqué rappelle la "
    f"barre d'un 7. Ensuite, nous avons tracé les dix chiffres à la souris dans l'application : "
    f"{DESSIN['n_ok']} sur {DESSIN['n']} sont reconnus. Ces bons résultats tiennent en partie à "
    "l'augmentation de données et à une préparation des images identique à celle de MNIST. Il faut toutefois "
    "les nuancer : les chiffres des polices sont propres et bien formés, et dix dessins ne font pas un test "
    "statistique. Les écritures européennes (7 barré, 1 avec un long empattement), absentes de MNIST comme le "
    "rappelait la carte de la partie 5, restent le cas le plus risqué.")
cur.texte(
    "Pour que chacun puisse faire ce test, nous avons mis en ligne une application Streamlit "
    "(mnist-reconnaissance-chiffres.streamlit.app) : on y dessine un chiffre, on tire une image du test ou on "
    "importe une photo. Les premiers essais ont montré que le classifieur répond toujours un chiffre, même "
    "devant un cadre rempli ou une étoile. L'application filtre donc les dessins : elle refuse un tracé qui "
    "couvre plus de 45 % de l'image, puis un second réseau, entraîné à distinguer les chiffres de dessins "
    f"quelconques, écarte le reste ({v(DET['chiffres_test_acceptes_pct'], 1)} % des chiffres du test acceptés, "
    f"{v(DET['non_chiffres_rejetes_pct'], 1)} % des dessins quelconques refusés).")
supprimer(p8)

# ---------------------------------------------------------------------------
# Recommandations et conclusion
# ---------------------------------------------------------------------------
p = trouver("À compléter – recommandations")
cur = Curseur(p)
for t in [
    "**Choisir un réseau convolutif** pour reconnaître des chiffres manuscrits. Il est à la fois le plus précis "
    f"({v(cnn['err'], 2)} % d'erreur) et rapide à l'usage, une fois entraîné. Si l'on ne peut pas entraîner de "
    "réseau, le SVM à noyau gaussien sur 50 composantes principales est la meilleure solution classique.",
    "**Régler les modèles par validation croisée et réserver le jeu de test à l'évaluation finale.** C'est la "
    "seule façon d'annoncer une performance crédible, et c'est le principal défaut de l'article étudié.",
    "**Donner un intervalle de confiance** avec chaque performance. Le bootstrap le fait simplement et permet "
    "de savoir si l'écart entre deux modèles est réel.",
    "**Tester le modèle sur des données proches de l'usage réel** avant de le déployer. Pour aller plus loin, il "
    "faudrait l'entraîner aussi sur des écritures européennes (7 barré, 1 avec empattement), par exemple "
    "avec EMNIST, et sur des chiffres photographiés.",
    "**Surveiller la qualité des étiquettes**, même dans un jeu de données de référence : deux modèles "
    "indépendants suffisent à repérer la plupart des erreurs grossières.",
]:
    cur.texte(t, puce=True)
supprimer(p)

p = trouver("À compléter – conclusion")
cur = Curseur(p)
cur.texte(
    "Ce projet nous a fait parcourir toutes les étapes d'une étude de data mining sur un jeu de données que "
    "tout le monde croit connaître. Les contrôles montrent que MNIST est propre sur le plan technique, mais "
    f"l'analyse des anomalies a tout de même révélé {C6['suspects_n']} étiquettes probablement fausses. "
    "L'exploration a montré que l'information est très redondante, ce qui a guidé la sélection des "
    "prédicteurs par ACP.")
cur.texte(
    f"Parmi les sept modèles comparés, le réseau convolutif est le meilleur : {v(cnn['acc'], 2)} % de bonnes "
    f"réponses sur 10 000 images jamais vues, avec un intervalle de confiance bootstrap de "
    f"[{v(b_cnn[1], 2)} % ; {v(b_cnn[2], 2)} %]. Il divise par {v(2.08 / cnn['err'], 0)} l'erreur du "
    "meilleur modèle de l'article, avec un protocole plus rigoureux. Le test sur nos propres chiffres est "
    "encourageant, puisque presque tous sont reconnus, mais il porte sur des chiffres bien formés : un modèle "
    "ne vaut que pour des données qui ressemblent à celles de son apprentissage. "
    "Il reste à l'entraîner sur des écritures plus variées, voire sur des lettres manuscrites.")
supprimer(p)

# ---------------------------------------------------------------------------
# Références (ordre alphabétique)
# ---------------------------------------------------------------------------
NOUVELLES = {
    "Breiman": "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "Cortes": "Cortes, C. & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.",
    "Efron": "Efron, B. & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",
    "Kingma": "Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. International "
              "Conference on Learning Representations (ICLR).",
    "Paszke": "Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. "
              "Advances in Neural Information Processing Systems (NeurIPS), 32.",
    "Simard": "Simard, P. Y., Steinkraus, D. & Platt, J. C. (2003). Best practices for convolutional neural "
              "networks applied to visual document analysis. International Conference on Document Analysis "
              "and Recognition (ICDAR), 958-963.",
}
h = trouver("Références bibliographiques", "Heading 1")
refs = []
el = h._p.getnext()
from docx.text.paragraph import Paragraph  # noqa: E402
while el is not None:
    par = Paragraph(el, h._parent)
    if el.tag.endswith("}p") and par.style.name.startswith("Heading"):
        break
    if el.tag.endswith("}p") and par.text.strip():
        refs.append(par)
    el = el.getnext()
for cle, texte in NOUVELLES.items():
    suivant = next((r for r in refs if r.text > texte), None)
    if suivant is None:
        cur = Curseur(refs[-1]); nouv = cur.texte(texte)
    else:
        from outils_docx import _nouveau_p, _runs  # noqa: E402
        nouv = _nouveau_p(suivant, avant=True)
        nouv.paragraph_format.space_after = suivant.paragraph_format.space_after
        _runs(nouv, texte)
    pp = nouv._p.get_or_add_pPr()
    modele_ppr = refs[0]._p.pPr
    if modele_ppr is not None:
        import copy  # noqa: E402
        nouv._p.remove(pp)
        nouv._p.insert(0, copy.deepcopy(modele_ppr))
    refs = sorted(refs + [nouv], key=lambda r: r.text)

# ---------------------------------------------------------------------------
# Acronymes
# ---------------------------------------------------------------------------
import copy  # noqa: E402
ta = doc.tables[0]
existants = [r.cells[0].text.strip() for r in ta.rows]
for sigle, sens in [("VC", "Validation croisée"), ("F1", "Moyenne harmonique de la précision et du rappel"),
                    ("ReLU", "Rectified Linear Unit (fonction d'activation)")]:
    if sigle in existants:
        continue
    tr = copy.deepcopy(ta.rows[-1]._tr)
    ta._tbl.append(tr)
    row = ta.rows[-1]
    for c, t in zip(row.cells, [sigle, sens]):
        for r in c.paragraphs[0].runs[1:]:
            r._r.getparent().remove(r._r)
        c.paragraphs[0].runs[0].text = t
# tri alphabétique des lignes (hors en-tête)
lignes_tr = [r._tr for r in ta.rows[1:]]
for tr in lignes_tr:
    ta._tbl.remove(tr)
for tr in sorted(lignes_tr, key=lambda tr: "".join(t.text for t in tr.iter(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")).upper()):
    ta._tbl.append(tr)

# ---------------------------------------------------------------------------
# Annexes
# ---------------------------------------------------------------------------
p = trouver("À compléter – insérer le diagramme de Gantt")
cur = Curseur(p)
cur.image(FIG / "figA1_gantt.png", 16.5)
FIGS["gantt"] = cur.legende("Figure", "Diagramme de Gantt du déroulement de la SAÉ (semaines du 28 septembre au 11 décembre 2026)")
supprimer(p)

p = trouver("L'ensemble du code est fourni dans le notebook")
remplacer_paragraphe(p, (
    "Le code complet est sur GitHub (github.com/coulibaly-abou/sae503-mnist-reconnaissance-chiffres) et l'application est en ligne (mnist-reconnaissance-chiffres.streamlit.app). Les "
    "parties 3 à 5 sont dans notebooks/SAE503_MNIST_exploration.ipynb, la partie 6 dans "
    "notebooks/SAE503_MNIST_anomalies.ipynb, les parties 7 et 8 dans scripts/modelisation.py (avec "
    "scripts/cnn.py pour le réseau convolutif) et le test sur les polices dans "
    "scripts/test_chiffres_externes.py. L'application est dans app/app.py. Nous reproduisons ici trois "
    "extraits courts. Premier extrait : la lecture des fichiers IDX avec contrôle de l'en-tête."))
dernier_code = [q for q in doc.paragraphs if q.text.strip().startswith("return magique, dims")][0]
EXTRAITS = [
    ("Deuxième extrait : le bootstrap de l'accuracy et de l'écart entre les deux modèles (partie 7).", [
        "tirages = rng.integers(0, n, size=(B, n))",
        "boot_cnn = 100 * ok_cnn[tirages].mean(1)",
        "boot_svm = 100 * ok_svm[tirages].mean(1)",
        "boot_diff = boot_cnn - boot_svm",
        "ic = np.percentile(boot_diff, [2.5, 97.5])",
    ]),
    ("Troisième extrait : le grid search du SVM en validation croisée (partie 8).", [
        "grille = {\"svc__C\": [1, 5, 10, 50], \"svc__gamma\": [\"scale\", 0.01, 0.02, 0.04, 0.06, 0.08]}",
        "pipe = make_pipeline(PCA(50), SVC())",
        "gs = GridSearchCV(pipe, grille, cv=StratifiedKFold(3, shuffle=True), scoring=\"accuracy\")",
        "gs.fit(X_sous_echantillon, y_sous_echantillon)",
        "svm_final = make_pipeline(PCA(50), SVC(**params)).fit(X_train, y_train)",
    ]),
]
for intro, lignes_code in EXTRAITS:
    cur = Curseur(dernier_code)
    cur.texte(intro)
    for l in lignes_code:
        nouv = copy.deepcopy(dernier_code._p)
        cur.dernier._p.addnext(nouv)
        par = Paragraph(nouv, dernier_code._parent)
        for r in par.runs[1:]:
            r._r.getparent().remove(r._r)
        par.runs[0].text = l
        cur.dernier = par
    dernier_code = cur.dernier

p = [q for q in doc.paragraphs if "Part d'images signalées" in q.text and q.style.name == "Legende figure"][0]
cur = Curseur(p)
cur.image(FIG / "fig8_2_comparaison_modeles.png", 16.5)
FIGS["comp"] = cur.legende("Figure", "Accuracy en validation croisée à 5 plis (gauche) et temps d'apprentissage (droite) des sept "
                      "modèles")
cur.image(FIG / "fig8_5_erreurs_cnn.png", 16.5)
FIGS["err"] = cur.legende("Figure", "Images du test mal classées par le réseau convolutif avec la plus forte confiance "
                      "(chiffre réel → chiffre prédit, confiance)")
cur.image(FIG / "fig8_6_chiffres_externes.png", 13)
FIGS["ext"] = cur.legende("Figure", "Chiffres écrits avec 12 polices de Windows : chiffre prédit au-dessus de chaque image "
                      "(vert : correct, rouge : erreur)")


# ---------------------------------------------------------------------------
# Numérotation des figures et mise à jour des renvois
# ---------------------------------------------------------------------------
import re  # noqa: E402
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
legendes = [q for q in doc.paragraphs if q.style.name == "Legende figure" and "SEQ Figure" in q._p.xml]
numero = {id(q._p): i + 1 for i, q in enumerate(legendes)}
for q in legendes:
    for f in q._p.iter(W + "fldSimple"):
        if "SEQ Figure" in f.get(W + "instr"):
            for t in f.iter(W + "t"):
                t.text = str(numero[id(q._p)])
ancien_vers_nouveau = {i + 1: numero[id(q._p)] for i, q in enumerate(ANCIENNES)}
cle_vers_num = {k: numero[id(q._p)] for k, q in FIGS.items()}
print("renumérotation des anciennes figures :", ancien_vers_nouveau)
print("nouvelles figures :", cle_vers_num)
legendes_ids = {id(q._p) for q in legendes}
for q in doc.paragraphs:
    if id(q._p) in legendes_ids or q.style.name == "table of figures":
        continue
    for t in q._p.iter(W + "t"):
        if t.text is None:
            continue
        txt = t.text
        if "§fig:" in txt:
            txt = re.sub(r"§fig:(\w+)§", lambda m: f"figure {cle_vers_num[m.group(1)]}", txt)
        elif q._p in ANCIENS_P:
            txt = re.sub(r"([Ff]igures? )(\d+)", lambda m: m.group(1) + str(ancien_vers_nouveau.get(int(m.group(2)), int(m.group(2)))), txt)
        t.text = txt

# ---------------------------------------------------------------------------
# Police Times New Roman partout, puis enregistrement
# ---------------------------------------------------------------------------
times_partout(doc)
sortie = P / "rapport" / "AMAVIGAN_COULIBALY_rapport_SAE503.docx"
doc.save(sortie)
print(sortie)
