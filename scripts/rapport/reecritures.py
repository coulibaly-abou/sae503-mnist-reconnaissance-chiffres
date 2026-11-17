"""Réécriture plus naturelle des paragraphes des parties 1 à 6 (version de travail du 16/10).

Clé : début du paragraphe d'origine. Valeur : nouveau texte (**gras**, *italique*).
Les chiffres et les renvois sont inchangés ; les numéros de figure sont ceux de la version
d'origine (la renumérotation est faite ensuite par construire_rapport.py).
"""

REECRITURES = {
    "La reconnaissance automatique de caractères":
        "Lire automatiquement des chiffres écrits à la main est l'un des plus vieux problèmes de l'apprentissage "
        "automatique. Dès les années 1990, le tri du courrier, la lecture des chèques ou la saisie de formulaires en "
        "ont fait un vrai enjeu industriel (LeCun et al., 1998). Le jeu de données MNIST, tiré des bases du NIST, est "
        "devenu depuis la référence pour comparer les méthodes de classification d'images (Deng, 2012). Les résultats "
        "publiés vont d'environ 12 % d'erreur pour un classifieur linéaire à 0,2 % pour les meilleurs réseaux "
        "convolutifs (Cireşan et al., 2012 ; Wan et al., 2013).",

    "Problématique.":
        "**Problématique.** Les auteurs veulent savoir laquelle de trois familles d'algorithmes « classiques » se "
        "trompe le moins sur MNIST, et comment les hyperparamètres font varier ce taux d'erreur.",

    "Données. Les 60 000":
        "**Données.** Ils utilisent les 60 000 images d'apprentissage et les 10 000 images de test. Chaque image de "
        "28 × 28 pixels est mise à plat en un vecteur de 784 valeurs, ramenées entre 0 et 1 pour faciliter "
        "l'apprentissage.",

    "Traitement. Chaque famille":
        "**Traitement.** Chaque famille est essayée avec plusieurs réglages. Pour le SVM, quatre noyaux (linéaire, "
        "polynomial, RBF, sigmoïde) et plusieurs valeurs de gamma. Pour le KNN, 3, 5 ou 7 voisins, avec un vote "
        "simple ou pondéré par la distance. Pour le réseau de neurones, une couche cachée de 60 ou 150 neurones et "
        "trois fonctions d'activation. Les auteurs comparent ensuite le meilleur réglage de chaque famille (figure 2).",

    "Résultats et conclusion.":
        "**Résultats et conclusion.** Le SVM à noyau RBF obtient la plus faible erreur, 2,08 %. Il devance le KNN à "
        "3 voisins pondéré par la distance (2,82 %) et le réseau de neurones à 150 neurones (3,80 %). Les auteurs en "
        "concluent que le SVM RBF est le meilleur choix pour ce problème.",

    "MNIST a été introduit par LeCun":
        "MNIST a été présenté par LeCun et al. (1998), dans l'article qui décrit le réseau convolutif LeNet-5. Il a "
        "été construit à partir des bases manuscrites du NIST (Grother, 1995) et reste depuis la série la plus "
        "utilisée pour comparer des classifieurs d'images (Deng, 2012). Le tableau 2 regroupe des résultats publiés "
        "sur le même jeu de test. On y voit que l'article étudié se situe au niveau des méthodes classiques de la fin "
        "des années 1990, loin des meilleurs résultats actuels.",

    "D'autres travaux ont traité":
        "D'autres équipes se sont intéressées à la qualité de ces données. Yadav et Bottou (2019) ont retrouvé "
        "l'origine de chaque image et reconstitué 50 000 images de test perdues (base QMNIST). Northcutt et al. "
        "(2021) estiment qu'une quinzaine d'images du test MNIST portent une mauvaise étiquette. Enfin, plusieurs "
        "jeux de données ont été créés sur le même modèle pour aller plus loin : EMNIST pour les lettres (Cohen et "
        "al., 2017) et Fashion-MNIST, où des vêtements, plus difficiles à classer, remplacent les chiffres (Xiao et "
        "al., 2017).",

    "MNIST (Modified NIST) a été construit":
        "MNIST (Modified NIST) a été construit par Yann LeCun, Corinna Cortes et Christopher Burges, chez AT&T Bell "
        "Labs, et publié en 1998 (LeCun et al., 1998). Il réunit deux bases du NIST, l'institut américain des normes "
        "installé à Gaithersburg dans le Maryland, regroupées plus tard dans la Special Database 19 (Grother, 1995) :",

    "Les formulaires ont été collectés":
        "Les formulaires ont été remplis puis numérisés au début des années 1990. Comme les employés écrivent plus "
        "proprement que les lycéens, chaque partie de MNIST mélange les deux sources à parts égales : 30 000 images "
        "de chaque pour l'apprentissage, 5 000 de chaque pour le test. Les quelque 250 personnes qui ont écrit les "
        "chiffres d'apprentissage ne sont pas celles du test. Le test mesure donc bien la capacité d'un modèle à lire "
        "des écritures qu'il n'a jamais vues. Nous avons travaillé sur les quatre fichiers officiels, au format IDX "
        "compressé.",

    "Les images binaires du NIST":
        "Au départ, les chiffres du NIST sont en noir et blanc pur. Pour MNIST, chacun a été réduit pour tenir dans "
        "un carré de 20 × 20 pixels sans être déformé. Cette réduction adoucit les bords du trait, d'où les niveaux "
        "de gris. Le chiffre a ensuite été placé dans une grille de 28 × 28 pixels, centré sur son centre de masse. "
        "Une observation est donc l'image d'un seul chiffre, décrite par 784 pixels et accompagnée de son étiquette "
        "(tableau 3).",

    "Au total, 70 000 images":
        "Au total, nous disposons de 70 000 images, dont 14,3 % servent au test. L'échantillon tiré au hasard en "
        "annexe 3 (figure 12) montre à quel point l'écriture varie pour un même chiffre : inclinaison, épaisseur du "
        "trait, 4 ouvert ou fermé, 7 avec ou sans barre.",

    "Le problème est donc une classification":
        "Il s'agit donc d'un problème de classification supervisée : prédire une variable qualitative à 10 modalités "
        "à partir de 784 variables quantitatives, très corrélées d'un pixel à son voisin. Les données n'ont ni "
        "dimension temporelle ni dimension géographique.",

    "La qualité a été évaluée":
        "Nous avons vérifié la qualité des données selon les critères habituels : intégrité, complétude, validité, "
        "unicité, représentativité et pertinence. Le tableau 5 résume ces contrôles.",

    "Intégrité, complétude et validité.":
        "**Intégrité, complétude et validité.** Les quatre fichiers respectent le format IDX. Il ne manque aucune "
        "valeur, et aucune ne sort des bornes prévues. Nous avons noté l'empreinte MD5 de chaque fichier pour pouvoir "
        "refaire l'analyse à l'identique.",

    "Unicité. Aucune image":
        "**Unicité.** Aucune image n'apparaît deux fois, ni dans l'apprentissage, ni dans le test, ni d'un ensemble "
        "à l'autre. Un modèle ne pourra donc pas « tricher » en retrouvant au test une image déjà vue.",

    "Équilibre des classes.":
        "**Équilibre des classes.** Chaque chiffre représente entre 9,0 % (le 5) et 11,2 % (le 1) des images "
        "d'apprentissage (figure 3), soit un rapport de 1,24 entre la classe la plus fréquente et la plus rare. Le "
        "test du χ² ne montre aucune différence de répartition entre apprentissage et test (χ² = 2,9, 9 ddl, "
        "p = 0,97). L'accuracy est donc une mesure adaptée, et nous la complétons par le F1-score macro.",

    "Pertinence des variables":
        "**Pertinence des variables et conformité.** 67 pixels, dans les coins et sur les bords, valent 0 sur les "
        "60 000 images (figure 4, gauche). Ils n'apportent aucune information et nous les avons retirés. 189 autres "
        "pixels ne sont encrés que dans moins de 0,1 % des images. Le protocole de construction est bien respecté : "
        "tous les chiffres tiennent dans une boîte de 20 × 20, 91,1 % font exactement 20 pixels de haut (figure 4, "
        "droite) et le centre de masse moyen tombe en (14,0 ; 14,0), à 0,5 pixel près au plus.",

    "Bilan. Le jeu de données":
        "**Bilan.** Sur le plan technique, le jeu de données est propre et n'a pas besoin d'être nettoyé. Ces "
        "contrôles ne voient pas, en revanche, les erreurs d'étiquetage ni les chiffres bizarres ou illisibles "
        "(Northcutt et al., 2021). C'est le rôle de la détection d'anomalies (partie 6).",

    "Les 784 pixels n'étant pas":
        "Un pixel isolé ne dit pas grand-chose. Nous avons donc résumé chaque image par quelques descripteurs : la "
        "quantité d'encre (somme des niveaux de gris divisée par 255, soit un nombre de pixels pleins), la "
        "couverture (part des pixels non nuls) et la largeur du chiffre. En moyenne, 80,9 % des pixels sont vides et "
        "une image est encrée sur 19,1 % de sa surface. Parmi les pixels encrés, les niveaux de gris se concentrent "
        "vers 252-255, au cœur du trait, avec un plateau de valeurs intermédiaires sur ses bords (figure 5, gauche).",

    "Le tableau 6 révèle":
        "Les chiffres diffèrent beaucoup entre eux (tableau 6). Le 1 est le plus économe en encre, avec 59,6 pixels "
        "pleins en moyenne, et le plus étroit (9,3 pixels de large). Le 0 en utilise plus du double (135,9) et occupe "
        "presque toute la largeur de la boîte (18,0 pixels). Le test de Kruskal-Wallis confirme que la quantité "
        "d'encre dépend nettement du chiffre (H = 20 964, p < 0,001). Ce descripteur aide donc à reconnaître un "
        "chiffre, mais il ne suffit pas à séparer ceux qui ont une densité proche, comme le 4, le 7 et le 9.",

    "Les boîtes à moustaches":
        "Les boîtes à moustaches (figure 5, droite) sont étirées vers le haut pour tous les chiffres, avec beaucoup "
        "de valeurs extrêmes : ce sont des chiffres écrits avec un trait très épais ou repassés plusieurs fois. Nous "
        "les retrouverons parmi les anomalies.",

    "Les images moyennes (figure 6)":
        "Les images moyennes de chaque chiffre (figure 6) restent lisibles, signe que le centrage est bon. Les images "
        "d'écart-type montrent où les écritures varient le plus : l'inclinaison du 1 dessine une sorte de sablier, et "
        "c'est sur la barre du 7 ou la boucle du 9 que les scripteurs diffèrent le plus.",

    "La corrélation entre images moyennes":
        "La corrélation entre ces images moyennes (annexe 3, figure 13) mesure à quel point deux chiffres se "
        "ressemblent. Les paires les plus proches sont le 4 et le 9 (0,90), le 5 et le 8 (0,87), puis le 7 et le 9 "
        "(0,85). Les plus éloignés sont le 0 et le 1 (0,29). On peut s'attendre à ce que les modèles confondent "
        "surtout ces paires.",

    "L'ACP (figure 7) montre":
        "L'ACP (figure 7) confirme que l'information est très redondante. Il suffit de 44 composantes pour garder "
        "80 % de la variance, de 87 pour 90 % et de 154 pour 95 %, alors qu'il y a 717 pixels utiles. Le premier plan "
        "n'explique que 16,8 % de la variance et sépare surtout le 0 du 1. La projection t-SNE (van der Maaten et "
        "Hinton, 2008), non linéaire, fait au contraire apparaître dix groupes bien distincts, sauf le 4, le 7 et le "
        "9 qui se chevauchent, et le 3, le 5 et le 8 qui restent voisins. Nous en retenons deux idées pour la "
        "partie 8 : une ACP peut réduire fortement le nombre de variables, et il faudra des modèles non linéaires.",

    "Pour représenter la forme":
        "Nous voulions représenter la forme des chiffres autrement que par des pixels. Pour chaque image, nous avons "
        "calculé comment l'encre se répartit autour du centre de masse, en 36 secteurs de 10°, puis fait la moyenne "
        "par chiffre. Le graphique radial obtenu donne une sorte d'empreinte de chaque chiffre (figure 8).",

    "Le 0 est presque isotrope":
        "Le 0 a une empreinte presque ronde : chaque secteur reçoit entre 2 et 4 % de l'encre, pour 2,8 % si elle "
        "était parfaitement répartie. Le 1, lui, concentre son encre dans deux lobes opposés, vers 70-80° et "
        "250-260°, jusqu'à 8,7 % par secteur : c'est l'inclinaison naturelle de l'écriture. On reconnaît aussi la "
        "barre et la branche du 7, les trois branches du 4 et la base du 2, en bas à gauche. Les empreintes du 4 et "
        "du 9, ou du 3 et du 5, restent proches, comme dans l'analyse précédente.",

    "MNIST ne contenant aucune":
        "MNIST ne contient aucune donnée géographique. Nous avons donc utilisé des données externes : les fonds de "
        "carte Natural Earth, libres de droits, et la position des institutions qui ont produit puis utilisé ces "
        "données. La carte (figure 9) suit le parcours des chiffres, de leur collecte dans le Maryland (NIST et "
        "Census Bureau) à la construction de MNIST dans le New Jersey (Bell Labs), jusqu'aux auteurs de l'article, "
        "en Californie et en Chine. Elle rappelle surtout que toutes ces écritures sont américaines. Un modèle appris "
        "sur MNIST risque donc d'avoir plus de mal avec d'autres habitudes, comme le 1 à long empattement ou le 7 "
        "barré, courants en Europe.",

    "Sur MNIST, une anomalie":
        "Dans MNIST, une anomalie peut être de deux sortes : une image inhabituelle pour son chiffre (style rare, "
        "trait très épais, chiffre coupé) ou une étiquette fausse, par exemple un 4 étiqueté 9. Nous les avons "
        "cherchées avec plusieurs méthodes, uniquement dans le jeu d'apprentissage. Le test n'est jamais modifié, "
        "pour que nos résultats restent comparables à ceux de la littérature (partie 2).",

    "L'approche univariée":
        "La méthode univariée porte sur la quantité d'encre. Pour chaque chiffre, nous calculons un z-score robuste, "
        "à partir de la médiane et de l'écart absolu médian, et nous signalons les images où |z| dépasse 3,5 "
        "(Iglewicz et Hoaglin, 1993). Les méthodes multivariées travaillent chiffre par chiffre sur les 50 premières "
        "composantes principales, qui gardent 82,5 % de la variance. Elles reposent sur trois idées différentes. "
        "L'erreur de reconstruction d'une ACP à 20 composantes, propre à chaque chiffre, mesure l'écart à une forme "
        "typique. L'Isolation Forest isole les points par des coupures au hasard (Liu et al., 2008). Le Local "
        "Outlier Factor compare la densité autour d'une image à celle de ses 20 voisins (Breunig et al., 2000). "
        "Chaque méthode signale 1 % des images de chaque chiffre, et nous appelons anomalie forte une image signalée "
        "par au moins deux méthodes sur trois. Pour les étiquettes, nous avons utilisé deux modèles de nature "
        "différente, un KNN à 10 voisins et un perceptron multicouche, en validation croisée à 5 plis : aucune image "
        "n'est jugée par un modèle qui l'a apprise.",

    "Le critère univarié":
        "Le critère univarié ne signale que des images trop encrées (0,32 %), surtout des 9 et des 1. Aucune image "
        "n'est anormalement pâle. Les trois méthodes multivariées s'accordent peu entre elles (indice de Jaccard de "
        "0,06 à 0,22) : chacune voit une forme différente d'atypie, d'où l'intérêt de les croiser. Nous obtenons 320 "
        "anomalies fortes (0,53 % des images), dont 44 repérées par les trois méthodes. Elles touchent tous les "
        "chiffres de façon assez égale, de 0,37 % pour le 5 à 0,68 % pour le 4, et sont nettement plus encrées que "
        "les autres images (141,7 pixels pleins en médiane, contre 99,7).",

    "La figure 10 montre que":
        "La figure 10 montre qu'il s'agit surtout de styles d'écriture rares mais lisibles : des 4 ouverts en forme "
        "de U, des 7 barrés à l'européenne, des 1 avec empattement et base, des 2 bouclés, des 6 très ronds. "
        "Quelques images sont vraiment abîmées (chiffre en forme de tache, trait coupé), mais elles sont rares.",

    "En validation croisée, le KNN et le perceptron":
        "En validation croisée, le KNN reconnaît 97,4 % des images et le perceptron 97,9 %. Le KNN conteste 191 "
        "étiquettes et le perceptron 302, mais ils ne tombent d'accord, avec confiance, sur une autre classe que pour "
        "78 images (0,13 %). En les regardant une par une (figure 11), nous voyons que la plupart représentent bien "
        "le chiffre proposé par les modèles : un 4 étiqueté 9, un 5 étiqueté 3, un 7 étiqueté 9. Les autres sont "
        "ambiguës, même pour nous. Seules 14 de ces images font partie des anomalies fortes : une erreur "
        "d'étiquette touche le plus souvent un chiffre bien écrit, que les méthodes non supervisées ne peuvent pas "
        "repérer. Cet ordre de grandeur rejoint celui de Northcutt et al. (2021) sur le jeu de test, environ 0,15 %.",

    "Images atypiques : conservées.":
        "**Images atypiques : conservées.** Elles reflètent la vraie diversité des écritures, que le modèle "
        "rencontrera aussi au test. Les retirer le rendrait moins robuste.",

    "Étiquettes suspectes : retirées":
        "**Étiquettes suspectes : retirées de l'apprentissage.** Elles apprendraient au modèle des associations "
        "fausses. Après ce retrait, il reste 59 922 images d'apprentissage, et l'équilibre des classes ne change pas "
        "(de 9,0 à 11,2 %).",
}
