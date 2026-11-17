"""Remplit README.md à partir de readme_modele.md et des chiffres clés."""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[2]
RES = P / "resultats"
C = json.loads((RES / "chiffres_cles_p7_p8.json").read_text(encoding="utf-8"))
EXT = json.loads((RES / "chiffres_cles_test_externe.json").read_text(encoding="utf-8"))
DES = json.loads((RES / "chiffres_cles_dessins.json").read_text(encoding="utf-8"))


def v(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


lignes = []
for nom, r in C["test"].items():
    gras = "**" if nom == "Réseau convolutif ajusté" else ""
    lignes.append(f"| {gras}{nom}{gras} | {gras}{v(r['err'])} %{gras} |")
b = C["bootstrap"]["Accuracy Réseau convolutif ajusté"]
texte = (P / "scripts" / "rapport" / "readme_modele.md").read_text(encoding="utf-8")
texte = (texte.replace("{TABLEAU}", "\n".join(lignes))
         .replace("{IC}", f"[{v(100 - b[2])} % ; {v(100 - b[1])} %]")
         .replace("{EXT}", v(EXT["accuracy_pct"], 0))
         .replace("{DESSIN}", f"{DES['n_ok']} sur {DES['n']}"))
(P / "README.md").write_text(texte, encoding="utf-8")
print("README.md écrit")
