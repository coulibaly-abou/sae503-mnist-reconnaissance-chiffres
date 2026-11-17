"""Outils d'insertion dans le rapport Word existant (python-docx + XML)."""
import copy
import re

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph

POLICE = "Times New Roman"


def _nouveau_p(ancre, avant=False):
    p = OxmlElement("w:p")
    if avant:
        ancre._p.addprevious(p)
    else:
        ancre._p.addnext(p)
    return Paragraph(p, ancre._parent)


def _runs(par, texte, taille=None, couleur=None):
    """Écrit un texte avec **gras** et *italique* simples."""
    for morceau in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", texte):
        if not morceau:
            continue
        if morceau.startswith("**"):
            r = par.add_run(morceau[2:-2]); r.bold = True
        elif morceau.startswith("*"):
            r = par.add_run(morceau[1:-1]); r.italic = True
        else:
            r = par.add_run(morceau)
        if taille:
            r.font.size = Pt(taille)
        if couleur:
            r.font.color.rgb = RGBColor.from_string(couleur)
    return par


class Curseur:
    """Insère des blocs les uns après les autres à partir d'un paragraphe d'ancrage."""

    def __init__(self, ancre):
        self.dernier = ancre

    def _suivant(self):
        p = _nouveau_p(self.dernier)
        self.dernier = p
        return p

    def titre(self, texte, niveau=2):
        p = self._suivant()
        p.style = f"Heading {niveau}"
        p.add_run(texte)
        return p

    def texte(self, texte, puce=False):
        p = self._suivant()
        if puce:
            p.style = "List Paragraph"
            pf = p.paragraph_format
            pf.left_indent = Cm(0.6); pf.first_line_indent = Cm(-0.4)
            texte = "• " + texte
        pf = p.paragraph_format
        pf.space_after = Pt(6 if not puce else 3)
        pf.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return _runs(p, texte)

    def image(self, chemin, largeur_cm=16.5):
        p = self._suivant()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(chemin), width=Cm(largeur_cm))
        return p

    def legende(self, type_, texte):
        """type_ = 'Figure' ou 'Tableau' ; numérotation automatique par champ SEQ."""
        p = self._suivant()
        p.style = "Legende figure" if type_ == "Figure" else "Legende tableau"
        p.add_run(f"{type_} ")
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), f" SEQ {type_} \\* ARABIC ")
        r = OxmlElement("w:r"); t = OxmlElement("w:t"); t.text = "0"; r.append(t); fld.append(r)
        p._p.append(fld)
        p.add_run(f" – {texte}")
        if type_ == "Tableau":
            p.paragraph_format.keep_with_next = True
        return p

    def tableau(self, entetes, lignes, largeurs_cm=None, modele=None):
        """Tableau dont la mise en forme copie celle d'un tableau existant (modele)."""
        doc_body = self.dernier._parent
        if modele is not None:
            tbl = copy.deepcopy(modele._tbl)
            for tr in tbl.findall(qn("w:tr"))[2:]:
                tbl.remove(tr)
        self.dernier._p.addnext(tbl)
        from docx.table import Table
        t = Table(tbl, doc_body)
        # ligne d'en-tête et ligne modèle
        ent, mod = t.rows[0], t.rows[1]
        ncol = len(entetes)
        _ajuster_colonnes(t, ncol)
        ent, mod = t.rows[0], t.rows[1]
        for c, txt in zip(ent.cells, entetes):
            _remplir(c, txt, gras=True)
        gabarit = copy.deepcopy(mod._tr)
        t._tbl.remove(mod._tr)
        for ligne in lignes:
            tr = copy.deepcopy(gabarit)
            t._tbl.append(tr)
            row = t.rows[-1]
            for c, txt in zip(row.cells, ligne):
                _remplir(c, str(txt))
        if largeurs_cm:
            for row in t.rows:
                for c, w in zip(row.cells, largeurs_cm):
                    c.width = Cm(w)
        # paragraphe vide après le tableau pour poursuivre
        p = OxmlElement("w:p")
        tbl.addnext(p)
        self.dernier = Paragraph(p, doc_body)
        self.dernier.paragraph_format.space_after = Pt(2)
        return t


def _ajuster_colonnes(t, ncol):
    grid = t._tbl.tblGrid
    cols = grid.findall(qn("w:gridCol"))
    total = sum(int(c.get(qn("w:w"))) for c in cols)
    for c in cols:
        grid.remove(c)
    for _ in range(ncol):
        gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(total // ncol)); grid.append(gc)
    for tr in t._tbl.findall(qn("w:tr")):
        tcs = tr.findall(qn("w:tc"))
        while len(tcs) > ncol:
            tr.remove(tcs[-1]); tcs = tr.findall(qn("w:tc"))
        while len(tcs) < ncol:
            tr.append(copy.deepcopy(tcs[-1])); tcs = tr.findall(qn("w:tc"))
        for tc in tcs:
            w = tc.find(qn("w:tcPr") + "/" + qn("w:tcW"))
            if w is not None:
                w.set(qn("w:w"), str(total // ncol)); w.set(qn("w:type"), "dxa")


def _remplir(cellule, texte, gras=False):
    p = cellule.paragraphs[0]
    for extra in cellule.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    rpr = None
    if p.runs:
        rpr = copy.deepcopy(p.runs[0]._r.rPr) if p.runs[0]._r.rPr is not None else None
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r = p.add_run(texte)
    if rpr is not None:
        r._r.insert(0, rpr)
    if gras:
        r.bold = True


def remplacer_paragraphe(par, texte):
    """Remplace le texte d'un paragraphe en gardant sa mise en forme de paragraphe."""
    for r in list(par.runs):
        r._r.getparent().remove(r._r)
    for h in par._p.findall(qn("w:hyperlink")):
        par._p.remove(h)
    _runs(par, texte)
    return par


def supprimer(par):
    par._p.getparent().remove(par._p)


def times_partout(doc, code_police="Courier New"):
    """Times New Roman dans les styles, les valeurs par défaut et chaque run (sauf code)."""
    styles = doc.styles.element
    for rfonts in styles.iter(qn("w:rFonts")):
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(a), POLICE)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rfonts.get(qn(a)) is not None:
                del rfonts.attrib[qn(a)]
    for part in [doc.part] + [s.header.part for s in doc.sections] + [s.footer.part for s in doc.sections]:
        for rfonts in part.element.iter(qn("w:rFonts")):
            code = rfonts.get(qn("w:ascii")) in ("Consolas", "Courier New", "Cascadia Mono")
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(a), code_police if code else POLICE)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if rfonts.get(qn(a)) is not None:
                    del rfonts.attrib[qn(a)]
