"""
paper_style.py
===============
Stile tipografico comune alle figure del paper (PMLR / L4DC, corpo Times).
Niente LaTeX installato sulla macchina (usetex impossibile): testo in
Nimbus Roman (clone metrico di Times, lo stesso usato da pdflatex con
\\usepackage{times}/mathptmx) e matematica in STIX (stile Times), cosi'
lettere, cifre e simboli nelle figure coincidono con quelli del testo.
Font incorporati come TrueType/CID (pdf.fonttype 42), richiesto dagli editori.
"""
import matplotlib.pyplot as plt

TEXT_COLOR = "black"

RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply():
    plt.rcParams.update(RC)
