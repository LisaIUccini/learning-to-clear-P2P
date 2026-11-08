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


# --- Layout comune delle figure fig1 noreg, fig2b, fig4 (6 e 3 densita') ------
# Dimensione della figura FISSA (niente bbox_inches="tight", che ritaglierebbe
# ogni figura in modo diverso): larghezza 6 in, pannelli alti PANEL_H_IN in
# tutte le figure, margini e distanze in pollici assoluti (legenda->pannelli e
# scritte->pannelli identiche). L'altezza totale e' la somma: fig1 e fig4 sono
# identiche (2.62 / 1.42 in), fig2b e' piu' alta solo del margine in basso che
# serve per i numeri dei tick x + l'xlabel. Valori ridotti al minimo senza
# sovrapposizioni.
FIG_W_IN = 6.00
PANEL_H_IN = 1.00       # altezza di ogni pannello (asse y), uguale in tutte le figure
LEGEND_TOP_IN = 0.25    # fascia in alto per la legenda (una riga)
DELTA_LABEL_PT = 3      # etichetta "delta = x" sotto l'asse: offset (pt), va="top"
ROW_GAP_IN = 0.20       # spazio fra le righe quando sotto l'asse c'e' solo l'etichetta delta
BOTTOM_IN = 0.17        # margine in basso con la sola etichetta delta
BOTTOM_TICKS_IN = 0.36  # ... con numeri dei tick x + xlabel condiviso
LEFT = {2: 0.10, 1: 0.115}  # margine sinistro (frazione), uguale per tutte; 1 riga: ylabel su 2 righe
RIGHT = 0.975


def grid_2x3(width_in=FIG_W_IN, *, row_gap_in=ROW_GAP_IN, bottom_in=BOTTOM_IN, top_in=LEGEND_TOP_IN,
             wspace=0.12, sharey=False, sharex=False, nrows=2):
    """Figura nrows x 3 (2x3, o 1x3 per le varianti _3dens), pannelli alti PANEL_H_IN."""
    width_in = FIG_W_IN
    height = top_in + nrows * PANEL_H_IN + (nrows - 1) * row_gap_in + bottom_in
    fig, axes = plt.subplots(nrows, 3, figsize=(width_in, height), sharey=sharey, sharex=sharex,
                             gridspec_kw={"hspace": row_gap_in / PANEL_H_IN, "wspace": wspace})
    fig.subplots_adjust(left=LEFT[nrows], right=RIGHT, bottom=bottom_in / height, top=1 - top_in / height)
    return fig, axes.ravel()


def supylabel(fig, text):
    """Etichetta y condivisa, attaccata al bordo sinistro della figura."""
    fig.supylabel(text, fontsize=9, linespacing=1.0, x=0.006, ha="left")


def save_fixed(fig, out_pdf, out_png, dpi=300):
    """Salva SENZA ritaglio: la pagina e' esattamente figsize."""
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=dpi)


def delta_label_below(ax, density, fontsize=8.5):
    ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -DELTA_LABEL_PT), textcoords="offset points",
                ha="center", va="top", fontsize=fontsize)
