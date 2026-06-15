"""
Tema visual central de la aplicacion: paleta, tipografia, estilos ttk,
estilo de matplotlib y pequenos helpers (header, tarjetas, separadores, chips).

Objetivo: aspecto de aplicacion profesional de analisis de datos.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import matplotlib

# --------------------------------------------------------------------------- #
# Paleta
# --------------------------------------------------------------------------- #
PRIMARY = "#3B5BDB"
PRIMARY_DARK = "#2C3E9E"
PRIMARY_LIGHT = "#E8EDFD"
BG = "#F3F5FA"          # fondo de la ventana
SURFACE = "#FFFFFF"     # tarjetas
SURFACE_ALT = "#F6F8FC"
BORDER = "#E3E8F1"
TEXT = "#1B2330"
TEXT_MUTED = "#6B7686"
SUCCESS = "#179A5B"
SUCCESS_BG = "#E7F6EE"
DANGER = "#D64545"
DANGER_BG = "#FBEBEB"
WARN = "#C77700"

# Colores de condicion (consistentes entre la UI y los graficos)
NS = "#C2298A"          # Descansado (rosa/magenta, como el paper)
SD = "#2F9E44"          # Privacion de sueno (verde, como el paper)

# --------------------------------------------------------------------------- #
# Tipografia
# --------------------------------------------------------------------------- #
FAMILY = "Segoe UI"
MONO = "Consolas"
F_BRAND = (FAMILY, 15, "bold")
F_SUB = (FAMILY, 9)
F_H1 = (FAMILY, 13, "bold")
F_H2 = (FAMILY, 11, "bold")
F_BODY = (FAMILY, 10)
F_SMALL = (FAMILY, 9)
F_BTN = (FAMILY, 10, "bold")


def apply(root: tk.Tk):
    """Aplica el tema a la ventana raiz y a matplotlib."""
    root.configure(bg=BG)
    try:
        root.option_add("*Font", F_BODY)
        root.option_add("*TCombobox*Listbox.font", F_BODY)
    except Exception:
        pass

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except Exception:
        pass

    try:
        _configure_styles(st)
    except Exception:
        pass

    _apply_mpl()


def _configure_styles(st):
    st.configure(".", background=SURFACE, foreground=TEXT, font=F_BODY,
                 bordercolor=BORDER, focuscolor=PRIMARY)
    st.configure("TFrame", background=SURFACE)
    st.configure("App.TFrame", background=BG)
    st.configure("Card.TFrame", background=SURFACE)
    st.configure("Toolbar.TFrame", background=SURFACE)

    st.configure("TLabel", background=SURFACE, foreground=TEXT, font=F_BODY)
    st.configure("App.TLabel", background=BG, foreground=TEXT)
    st.configure("Muted.TLabel", background=SURFACE, foreground=TEXT_MUTED,
                 font=F_SMALL)
    st.configure("H1.TLabel", background=SURFACE, foreground=TEXT, font=F_H1)
    st.configure("H2.TLabel", background=SURFACE, foreground=TEXT, font=F_H2)
    st.configure("Accent.TLabel", background=SURFACE, foreground=PRIMARY,
                 font=F_H2)

    # Botones
    st.configure("TButton", background=SURFACE_ALT, foreground=TEXT,
                 font=F_BTN, relief="flat", borderwidth=0, padding=(14, 8))
    st.map("TButton",
           background=[("active", BORDER), ("pressed", BORDER)])
    st.configure("Primary.TButton", background=PRIMARY, foreground="#FFFFFF",
                 font=F_BTN, relief="flat", borderwidth=0, padding=(16, 8))
    st.map("Primary.TButton",
           background=[("active", PRIMARY_DARK), ("pressed", PRIMARY_DARK)],
           foreground=[("disabled", "#C9D2E3")])
    st.configure("Ghost.TButton", background=SURFACE, foreground=PRIMARY,
                 font=F_BTN, relief="flat", borderwidth=0, padding=(12, 7))
    st.map("Ghost.TButton", background=[("active", PRIMARY_LIGHT)])
    # Boton claro sobre el header oscuro
    st.configure("OnDark.TButton", background="#5872E0", foreground="#FFFFFF",
                 font=F_BTN, relief="flat", borderwidth=0, padding=(12, 7))
    st.map("OnDark.TButton", background=[("active", "#6E86EA")])

    # Combobox
    st.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE,
                 foreground=TEXT, arrowcolor=PRIMARY, bordercolor=BORDER,
                 lightcolor=BORDER, darkcolor=BORDER, padding=5)
    st.map("TCombobox",
           fieldbackground=[("readonly", SURFACE)],
           foreground=[("readonly", TEXT)],
           bordercolor=[("focus", PRIMARY)])

    # Check / Radio
    for s in ("TCheckbutton", "TRadiobutton"):
        st.configure(s, background=SURFACE, foreground=TEXT, font=F_BODY,
                     focuscolor=SURFACE)
        st.map(s, background=[("active", SURFACE)],
               indicatorcolor=[("selected", PRIMARY)])

    # Notebook (pestanas)
    st.configure("TNotebook", background=BG, borderwidth=0,
                 tabmargins=(10, 8, 10, 0))
    st.configure("TNotebook.Tab", background=BG, foreground=TEXT_MUTED,
                 font=F_BTN, padding=(16, 9), borderwidth=0)
    st.map("TNotebook.Tab",
           background=[("selected", SURFACE)],
           foreground=[("selected", PRIMARY_DARK)])

    # Progressbar
    st.configure("TProgressbar", background=PRIMARY, troughcolor=BORDER,
                 borderwidth=0, thickness=8)

    # Scrollbar
    st.configure("TScrollbar", background=SURFACE_ALT, troughcolor=BG,
                 bordercolor=BG, arrowcolor=TEXT_MUTED)

    # Panedwindow
    st.configure("TPanedwindow", background=BG)
    st.configure("Card.TLabelframe", background=SURFACE, borderwidth=0)
    st.configure("Card.TLabelframe.Label", background=SURFACE,
                 foreground=TEXT_MUTED, font=F_H2)


def _apply_mpl():
    """rcParams de matplotlib para que los graficos combinen con la UI."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": "#EDF1F8",
        "grid.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": TEXT,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 100,
    })


# --------------------------------------------------------------------------- #
# Helpers de layout
# --------------------------------------------------------------------------- #
def hsep(parent, color=BORDER):
    """Separador horizontal de 1px."""
    return tk.Frame(parent, height=1, bg=color)


def card(parent, **pack):
    """Tarjeta blanca con borde sutil. Devuelve el frame interior."""
    outer = tk.Frame(parent, bg=BORDER, bd=0,
                     highlightthickness=0)
    inner = tk.Frame(outer, bg=SURFACE)
    inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
    return outer, inner


def section_title(parent, text, subtitle=None):
    """Titulo de seccion con subtitulo opcional, sobre fondo de tarjeta."""
    f = ttk.Frame(parent, style="Card.TFrame")
    ttk.Label(f, text=text, style="H1.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(f, text=subtitle, style="Muted.TLabel").pack(anchor="w",
                                                               pady=(1, 0))
    return f


def chip(parent, text, color):
    """Pequena etiqueta con punto de color (para leyendas NS/SD)."""
    f = ttk.Frame(parent, style="Card.TFrame")
    dot = tk.Canvas(f, width=12, height=12, bg=SURFACE, highlightthickness=0)
    dot.create_oval(2, 2, 11, 11, fill=color, outline=color)
    dot.pack(side=tk.LEFT)
    ttk.Label(f, text=" " + text, style="Muted.TLabel").pack(side=tk.LEFT)
    return f


class ScrollFrame(ttk.Frame):
    """Contenedor vertical con scroll. Agregar hijos a self.inner."""
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=SURFACE, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame")
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(
            self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.bind("<Enter>", lambda e: self.canvas.bind_all(
            "<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
