"""
Panel "Todos los sujetos": analisis grupal (promedio de TODOS los sujetos)
comparado lado a lado con las figuras del paper.

Si existe 'group_average.npz' junto al codigo, se carga al instante. El boton
"Recalcular" fuerza el calculo en vivo (lento).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.image as mpimg

from ..widgets import MplCanvas
from .. import theme as TH
from core import bands as B
from core import topo as T

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
FIG4 = os.path.join(ASSET_DIR, "fig4_paper_psd_cz.png")
FIG5 = os.path.join(ASSET_DIR, "fig5_paper_topomaps.png")
GROUP_NPZ = os.path.join(os.path.dirname(__file__), "..", "..",
                         "group_average.npz")
COLORS = {"rested": TH.NS, "deprived": TH.SD}


class AllSubjectsPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._grand = None
        self._running = False
        self._build()

    def _build(self):
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 8))
        col = ttk.Frame(head)
        col.pack(side=tk.LEFT)
        ttk.Label(col, text="Promedio de todos los sujetos",
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(col, text="Nuestros datos (izquierda) comparados con las "
                  "figuras del paper (derecha).", style="Muted.TLabel").pack(
            anchor="w", pady=(2, 0))
        ttk.Button(head, text="↻  Recalcular", style="Ghost.TButton",
                   command=self._recompute).pack(side=tk.RIGHT)
        TH.hsep(self).pack(fill=tk.X, padx=18)

        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=18, pady=12)
        top = ttk.Labelframe(paned, text="  PSD del canal Cz  ",
                             style="Card.TLabelframe")
        bot = ttk.Labelframe(paned, text="  Topografías theta / alpha / beta  ",
                             style="Card.TLabelframe")
        paned.add(top, weight=1)
        paned.add(bot, weight=1)
        self.psd_canvas = MplCanvas(top, figsize=(9, 3.2), toolbar=True)
        self.psd_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.topo_canvas = MplCanvas(bot, figsize=(9, 3.4), toolbar=True)
        self.topo_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        TH.hsep(self).pack(side=tk.BOTTOM, fill=tk.X, padx=18)
        foot = ttk.Frame(self)
        foot.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=(6, 10))
        ttk.Label(foot, text="Las figuras de la derecha (Fig. 4 y Fig. 5) "
                  "provienen del paper:", style="Muted.TLabel").pack(anchor="w")
        ttk.Label(foot, text="Xiang, C. et al. (2024). A resting-state EEG "
                  "dataset for sleep deprivation. Scientific Data 11:427.  "
                  "https://doi.org/10.1038/s41597-024-03268-2",
                  style="Muted.TLabel", foreground=TH.PRIMARY).pack(anchor="w")

    # ------------------------------------------------------------------ #
    def refresh(self, *_):
        if self.app.analyzer is None:
            return
        if self._running:
            return
        if self._grand is None:
            self._grand = self._load_cache()
        if self._grand is not None:
            self._draw()
            return
        self._compute()

    def _load_cache(self):
        if not os.path.exists(GROUP_NPZ):
            return None
        try:
            z = np.load(GROUP_NPZ, allow_pickle=True)
            ch = [str(c) for c in z["ch"]]
            bands = list(B.BANDS)
            grand = {"freqs": z["freqs"], "ch_names": ch,
                     "psd_db": {}, "band_topo": {}, "n": {}}
            for c in ["rested", "deprived"]:
                grand["psd_db"][c] = z[f"psd_{c}"]
                grand["n"][c] = int(z[f"n_{c}"])
                grand["band_topo"][c] = {b: z[f"topo_{c}_{b}"] for b in bands}
            self.app.set_status("Promedio grupal cargado (precalculado).")
            return grand
        except Exception:
            return None

    def _recompute(self):
        self._grand = None
        self._compute()

    def _compute(self):
        if self._running or self.app.analyzer is None:
            return
        self._running = True
        subs = list(self.app.subjects)
        az = self.app.analyzer

        def job(progress):
            return az.grand_average(subs, progress=progress)

        self.app.set_status(f"Calculando promedio de {len(subs)} sujetos "
                            f"(puede tardar varios minutos)...")
        self._placeholder()
        self.app.worker.run(job, on_done=self._done, on_error=self._err,
                            on_progress=lambda i, n, m: self.app.set_progress(
                                i, n, f"Promediando {m} ({i}/{n})"))

    def _done(self, grand):
        self._grand = grand
        self._running = False
        self.app.set_progress(1, 1)
        n = grand["n"].get("rested", 0)
        self.app.set_status(f"Promedio calculado sobre {n} sujetos.")
        self._draw()

    def _err(self, exc, tb):
        self._running = False
        messagebox.showerror("Error", str(exc))

    # ------------------------------------------------------------------ #
    def _placeholder(self):
        for cv, txt in ((self.psd_canvas, "Calculando PSD grupal..."),
                        (self.topo_canvas, "Calculando topografías grupales...")):
            cv.clear()
            ax = cv.figure.add_subplot(111)
            ax.grid(False)
            ax.text(0.5, 0.5, txt, ha="center", va="center",
                    color=TH.TEXT_MUTED)
            ax.axis("off")
            cv.draw()

    def _paper_image(self, ax, path, title):
        ax.grid(False)
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path), aspect='auto')
        else:
            ax.text(0.5, 0.5, "(figura del paper no encontrada)",
                    ha="center", va="center")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def _draw(self):
        self._draw_psd()
        self._draw_topo()

    def _draw_psd(self):
        g = self._grand
        self.psd_canvas.clear()
        fig = self.psd_canvas.figure
        ax0, ax1 = fig.subplots(1, 2)
        freqs = g["freqs"]
        ch = g["ch_names"]
        cz = ch.index("Cz") if "Cz" in ch else 0
        for cond in ["rested", "deprived"]:
            if cond in g["psd_db"]:
                ax0.plot(freqs, g["psd_db"][cond][cz], color=COLORS[cond],
                         lw=2.0, label=B.CONDITIONS[cond]["short"])
        ax0.set_xlim(B.SPECTRUM_FMIN, B.SPECTRUM_FMAX)
        ax0.set_xlabel("Frecuencia (Hz)")
        ax0.set_ylabel("Potencia absoluta (dB)")
        n = g["n"].get("rested", 0)
        ax0.set_title(f"Nuestros datos  ·  Cz  (n={n})", loc="left")
        ax0.legend()
        self._paper_image(ax1, FIG4, "Paper  ·  Fig. 4 (Cz)")
        fig.tight_layout()
        self.psd_canvas.draw()

    def _draw_topo(self):
        g = self._grand
        self.topo_canvas.clear()
        fig = self.topo_canvas.figure
        bands = list(B.BANDS.keys())
        conds = ["rested", "deprived"]
        ch = g["ch_names"]
        outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.18)
        left = outer[0, 0].subgridspec(2, len(bands), hspace=0.15, wspace=0.05)
        im = None
        topo_axes = []
        # Escala FIJA como el paper (Fig. 5): de -20 a 10 dB (densidad).
        vmin, vmax = -20, 10
        for j, band in enumerate(bands):
            for i, cond in enumerate(conds):
                ax = fig.add_subplot(left[i, j])
                ax.grid(False)
                topo_axes.append(ax)
                if cond not in g["band_topo"]:
                    ax.axis("off")
                    continue
                im = T.plot_band_topo(ax, g["band_topo"][cond][band], ch,
                                      vlim=(vmin, vmax), cmap="jet")
                if i == 0:
                    ax.set_title(band, fontsize=9)
                if j == 0:
                    ax.text(-0.3, 0.5, B.CONDITIONS[cond]["short"],
                            transform=ax.transAxes, fontsize=10,
                            fontweight="bold", va="center", rotation=90,
                            color=TH.NS if cond == "rested" else TH.SD)
        if im is not None:
            fig.colorbar(im, ax=topo_axes, shrink=0.6, label="dB",
                         location="left", pad=0.02)
        ax_paper = fig.add_subplot(outer[0, 1])
        self._paper_image(ax_paper, FIG5, "Paper  ·  Fig. 5")
        self.topo_canvas.draw()
