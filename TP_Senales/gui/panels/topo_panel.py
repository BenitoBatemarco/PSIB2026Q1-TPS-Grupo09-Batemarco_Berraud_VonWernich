"""Panel Topomapas: theta/alpha/beta para NS y SD del sujeto (automatico)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..widgets import MplCanvas
from .. import theme as TH
from core import bands as B
from core import topo as T


class TopoPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 8))
        ttk.Label(head, text="Mapas topográficos",
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(head, text="Distribucion de theta / alpha / beta en el cuero "
                  "cabelludo.  Fila 1: Descansado (NS)   ·   Fila 2: Privación "
                  "(SD).", style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        TH.hsep(self).pack(fill=tk.X, padx=18)

        self.canvas = MplCanvas(self, figsize=(9, 6))
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=18, pady=(8, 14))

    def refresh(self, *_):
        if self.app.analyzer is None:
            return
        bands = list(B.BANDS.keys())
        conds = ["rested", "deprived"]
        vals = {c: {} for c in conds}
        ch_ref = None
        for cond in conds:
            if not self.app.has_condition(cond):
                continue
            res = self.app.get_result(cond)
            ch_ref = res.ch_names
            for band in bands:
                v, _ = T.band_topo_values(res.psd, res.freqs, res.ch_names,
                                          B.BANDS[band], in_db=True)
                vals[cond][band] = v
        if ch_ref is None:
            return

        self.canvas.clear()
        fig = self.canvas.figure
        axes = fig.subplots(2, len(bands))
        im = None
        # Escala FIJA como el paper (Fig. 5): de -20 a 10 dB (densidad).
        vmin, vmax = -20, 10
        for j, band in enumerate(bands):
            for i, cond in enumerate(conds):
                ax = axes[i][j]
                ax.grid(False)
                if band not in vals[cond]:
                    ax.axis("off")
                    continue
                im = T.plot_band_topo(ax, vals[cond][band], ch_ref,
                                      vlim=(vmin, vmax), cmap="jet")
                if i == 0:
                    ax.set_title(band, fontsize=12)
                if j == 0:
                    ax.text(-0.25, 0.5, B.CONDITIONS[cond]["short"],
                            transform=ax.transAxes, fontsize=13,
                            fontweight="bold", va="center", ha="center",
                            rotation=90,
                            color=TH.NS if cond == "rested" else TH.SD)
        if im is not None:
            fig.colorbar(im, ax=list(axes.ravel()), shrink=0.6, label="dB")
        fig.suptitle(f"{self.app.current_subject}", fontsize=12,
                     fontweight="bold")
        self.canvas.draw()
