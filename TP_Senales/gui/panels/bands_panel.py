"""Panel Bandas: potencia por banda y region, NS vs SD (automatico)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

from ..widgets import MplCanvas
from .. import theme as TH
from core import bands as B
from core import spectral as S


class BandsPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 8))
        ttk.Label(head, text="Potencia por banda y región",
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(head, text="Comparación NS vs SD de theta / alpha / beta en "
                  "cada region cerebral.", style="Muted.TLabel").pack(
            anchor="w", pady=(2, 0))
        TH.hsep(self).pack(fill=tk.X, padx=18)

        ctl = ttk.Frame(self)
        ctl.pack(fill=tk.X, padx=18, pady=10)
        ttk.Label(ctl, text="Medida", style="H2.TLabel").pack(side=tk.LEFT,
                                                              padx=(0, 8))
        self.rel = tk.BooleanVar(value=False)
        ttk.Radiobutton(ctl, text="Potencia absoluta (uV2)", value=False,
                        variable=self.rel,
                        command=self.refresh).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(ctl, text="Potencia relativa (%)", value=True,
                        variable=self.rel,
                        command=self.refresh).pack(side=tk.LEFT, padx=6)

        self.canvas = MplCanvas(self, figsize=(9, 5.8))
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 14))

    def refresh(self, *_):
        if self.app.analyzer is None:
            return
        rel = self.rel.get()
        data = {}
        for cond in ["rested", "deprived"]:
            if not self.app.has_condition(cond):
                continue
            res = self.app.get_result(cond)
            data[cond] = S.region_band_power(res.band_powers(relative=rel))
        if not data:
            return
        regions = list(B.REGIONS.keys())
        bands = list(B.BANDS.keys())
        self.canvas.clear()
        axes = self.canvas.figure.subplots(1, len(bands), sharey=True)
        if len(bands) == 1:
            axes = [axes]
        x = np.arange(len(regions))
        for ax, band in zip(axes, bands):
            ns = [data.get("rested", {}).get(band, {}).get(r, np.nan)
                  for r in regions]
            sd = [data.get("deprived", {}).get(band, {}).get(r, np.nan)
                  for r in regions]
            ax.bar(x - 0.2, ns, 0.4, label="NS", color=TH.NS)
            ax.bar(x + 0.2, sd, 0.4, label="SD", color=TH.SD)
            ax.set_xticks(x)
            ax.set_xticklabels(regions, rotation=40, fontsize=8, ha="right")
            ax.set_title(band, loc="center")
            ax.grid(axis="x")
        unit = "relativa" if rel else "uV2"
        axes[0].set_ylabel(f"Potencia {unit}")
        axes[-1].legend()
        self.canvas.figure.suptitle(f"{self.app.current_subject}", fontsize=12,
                                    fontweight="bold")
        self.canvas.figure.tight_layout()
        self.canvas.draw()
