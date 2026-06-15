"""Panel PSD: comparacion NS vs SD con bandas sombreadas (automatico)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..widgets import MplCanvas
from .. import theme as TH
from core import bands as B


class PsdPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 8))
        ttk.Label(head, text="Espectro de potencia (PSD)",
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(head, text="Método de Welch en dB. NS y SD superpuestos, "
                  "con las bandas theta / alpha / beta sombreadas.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        TH.hsep(self).pack(fill=tk.X, padx=18)

        ctl = ttk.Frame(self)
        ctl.pack(fill=tk.X, padx=18, pady=10)
        ttk.Label(ctl, text="Canal", style="H2.TLabel").pack(side=tk.LEFT)
        chans = B.REPRESENTATIVE_CHANNELS + [c for c in B.POSTERIOR_CHANNELS
                                             if c not in B.REPRESENTATIVE_CHANNELS]
        self.chan = ttk.Combobox(ctl, state="readonly", width=8, values=chans)
        self.chan.set("Oz")
        self.chan.pack(side=tk.LEFT, padx=(8, 14))
        self.chan.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Label(ctl, text="Sugerencia: un canal posterior (Oz, O1, O2) tiene "
                  "menos artefacto ocular.", style="Muted.TLabel").pack(
            side=tk.LEFT)

        self.canvas = MplCanvas(self, figsize=(9, 5.8))
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 14))

    def refresh(self, *_):
        if self.app.analyzer is None:
            return
        ch = self.chan.get()
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        colors = {"rested": TH.NS, "deprived": TH.SD}
        for cond in ["rested", "deprived"]:
            if not self.app.has_condition(cond):
                continue
            res = self.app.get_result(cond)
            psd_db = res.channel_psd_db(ch)
            if psd_db is None:
                continue
            ax.plot(res.freqs, psd_db, color=colors[cond], lw=2.0,
                    label=B.CONDITIONS[cond]["label"])
        band_colors = {"theta": "#FCE6F1", "alpha": "#E5F3E8",
                       "beta": "#ECEAF7"}
        for name, (f1, f2) in B.BANDS.items():
            ax.axvspan(f1, f2, color=band_colors.get(name, "#EEE"), alpha=0.7,
                       zorder=0)
            ax.text((f1 + f2) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color=TH.TEXT_MUTED)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Potencia absoluta (dB)")
        ax.set_title(f"{self.app.current_subject}  ·  canal {ch}", loc="left")
        ax.set_xlim(B.SPECTRUM_FMIN, B.SPECTRUM_FMAX)
        ax.legend()
        self.canvas.figure.tight_layout()
        self.canvas.draw()
