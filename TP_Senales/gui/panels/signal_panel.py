"""Panel Senal: las dos condiciones (NS y SD) del sujeto, en la misma pantalla."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

from ..widgets import MplCanvas
from .. import theme as TH
from core import bands as B

CHAN_SETS = {
    "Representativos": B.REPRESENTATIVE_CHANNELS,
    "Posteriores": B.POSTERIOR_CHANNELS,
    "Frontales": B.FRONTAL_EOG_PROXY,
}


class SignalPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 8))
        ttk.Label(head, text="Señal EEG  ·  NS vs SD", style="H1.TLabel").pack(
            anchor="w")
        ttk.Label(head, text="Las dos condiciones del sujeto, filtradas "
                  "(0.2-45 Hz, ojos abiertos).",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        TH.hsep(self).pack(fill=tk.X, padx=18)

        ctl = ttk.Frame(self)
        ctl.pack(fill=tk.X, padx=18, pady=10)
        ttk.Label(ctl, text="Canales", style="H2.TLabel").pack(side=tk.LEFT)
        self.chan_mode = ttk.Combobox(ctl, state="readonly", width=15,
                                      values=list(CHAN_SETS.keys()))
        self.chan_mode.set("Representativos")
        self.chan_mode.pack(side=tk.LEFT, padx=(8, 18))
        self.chan_mode.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Label(ctl, text="Inicio (s)", style="Muted.TLabel").pack(side=tk.LEFT)
        self.start = tk.DoubleVar(value=0.0)
        ttk.Scale(ctl, from_=0, to=290, variable=self.start, length=320,
                  command=lambda e: self.refresh()).pack(side=tk.LEFT, padx=8)
        ttk.Label(ctl, text="Ventana (s)", style="Muted.TLabel").pack(side=tk.LEFT)
        self.win = tk.IntVar(value=10)
        ttk.Spinbox(ctl, from_=4, to=30, textvariable=self.win, width=4,
                    command=self.refresh).pack(side=tk.LEFT, padx=8)

        self.canvas = MplCanvas(self, figsize=(9, 6))
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 14))

    def refresh(self, *_):
        if self.app.analyzer is None:
            return
        chans = CHAN_SETS[self.chan_mode.get()]
        self.canvas.clear()
        fig = self.canvas.figure
        conds = ["rested", "deprived"]
        axes = fig.subplots(2, 1, sharex=True)
        colors = {"rested": TH.NS, "deprived": TH.SD}
        for ax, cond in zip(axes, conds):
            ax.grid(False)
            if not self.app.has_condition(cond):
                ax.text(0.5, 0.5, f"Sin datos ({cond})", ha="center")
                ax.axis("off")
                continue
            raw = self.app.get_raw(cond)
            chs = [c for c in chans if c in raw.ch_names]
            sf = raw.info["sfreq"]
            t0 = float(self.start.get())
            w = int(self.win.get())
            i0, i1 = int(t0 * sf), min(int((t0 + w) * sf), raw.n_times)
            data = raw.get_data(picks=chs)[:, i0:i1] * 1e6
            times = np.arange(i0, i1) / sf
            spacing = np.nanpercentile(np.abs(data), 98) * 2.5 + 1e-9
            for k, ch in enumerate(chs):
                ax.plot(times, data[k] + k * spacing, lw=0.5,
                        color=colors[cond])
                ax.text(times[0], k * spacing, ch + " ", ha="right",
                        va="center", fontsize=8, color=TH.TEXT_MUTED)
            ax.set_yticks([])
            ax.set_ylabel(B.CONDITIONS[cond]["short"], fontsize=12,
                          fontweight="bold", color=colors[cond])
            ax.set_title(B.CONDITIONS[cond]["label"], fontsize=10,
                         color=colors[cond], loc="left")
            ax.margins(x=0)
        axes[-1].set_xlabel("Tiempo (s)")
        fig.suptitle(f"{self.app.current_subject}", fontsize=12,
                     fontweight="bold")
        fig.tight_layout()
        self.canvas.draw()
