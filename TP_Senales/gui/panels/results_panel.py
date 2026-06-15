# -*- coding: utf-8 -*-
"""
Pestaña "Resultados": contenedor con dos subpestañas.

  A) "Por sujeto": regla intra-sujeto (cada persona se compara consigo misma);
     KPI con la exactitud y la lista BIEN/MAL por sujeto.
  B) "Predicción por señal": clasifica cada una de las 60 señales con el umbral
     global y muestra si la predicción (descansado vs privación) es correcta.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

from .. import theme as TH
from core import classification as CL


def _scrolled_text(parent, mono=True):
    frame = ttk.Frame(parent)
    txt = tk.Text(frame, wrap="none",
                  font=(TH.MONO if mono else TH.FAMILY, 10), padx=18, pady=10,
                  relief="flat", borderwidth=0, background=TH.SURFACE,
                  foreground=TH.TEXT, highlightthickness=0, cursor="arrow",
                  spacing3=3)
    sb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    txt.tag_configure("bien", foreground=TH.SUCCESS,
                      font=(TH.MONO, 10, "bold"))
    txt.tag_configure("mal", foreground=TH.DANGER, font=(TH.MONO, 10, "bold"))
    txt.tag_configure("dim", foreground=TH.TEXT_MUTED)
    txt.tag_configure("hdr", foreground=TH.PRIMARY_DARK,
                      font=(TH.MONO, 10, "bold"))
    return frame, txt


def _hero(parent):
    wrap = tk.Frame(parent, bg=TH.SURFACE)
    hero = tk.Frame(wrap, bg=TH.PRIMARY_LIGHT)
    hero.pack(fill=tk.X)
    inner = tk.Frame(hero, bg=TH.PRIMARY_LIGHT)
    inner.pack(fill=tk.X, padx=24, pady=20)
    big = tk.Label(inner, text="–", font=(TH.FAMILY, 48, "bold"),
                   bg=TH.PRIMARY_LIGHT, fg=TH.PRIMARY_DARK)
    big.pack(side=tk.LEFT)
    info = tk.Frame(inner, bg=TH.PRIMARY_LIGHT)
    info.pack(side=tk.LEFT, padx=22)
    title = tk.Label(info, text="", font=(TH.FAMILY, 17, "bold"),
                     bg=TH.PRIMARY_LIGHT, fg=TH.TEXT, anchor="w",
                     justify="left")
    title.pack(anchor="w")
    detail = tk.Label(info, text="", font=TH.F_BODY, bg=TH.PRIMARY_LIGHT,
                      fg=TH.TEXT_MUTED, anchor="w", justify="left")
    detail.pack(anchor="w", pady=(4, 0))
    return wrap, big, title, detail


# =========================================================================== #
class ResultsPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 6))
        col = ttk.Frame(head)
        col.pack(side=tk.LEFT)
        ttk.Label(col, text="Resultados de la clasificación",
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(col, text="Por sujeto (intra-sujeto) y predicción de cada "
                  "señal por umbral.", style="Muted.TLabel").pack(anchor="w",
                                                                  pady=(2, 0))
        rcol = ttk.Frame(head)
        rcol.pack(side=tk.RIGHT)
        ttk.Label(rcol, text="Métrica", style="Muted.TLabel").pack(side=tk.LEFT,
                                                                   padx=(0, 6))
        self.metric_cb = ttk.Combobox(
            rcol, state="readonly", width=24,
            values=[CL.METRIC_INFO[m][0] for m in CL.KEY_METRICS])
        _def = (self.app.chosen_metric if self.app.chosen_metric in
                CL.METRIC_INFO else CL.KEY_METRICS[0])
        self.metric_cb.set(CL.METRIC_INFO[_def][0])
        self.metric_cb.pack(side=tk.LEFT)
        self.metric_cb.bind("<<ComboboxSelected>>", lambda e: self._show())

        self.subnb = ttk.Notebook(self)
        self.subnb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 10))
        self.subj_panel = SubjectResultsPanel(self.subnb, self)
        self.rec_panel = RecordingResultsPanel(self.subnb, self)
        self.subpanels = [self.subj_panel, self.rec_panel]
        self.subnb.add(self.subj_panel, text="   Por sujeto   ")
        self.subnb.add(self.rec_panel, text="   Predicción por señal   ")
        self.subnb.bind("<<NotebookTabChanged>>", lambda e: self._show())

    def selected_feature(self):
        label = self.metric_cb.get()
        for m in CL.KEY_METRICS:
            if CL.METRIC_INFO[m][0] == label:
                return m
        return CL.KEY_METRICS[0]

    def refresh(self, *_):
        if self.app.analyzer is None:
            return
        if self.app.feature_df is None:
            self.app.ensure_features(self._show)
        else:
            self._show()

    def _show(self):
        if self.app.feature_df is None:
            return
        idx = self.subnb.index(self.subnb.select())
        self.subpanels[idx].refresh()


# =========================================================================== #
class SubjectResultsPanel(ttk.Frame):
    def __init__(self, master, container):
        super().__init__(master)
        self.c = container
        self.app = container.app
        self.hero, self.big, self.title, self.detail = _hero(self)
        self.hero.pack(fill=tk.X, padx=14, pady=(12, 6))
        self.frame, self.txt = _scrolled_text(self)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 12))

    def refresh(self, *_):
        df = self.app.feature_df
        if df is None:
            return
        feat = self.c.selected_feature()
        label, unit, _ = CL.METRIC_INFO[feat]
        ver = CL.all_verdicts(df, feat)
        ok = int(ver["ok"].sum())
        n = int(ver["ok"].notna().sum())
        acc = ok / n if n else 0
        direction = ver["direction"].iloc[0] if len(ver) else ""
        self.big.config(text=f"{acc:.0%}")
        self.title.config(text=label)
        self.detail.config(text=f"{ok} de {n} sujetos bien clasificados   ·   "
                                f"tendencia {direction}")
        t = self.txt
        t.config(state="normal")
        t.delete("1.0", tk.END)
        t.insert(tk.END, "   Regla intra-sujeto: SD se asigna al registro con "
                 f"mayor valor de la métrica ({direction}).\n\n", "dim")
        unit_s = f" {unit}" if unit else ""
        for _, r in ver.iterrows():
            ns = "n/d" if np.isnan(r["ns"]) else f"{r['ns']:.2f}"
            sd = "n/d" if np.isnan(r["sd"]) else f"{r['sd']:.2f}"
            t.insert(tk.END, f"   {r['subject']:8s}   NS {ns:>8}{unit_s}   →   "
                     f"SD {sd:>8}{unit_s}     ")
            if r["ok"] is None or (isinstance(r["ok"], float)
                                   and np.isnan(r["ok"])):
                t.insert(tk.END, "—  s/d\n", "dim")
            elif r["ok"]:
                t.insert(tk.END, "✓  BIEN\n", "bien")
            else:
                t.insert(tk.END, "✗  MAL\n", "mal")
        t.config(state="disabled")


# =========================================================================== #
class RecordingResultsPanel(ttk.Frame):
    def __init__(self, master, container):
        super().__init__(master)
        self.c = container
        self.app = container.app
        self.hero, self.big, self.title, self.detail = _hero(self)
        self.hero.pack(fill=tk.X, padx=14, pady=(12, 6))
        self.frame, self.txt = _scrolled_text(self)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 12))

    def refresh(self, *_):
        df = self.app.feature_df
        if df is None:
            return
        feat = self.c.selected_feature()
        label, unit, _ = CL.METRIC_INFO[feat]
        arp = CL.all_recording_predictions(df, feat)
        if not arp:
            return
        self.big.config(text=f"{arp['acc']:.0%}")
        self.title.config(text=label)
        op = ">" if arp["direction"] == "SD>NS" else "<"
        self.detail.config(
            text=f"{arp['ok']} de {arp['n']} señales bien clasificadas   ·   "
                 f"regla: valor {op} {arp['threshold']:.2f} → privación")
        t = self.txt
        t.config(state="normal")
        t.delete("1.0", tk.END)
        t.insert(tk.END, "   Cada señal se clasifica por separado con el umbral "
                 "global.\n\n", "dim")
        tab = arp["table"]
        unit_s = f" {unit}" if unit else ""
        for sub in sorted(tab["subject"].unique()):
            sd_rows = tab[tab["subject"] == sub]
            t.insert(tk.END, f"   {sub}\n", "hdr")
            for _, r in sd_rows.iterrows():
                real = "Descansado" if r["condition"] == CL.NS else "Privación"
                pred = ("Descansado" if r["pred"] == CL.NS else
                        "Privación" if r["pred"] == CL.SD else "n/d")
                val = "n/d" if np.isnan(r["value"]) else f"{r['value']:.2f}"
                t.insert(tk.END, f"      real {real:11s}  valor {val:>7}{unit_s}"
                         f"   →  predicho {pred:11s}  ")
                if r["correct"]:
                    t.insert(tk.END, "✓\n", "bien")
                elif r["correct"] is False:
                    t.insert(tk.END, "✗\n", "mal")
                else:
                    t.insert(tk.END, "—\n", "dim")
        t.config(state="disabled")
