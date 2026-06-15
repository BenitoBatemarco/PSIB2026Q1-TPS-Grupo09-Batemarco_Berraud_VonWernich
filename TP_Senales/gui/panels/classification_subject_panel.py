# -*- coding: utf-8 -*-
"""
Pestaña "Clasificación": contenedor con tres subpestañas.

  A) "Potencia absoluta en Cz": verifica el hallazgo del paper (la potencia
     absoluta es mayor en SD que en NS) para el sujeto actual, con veredicto.
  B) "Clasificación por umbrales": predice, para cada señal del sujeto, si
     corresponde a descansado o privado, con una métrica y un umbral manual.
  Sin Machine Learning.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

from .. import theme as TH
from core import classification as CL

PAPER_METRIC = "total_abs_cz"


def _fmt(v, unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/d"
    return f"{v:.2f}{(' ' + unit) if unit else ''}"


# =========================================================================== #
class ClassificationSubjectPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=18, pady=(16, 6))
        ttk.Label(head, text="Clasificación del sujeto",
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(head, text="Verificamos el hallazgo del paper, predecimos la "
                  "condición de cada señal y justificamos la métrica elegida.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        self.subnb = ttk.Notebook(self)
        self.subnb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 10))
        self.abs_panel = AbsPowerCzPanel(self.subnb, app)
        self.pred_panel = PredictionPanel(self.subnb, app)
        self.subpanels = [self.abs_panel, self.pred_panel]
        self.subnb.add(self.abs_panel, text="   Potencia absoluta en Cz   ")
        self.subnb.add(self.pred_panel, text="   Clasificación por umbrales   ")
        self.subnb.bind("<<NotebookTabChanged>>", lambda e: self._show())

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
class AbsPowerCzPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        wrap = tk.Frame(self, bg=TH.SURFACE)
        wrap.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(wrap, text="Potencia absoluta total en Cz",
                 font=TH.F_H1, bg=TH.SURFACE, fg=TH.TEXT).pack(anchor="w")
        tk.Label(wrap, text="El paper afirma que la potencia absoluta es mayor "
                 "durante la privación (SD) que en descanso (NS). Lo "
                 "verificamos en este sujeto (comparación intra-sujeto).",
                 font=TH.F_SMALL, bg=TH.SURFACE, fg=TH.TEXT_MUTED,
                 wraplength=820, justify="left").pack(anchor="w", pady=(2, 14))

        hero = tk.Frame(wrap, bg=TH.SURFACE_ALT)
        hero.pack(fill=tk.X, pady=6)
        inner = tk.Frame(hero, bg=TH.SURFACE_ALT)
        inner.pack(fill=tk.X, padx=22, pady=20)

        tns = tk.Frame(inner, bg=TH.SURFACE_ALT)
        tns.pack(side=tk.LEFT)
        self.ns_val = tk.Label(tns, text="–", font=(TH.FAMILY, 30, "bold"),
                               bg=TH.SURFACE_ALT, fg=TH.NS)
        self.ns_val.pack()
        tk.Label(tns, text="NS · Descansado  (µV²)", font=TH.F_SMALL,
                 bg=TH.SURFACE_ALT, fg=TH.TEXT_MUTED).pack()

        tk.Label(inner, text="→", font=(TH.FAMILY, 24), bg=TH.SURFACE_ALT,
                 fg=TH.TEXT_MUTED).pack(side=tk.LEFT, padx=26)

        tsd = tk.Frame(inner, bg=TH.SURFACE_ALT)
        tsd.pack(side=tk.LEFT)
        self.sd_val = tk.Label(tsd, text="–", font=(TH.FAMILY, 30, "bold"),
                               bg=TH.SURFACE_ALT, fg=TH.SD)
        self.sd_val.pack()
        tk.Label(tsd, text="SD · Privación  (µV²)", font=TH.F_SMALL,
                 bg=TH.SURFACE_ALT, fg=TH.TEXT_MUTED).pack()

        self.verdict = tk.Label(inner, text="", font=(TH.FAMILY, 13, "bold"),
                                padx=18, pady=12, bg=TH.SURFACE_ALT,
                                fg=TH.TEXT_MUTED)
        self.verdict.pack(side=tk.RIGHT)


    def refresh(self, *_):
        df = self.app.feature_df
        if df is None:
            return
        sub = self.app.current_subject
        ns, sd = CL.subject_values(df, sub).get(PAPER_METRIC, (np.nan, np.nan))
        self.ns_val.config(text=_fmt(ns))
        self.sd_val.config(text=_fmt(sd))
        if np.isnan(ns) or np.isnan(sd):
            self.verdict.config(text="SIN DATOS", bg=TH.SURFACE_ALT,
                                fg=TH.TEXT_MUTED)
        elif sd > ns:
            self.verdict.config(text="✓  CUMPLE CON EL PAPER", bg=TH.SUCCESS_BG,
                                fg=TH.SUCCESS)
        else:
            self.verdict.config(text="✗  NO CUMPLE", bg=TH.DANGER_BG,
                                fg=TH.DANGER)


# =========================================================================== #
class PredictionPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        wrap = tk.Frame(self, bg=TH.SURFACE)
        wrap.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(wrap, text="Predicción de la condición de cada señal",
                 font=TH.F_H1, bg=TH.SURFACE, fg=TH.TEXT).pack(anchor="w",
                                                              pady=(0, 12))

        # Regla de decisión, en varias líneas
        rule = tk.Frame(wrap, bg=TH.PRIMARY_LIGHT)
        rule.pack(fill=tk.X, pady=4)
        ri = tk.Frame(rule, bg=TH.PRIMARY_LIGHT)
        ri.pack(fill=tk.X, padx=18, pady=14)
        tk.Label(ri, text="Regla de decisión", font=TH.F_H2,
                 bg=TH.PRIMARY_LIGHT, fg=TH.PRIMARY_DARK).pack(anchor="w")
        self.rule_metric = tk.Label(ri, text="", font=TH.F_BODY,
                                    bg=TH.PRIMARY_LIGHT, fg=TH.TEXT)
        self.rule_metric.pack(anchor="w", pady=(6, 0))
        self.rule_thr = tk.Label(ri, text="", font=TH.F_BODY,
                                 bg=TH.PRIMARY_LIGHT, fg=TH.TEXT)
        self.rule_thr.pack(anchor="w")
        self.rule_hi = tk.Label(ri, text="", font=TH.F_BODY,
                                bg=TH.PRIMARY_LIGHT, fg=TH.SD)
        self.rule_hi.pack(anchor="w", pady=(6, 0))
        self.rule_lo = tk.Label(ri, text="", font=TH.F_BODY,
                                bg=TH.PRIMARY_LIGHT, fg=TH.NS)
        self.rule_lo.pack(anchor="w")

        self.rows = tk.Frame(wrap, bg=TH.SURFACE)
        self.rows.pack(fill=tk.X, pady=(14, 0))

    def _row(self, true_label, value, pred, correct, unit):
        card = tk.Frame(self.rows, bg=TH.SURFACE_ALT)
        card.pack(fill=tk.X, pady=6)
        inner = tk.Frame(card, bg=TH.SURFACE_ALT)
        inner.pack(fill=tk.X, padx=18, pady=14)
        name = ("Señal de DESCANSO  (real: NS)" if true_label == CL.NS
                else "Señal de PRIVACIÓN  (real: SD)")
        left = tk.Frame(inner, bg=TH.SURFACE_ALT)
        left.pack(side=tk.LEFT)
        tk.Label(left, text=name, font=(TH.FAMILY, 12, "bold"),
                 bg=TH.SURFACE_ALT, fg=TH.TEXT).pack(anchor="w")
        tk.Label(left, text=f"valor de la métrica = {_fmt(value, unit)}",
                 font=TH.F_BODY, bg=TH.SURFACE_ALT, fg=TH.TEXT_MUTED).pack(
            anchor="w", pady=(3, 0))
        pred_txt = ("Descansado" if pred == CL.NS else
                    "Privado de sueño" if pred == CL.SD else "n/d")
        if correct:
            bg, fg, mark = TH.SUCCESS_BG, TH.SUCCESS, "✓ correcto"
        elif correct is False:
            bg, fg, mark = TH.DANGER_BG, TH.DANGER, "✗ incorrecto"
        else:
            bg, fg, mark = TH.SURFACE, TH.TEXT_MUTED, "—"
        rt = tk.Frame(inner, bg=TH.SURFACE_ALT)
        rt.pack(side=tk.RIGHT)
        tk.Label(rt, text=f"Predicción: {pred_txt}",
                 font=(TH.FAMILY, 13, "bold"), bg=TH.SURFACE_ALT,
                 fg=TH.TEXT).pack(anchor="e")
        tk.Label(rt, text=mark, font=TH.F_BTN, padx=12, pady=4, bg=bg,
                 fg=fg).pack(anchor="e", pady=(4, 0))

    def refresh(self, *_):
        df = self.app.feature_df
        if df is None:
            return
        for w in self.rows.winfo_children():
            w.destroy()
        sub = self.app.current_subject
        feat = self.app.chosen_metric
        label, unit, _ = CL.METRIC_INFO.get(feat, (CL.feature_label(feat),
                                                    "", ""))
        rp = CL.recording_predictions(df, sub, feat)
        if not rp:
            return
        hi = "mayor" if rp["direction"] == "SD>NS" else "menor"
        self.rule_metric.config(text=f"Métrica:  {label}")
        self.rule_thr.config(text=f"Umbral:  {rp['threshold']:.2f} {unit}")
        self.rule_hi.config(
            text=f"Si el valor es {hi} al umbral  →  Privado de sueño (SD)")
        self.rule_lo.config(
            text=f"Si no  →  Descansado (NS)")
        for r in rp["rows"]:
            self._row(r["true"], r["value"], r["pred"], r["correct"], unit)
