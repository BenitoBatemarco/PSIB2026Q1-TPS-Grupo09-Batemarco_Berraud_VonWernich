"""
Ventana principal - version "presentacion": simple, dinamica y con un diseno
visual profesional (ver gui/theme.py).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core import io as IO
from core import preprocessing as PP
from core import bands as B
from core import classification as CL
from core.pipeline import Analyzer
from . import theme as TH
from .widgets import Worker
from .panels.signal_panel import SignalPanel
from .panels.psd_panel import PsdPanel
from .panels.bands_panel import BandsPanel
from .panels.topo_panel import TopoPanel
from .panels.all_subjects_panel import AllSubjectsPanel
from .panels.classification_subject_panel import ClassificationSubjectPanel
from .panels.results_panel import ResultsPanel

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIXED_PARAMS = dict(
    l_freq=0.2, h_freq=45.0, notch=False, notch_freq=50.0,
    epoch_len=4.0, reject_blinks=True, blink_uv=150.0,
    reject_amp_uv=0.0, interpolate_bads=True, bad_z=4.0, average_ref=True,
)

CANDIDATE_DIRS = [
    os.path.expanduser(r"~/OneDrive/Escritorio/PSIB_Señales"),
    os.path.expanduser(r"~/OneDrive/Desktop/PSIB_Señales"),
    os.path.expanduser(r"~/Escritorio/PSIB_Señales"),
    os.path.expanduser(r"~/Desktop/PSIB_Señales"),
    os.path.expanduser(r"~/Desktop/PSIB_senales"),
]

TABS = [
    ("signal_panel", SignalPanel, "Señal"),
    ("psd_panel", PsdPanel, "PSD"),
    ("bands_panel", BandsPanel, "Bandas"),
    ("topo_panel", TopoPanel, "Topomapas"),
    ("all_panel", AllSubjectsPanel, "Todos los sujetos"),
    ("clasif_panel", ClassificationSubjectPanel, "Clasificación"),
    ("results_panel", ResultsPanel, "Resultados"),
]


class App(tk.Tk):
    def __init__(self, initial_dir=None):
        super().__init__()
        self.title("PSIB · Análisis EEG · Privación de sueño (ds004902)")
        self.geometry("1240x820")
        self.minsize(1040, 680)
        TH.apply(self)

        self.params = dict(FIXED_PARAMS)
        self.root_dir = ""
        self.analyzer = None
        self.subjects = []
        self.current_subject = None
        self.worker = Worker(self)
        self._raw_cache = {}
        self._stale = set()
        self.feature_df = None
        self.ranking = None
        self.chosen_metric = "Cz_theta_abs"
        self._features_running = False

        self._build_header()
        self._build_toolbar()
        self._build_statusbar()

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 0))

        self.panels = []
        for attr, cls, label in TABS:
            panel = cls(self.notebook, self)
            setattr(self, attr, panel)
            self.panels.append(panel)
            self.notebook.add(panel, text="   " + label + "   ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.after(150, lambda: self._autoload(initial_dir))

    # ------------------------------------------------------------------ #
    def _build_header(self):
        h = tk.Frame(self, bg=TH.PRIMARY, height=64)
        h.pack(side=tk.TOP, fill=tk.X)
        h.pack_propagate(False)
        left = tk.Frame(h, bg=TH.PRIMARY)
        left.pack(side=tk.LEFT, padx=18)
        tk.Label(left, text="PSIB · Análisis EEG", bg=TH.PRIMARY, fg="#FFFFFF",
                 font=TH.F_BRAND).pack(anchor="w", pady=(12, 0))
        tk.Label(left, text="Privación de sueño  ·  dataset ds004902  ·  "
                 "ojos abiertos", bg=TH.PRIMARY, fg="#C7D2FE",
                 font=TH.F_SUB).pack(anchor="w")

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=TH.SURFACE)
        bar.pack(side=tk.TOP, fill=tk.X)
        inner = ttk.Frame(bar, style="Toolbar.TFrame")
        inner.pack(fill=tk.X, padx=18, pady=10)

        ttk.Label(inner, text="Sujeto", style="H2.TLabel").pack(side=tk.LEFT)
        self.subject_var = tk.StringVar()
        self.subject_cb = ttk.Combobox(inner, textvariable=self.subject_var,
                                       state="readonly", width=12,
                                       font=TH.F_BODY)
        self.subject_cb.pack(side=tk.LEFT, padx=(10, 6))
        self.subject_cb.bind("<<ComboboxSelected>>", self._on_subject_change)
        ttk.Button(inner, text="‹", style="Ghost.TButton", width=3,
                   command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(inner, text="›", style="Ghost.TButton", width=3,
                   command=lambda: self._step(1)).pack(side=tk.LEFT, padx=1)

        # Leyenda de condiciones (a la derecha)
        TH.chip(inner, "SD · Privación de sueño", TH.SD).pack(side=tk.RIGHT,
                                                              padx=(16, 0))
        TH.chip(inner, "NS · Descansado", TH.NS).pack(side=tk.RIGHT, padx=16)

        TH.hsep(self).pack(side=tk.TOP, fill=tk.X)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=TH.SURFACE)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        TH.hsep(bar).pack(side=tk.TOP, fill=tk.X)
        row = ttk.Frame(bar, style="Toolbar.TFrame")
        row.pack(fill=tk.X, padx=18, pady=6)
        self.status_var = tk.StringVar(value="Iniciando...")
        ttk.Label(row, textvariable=self.status_var, style="Muted.TLabel").pack(
            side=tk.LEFT)
        self.progress = ttk.Progressbar(row, length=220, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

    def set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def set_progress(self, i, n, msg=""):
        self.progress["maximum"] = max(n, 1)
        self.progress["value"] = i
        if msg:
            self.set_status(msg)
        self.update_idletasks()

    # ------------------------------------------------------------------ #
    def _autoload(self, initial_dir):
        folder = None
        for cand in ([initial_dir] if initial_dir else []) + CANDIDATE_DIRS:
            if cand and os.path.isdir(cand) and IO.find_subjects(cand):
                folder = cand
                break
        if folder is None:
            self.set_status("Seleccioná la carpeta del dataset (PSIB_Señales).")
            self._choose_folder()
            return
        self._load_folder(folder)

    def _choose_folder(self):
        d = filedialog.askdirectory(title="Seleccioná la carpeta PSIB_Señales")
        if d:
            self._load_folder(d)

    def _load_folder(self, folder):
        info = IO.scan_dataset(folder)
        if info["n_subjects"] == 0:
            messagebox.showerror("Error", "No hay carpetas sub-XX en esa ruta.")
            return
        self.root_dir = folder
        self.subjects = list(info["subjects"].keys())
        self.analyzer = Analyzer(folder, params=self.params)
        self.feature_df = None
        self.subject_cb["values"] = self.subjects
        self.current_subject = self.subjects[0]
        self.subject_var.set(self.current_subject)
        self.set_status(f"Dataset cargado · {info['n_subjects']} sujetos · "
                        f"sujeto {self.current_subject}")
        self._on_subject_change()

    def _step(self, d):
        if not self.subjects:
            return
        i = self.subjects.index(self.current_subject)
        i = max(0, min(len(self.subjects) - 1, i + d))
        self.subject_var.set(self.subjects[i])
        self._on_subject_change()

    # ------------------------------------------------------------------ #
    def _on_subject_change(self, _evt=None):
        self.current_subject = self.subject_var.get()
        self._raw_cache.clear()
        self._stale = set(self.panels)
        sub = self.current_subject
        az = self.analyzer
        if az is None:
            return

        def job(progress):
            for k, cond in enumerate(["rested", "deprived"]):
                if IO.set_path(self.root_dir, sub, cond) is not None:
                    az.analyze(sub, cond)
                progress(k + 1, 2, f"Analizando {sub} ({cond})")
            return True

        self.set_status(f"Procesando {sub}...")
        self.worker.run(job,
                        on_done=lambda _: self._refresh_current(),
                        on_error=self._err,
                        on_progress=self.set_progress)

    def _on_tab_changed(self, _evt=None):
        self._refresh_current()

    def _refresh_current(self):
        if self.analyzer is None or self.current_subject is None:
            return
        panel = self.panels[self.notebook.index(self.notebook.select())]
        if panel in self._stale:
            try:
                panel.refresh()
                self._stale.discard(panel)
                self.set_status(f"Sujeto {self.current_subject} · listo")
                self.set_progress(1, 1)
            except Exception as exc:  # noqa
                self._err(exc, "")

    def _err(self, exc, tb):
        messagebox.showerror("Error", str(exc))
        self.set_status(f"Error: {exc}")

    # ---- utilidades compartidas para los paneles ----
    def get_result(self, condition):
        return self.analyzer.analyze(self.current_subject, condition)

    def has_condition(self, condition):
        return IO.set_path(self.root_dir, self.current_subject,
                           condition) is not None

    def get_raw(self, condition):
        # Para la VISUALIZACION de la senal usamos un filtro IIR (rapido),
        # no el FIR del pipeline: es casi instantaneo y visualmente equivalente.
        key = (self.current_subject, condition)
        if key not in self._raw_cache:
            import warnings
            raw = IO.load_raw(self.root_dir, self.current_subject,
                              condition, preload=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.filter(self.params["l_freq"], self.params["h_freq"],
                           method="iir", verbose="ERROR")
            self._raw_cache[key] = raw
        return self._raw_cache[key]

    # ---- clasificacion: tabla de metricas compartida ----
    def ensure_features(self, on_ready, force=False):
        if self.feature_df is not None and not force:
            on_ready()
            return
        if not force:
            csv = os.path.join(APP_ROOT, "features_clasificacion.csv")
            if os.path.exists(csv):
                try:
                    import pandas as pd
                    self.feature_df = pd.read_csv(csv)
                    self._finalize_features()
                    on_ready()
                    return
                except Exception:
                    pass
        if self._features_running or self.analyzer is None:
            return
        self._features_running = True
        subs = list(self.subjects)
        az = self.analyzer

        def job(progress):
            return CL.build_feature_table(az, subs, progress=progress)

        self.set_status("Calculando métricas de todos los sujetos...")
        self.worker.run(
            job,
            on_done=lambda df: self._features_done(df, on_ready),
            on_error=self._err,
            on_progress=lambda i, n, m: self.set_progress(
                i, n, f"Métricas {m} ({i}/{n})"))

    def _features_done(self, df, on_ready):
        self.feature_df = df
        self._features_running = False
        self._finalize_features()
        self.set_progress(1, 1)
        on_ready()

    def _finalize_features(self):
        self.ranking = CL.rank_features(self.feature_df)
        if self.ranking is not None and len(self.ranking):
            self.chosen_metric = self.ranking.iloc[0]["feature"]


def launch(initial_dir=None):
    App(initial_dir=initial_dir).mainloop()
