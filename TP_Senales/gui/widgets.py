"""
Widgets reutilizables: lienzo matplotlib embebido y worker de hilos.
"""
from __future__ import annotations

import threading
import queue
import traceback

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from . import theme as TH


class MplCanvas(ttk.Frame):
    """Frame con una figura matplotlib y barra de herramientas de navegacion."""

    def __init__(self, master, figsize=(7, 4.5), toolbar=True):
        super().__init__(master, style="Card.TFrame")
        self.figure = Figure(figsize=figsize, dpi=100, facecolor=TH.SURFACE)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=TH.SURFACE, highlightthickness=0)
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))
        if toolbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self,
                                                pack_toolbar=False)
            try:
                self.toolbar.configure(bg=TH.SURFACE)
                for child in self.toolbar.winfo_children():
                    try:
                        child.configure(bg=TH.SURFACE)
                    except tk.TclError:
                        pass
            except Exception:
                pass
            self.toolbar.update()
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=4)

    def clear(self):
        self.figure.clear()

    def draw(self):
        self.canvas.draw_idle()

    def save(self, path, dpi=200):
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight")


class Worker:
    """
    Ejecuta una funcion en un hilo de fondo y entrega el resultado al hilo
    principal de Tk. Cada llamada a run() usa su PROPIA cola y su propio bucle
    de sondeo, de modo que varios trabajos simultaneos no se mezclan entre si
    (Tk no es thread-safe; por eso se sondea con root.after).
    """

    def __init__(self, root):
        self.root = root

    def run(self, func, on_done=None, on_error=None, on_progress=None):
        q: queue.Queue = queue.Queue()

        def report_progress(i, n, msg=""):
            q.put(("progress", (i, n, msg)))

        def target():
            try:
                q.put(("done", func(report_progress)))
            except Exception as exc:  # noqa
                q.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=target, daemon=True).start()

        def poll():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "progress":
                        if on_progress:
                            on_progress(*payload)
                    elif kind == "done":
                        if on_done:
                            on_done(payload)
                        return
                    elif kind == "error":
                        if on_error:
                            on_error(*payload)
                        return
            except queue.Empty:
                pass
            self.root.after(80, poll)

        poll()
