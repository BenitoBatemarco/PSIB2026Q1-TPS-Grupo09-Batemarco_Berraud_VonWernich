
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import io as IO
from . import preprocessing as PP
from . import spectral as S
from . import bands as B

# Cache de PSD por sujeto/condicion, junto al codigo. Acelera muchisimo: evita
# re-filtrar / interpolar / Welch en cada cambio de sujeto o de pestana.
PSD_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "subjects_psd.npz")


def _load_psd_cache():
    if not os.path.exists(PSD_CACHE_PATH):
        return None
    try:
        z = np.load(PSD_CACHE_PATH, allow_pickle=True)
        ch = [str(c) for c in z["ch"]]
        freqs = z["freqs"]
        data = {k: z[k] for k in z.files if "__" in k}
        return {"freqs": freqs, "ch": ch, "data": data}
    except Exception:
        return None


class AnalysisResult:
    """Resultado de analizar un (sujeto, condicion)."""
    def __init__(self, subject, condition, epochs, report, psd, freqs, ch_names):
        self.subject = subject
        self.condition = condition
        self.epochs = epochs
        self.report = report
        self.psd = psd            # uV^2/Hz (n_ch, n_freqs)
        self.freqs = freqs
        self.ch_names = ch_names

    def band_powers(self, relative=False, band_dict=None):
        return S.all_band_powers(self.psd, self.freqs, self.ch_names,
                                 band_dict=band_dict, relative=relative)

    def channel_psd_db(self, channel):
        if channel not in self.ch_names:
            return None
        idx = self.ch_names.index(channel)
        return S.to_db(self.psd[idx])


class Analyzer:
    """Carga y analiza registros, cacheando resultados por (sujeto, condicion)."""

    def __init__(self, root, params=None):
        self.root = root
        self.params = dict(PP.DEFAULTS)
        if params:
            self.params.update(params)
        self._cache: dict = {}
        self._disk = _load_psd_cache()   # PSD precalculado (si existe)

    def set_params(self, params):
        """Actualiza parametros de preprocesamiento e invalida la cache."""
        self.params.update(params)
        self._cache.clear()
        # El cache de disco corresponde a los parametros por defecto; si se
        # cambian, se ignora y se recalcula todo.
        self._disk = None

    def _from_disk(self, subject, condition):
        if not self._disk:
            return None
        psd = self._disk["data"].get(f"{subject}__{condition}")
        if psd is None:
            return None
        return AnalysisResult(subject, condition, None, {"cached": True},
                              np.asarray(psd, dtype=float),
                              self._disk["freqs"], list(self._disk["ch"]))

    def analyze(self, subject, condition, force=False) -> AnalysisResult:
        key = (subject, condition)
        if not force and key in self._cache:
            return self._cache[key]
        if not force:
            cached = self._from_disk(subject, condition)
            if cached is not None:
                self._cache[key] = cached
                return cached
        raw = IO.load_raw(self.root, subject, condition, preload=True)
        epochs, report = PP.make_epochs(raw, self.params)
        psd, freqs, ch_names = S.compute_psd(epochs)
        res = AnalysisResult(subject, condition, epochs, report,
                             psd, freqs, ch_names)
        self._cache[key] = res
        return res

    # ---- Batch / agregacion sobre sujetos -------------------------------
    def collect_band_power(self, subjects, channels=None, relative=False,
                           band_dict=None, progress=None):
        """
        Potencia por banda para una lista de sujetos y canales, en ambas
        condiciones. DataFrame largo: subject, condition, band, channel, power.
        """
        if band_dict is None:
            band_dict = B.BANDS
        if channels is None:
            channels = B.REPRESENTATIVE_CHANNELS
        rows = []
        n = len(subjects)
        for i, sub in enumerate(subjects):
            for cond in B.CONDITIONS:
                if IO.set_path(self.root, sub, cond) is None:
                    continue
                res = self.analyze(sub, cond)
                bp = res.band_powers(relative=relative, band_dict=band_dict)
                for band, perch in bp.items():
                    for ch in channels:
                        if ch in perch:
                            rows.append(dict(subject=sub, condition=cond,
                                             band=band, channel=ch,
                                             power=perch[ch]))
            if progress:
                progress(i + 1, n, sub)
        return pd.DataFrame(rows)

    def collect_region_power(self, subjects, relative=False, band_dict=None,
                             progress=None):
        """Potencia por banda promediada por region, formato largo."""
        if band_dict is None:
            band_dict = B.BANDS
        rows = []
        n = len(subjects)
        for i, sub in enumerate(subjects):
            for cond in B.CONDITIONS:
                if IO.set_path(self.root, sub, cond) is None:
                    continue
                res = self.analyze(sub, cond)
                bp = res.band_powers(relative=relative, band_dict=band_dict)
                reg = S.region_band_power(bp)
                for band, perreg in reg.items():
                    for region, val in perreg.items():
                        rows.append(dict(subject=sub, condition=cond,
                                         band=band, region=region, power=val))
            if progress:
                progress(i + 1, n, sub)
        return pd.DataFrame(rows)

    def grand_psd(self, subjects, condition, progress=None):
        """PSD promedio (grand average) en dB sobre sujetos para una condicion."""
        psds, freqs_ref, ch_ref = [], None, None
        n = len(subjects)
        for i, sub in enumerate(subjects):
            if IO.set_path(self.root, sub, condition) is None:
                continue
            res = self.analyze(sub, condition)
            if freqs_ref is None:
                freqs_ref, ch_ref = res.freqs, res.ch_names
            psds.append(S.to_db(res.psd))
            if progress:
                progress(i + 1, n, sub)
        if not psds:
            return None
        arr = np.stack(psds)
        return arr.mean(0), arr.std(0) / np.sqrt(arr.shape[0]), freqs_ref, ch_ref

    def grand_average(self, subjects, progress=None):
        """
        Promedio entre sujetos (todos), para la vista grupal. Devuelve dict con
        freqs, ch_names, psd_db por condicion, band_topo por condicion/banda y n.
        """
        from . import topo as TP
        conds = list(B.CONDITIONS)
        acc_psd = {c: [] for c in conds}
        acc_topo = {c: {b: [] for b in B.BANDS} for c in conds}
        freqs_ref, ch_ref = None, None
        n = len(subjects)
        for i, sub in enumerate(subjects):
            for cond in conds:
                if IO.set_path(self.root, sub, cond) is None:
                    continue
                res = self.analyze(sub, cond)
                if freqs_ref is None:
                    freqs_ref, ch_ref = res.freqs, res.ch_names
                acc_psd[cond].append(S.to_db(res.psd))
                for band in B.BANDS:
                    vals, _ = TP.band_topo_values(res.psd, res.freqs,
                                                  res.ch_names, B.BANDS[band],
                                                  in_db=True)
                    acc_topo[cond][band].append(vals)
            if progress:
                progress(i + 1, n, sub)
        out = {"freqs": freqs_ref, "ch_names": ch_ref,
               "psd_db": {}, "band_topo": {}, "n": {}}
        for cond in conds:
            if acc_psd[cond]:
                out["psd_db"][cond] = np.mean(np.stack(acc_psd[cond]), axis=0)
                out["n"][cond] = len(acc_psd[cond])
                out["band_topo"][cond] = {
                    b: np.mean(np.stack(acc_topo[cond][b]), axis=0)
                    for b in B.BANDS}
        return out
