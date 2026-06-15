"""
Preprocesamiento de EEG 

Replica el pipeline del paper en todo lo posible:
  1. Filtro pasabanda 0.2 - 45 Hz (FIR).
  2. (Notch opcional 50 Hz)
  3. Segmentacion en epochs de 4 s (-> 75 epochs en 5 min).
  4. Rechazo de epochs con artefactos por umbral de amplitud.
  5. Interpolacion de canales malos (opcional).
  6. Re-referencia a promedio (average reference).

"""
from __future__ import annotations

import warnings
import numpy as np
import mne

from . import bands as B

# Parametros por defecto del pipeline (editables desde la GUI).
DEFAULTS = dict(
    l_freq=0.2,
    h_freq=45.0,
    notch=False,          # True para aplicar notch 50 Hz
    notch_freq=50.0,
    epoch_len=4.0,        # segundos (paper: 4 s)
    reject_blinks=False,  # eleccion del usuario: NO por defecto (enfasis posterior)
    blink_uv=150.0,       # umbral pico-pico en canales frontales (uV)
    reject_amp_uv=0.0,    # 0 = sin rechazo global por amplitud
    interpolate_bads=False,
    bad_z=4.0,            # z-score de varianza para marcar canal malo
    average_ref=True,
)


def filter_raw(raw, l_freq=0.2, h_freq=45.0, notch=False, notch_freq=50.0):
    """Filtra una copia del Raw. No modifica el original."""
    raw = raw.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin",
                   verbose="ERROR")
        if notch:
            raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    return raw


def detect_bad_channels(epochs, z_thresh=4.0):
    """
    Marca canales con varianza atipica (z-score robusto sobre la varianza
    media por canal). Devuelve la lista de nombres de canal.
    """
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    var = data.var(axis=2).mean(axis=0)  # varianza media por canal
    med = np.median(var)
    mad = np.median(np.abs(var - med)) + 1e-30
    z = 0.6745 * (var - med) / mad
    bads = [epochs.ch_names[i] for i in np.where(z > z_thresh)[0]]
    return bads


def make_epochs(raw, params: dict | None = None):
    """
    Aplica el pipeline completo y devuelve (epochs, report).

    report incluye: n_epochs_total, n_rejected, bad_channels, etc.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    filt = filter_raw(raw, p["l_freq"], p["h_freq"], p["notch"], p["notch_freq"])

    # Epochs de longitud fija sin solapamiento.
    epochs = mne.make_fixed_length_epochs(
        filt, duration=p["epoch_len"], preload=True, verbose="ERROR")
    n_total = len(epochs)
    report = {"n_epochs_total": n_total, "n_rejected": 0,
              "bad_channels": [], "n_kept": n_total}

    # Interpolacion de canales malos.
    if p["interpolate_bads"]:
        bads = detect_bad_channels(epochs, p["bad_z"])
        if bads:
            epochs.info["bads"] = bads
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                epochs.interpolate_bads(reset_bads=True, verbose="ERROR")
        report["bad_channels"] = bads

    # Rechazo de epochs con parpadeos (amplitud pico-pico en frontales).
    keep = np.ones(len(epochs), dtype=bool)
    if p["reject_blinks"]:
        frontal = [c for c in B.FRONTAL_EOG_PROXY if c in epochs.ch_names]
        if frontal:
            data = epochs.copy().pick(frontal).get_data() * 1e6  # uV
            ptp = data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_frontal)
            keep &= (ptp.max(axis=1) < p["blink_uv"])

    # Rechazo global por amplitud (opcional).
    if p["reject_amp_uv"] and p["reject_amp_uv"] > 0:
        data = epochs.get_data() * 1e6
        ptp = data.max(axis=2) - data.min(axis=2)
        keep &= (ptp.max(axis=1) < p["reject_amp_uv"])

    if not keep.all():
        epochs = epochs[np.where(keep)[0]]
    report["n_rejected"] = int((~keep).sum())
    report["n_kept"] = len(epochs)

    # Re-referencia a promedio (paper: average reference).
    if p["average_ref"] and len(epochs) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epochs.set_eeg_reference("average", verbose="ERROR")

    return epochs, report
