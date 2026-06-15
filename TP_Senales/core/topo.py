"""
Topografias / topomaps por banda de frecuencia.

Replica la Fig. 5 del paper: topografias de theta, alpha y beta para NS y SD.
Requiere los 61 canales con montaje 10-20 (lo aplica io.load_raw).

Las funciones devuelven valores por canal listos para mne.viz.plot_topomap,
o dibujan directamente sobre ejes matplotlib provistos por la GUI.
"""
from __future__ import annotations

import warnings
import numpy as np
import mne

from . import bands as B
from . import spectral as S


def band_topo_values(psd, freqs, ch_names, band, in_db=True, density=True):
    """
    Valor por canal para una banda (para un topomap).
    Como el paper, se usa la DENSIDAD media de potencia en la banda
    (potencia integrada dividida por el ancho de banda, uV^2/Hz) y se expresa
    en dB. Asi beta (banda ancha) queda por debajo de theta/alpha, igual que en
    la Fig. 5 del paper. Devuelve (valores, ch_names).
    """
    power = S.band_power(psd, freqs, band, relative=False)  # integral (uV^2)
    if density:
        bw = max(band[1] - band[0], 1e-9)
        power = power / bw                                  # uV^2/Hz
    if in_db:
        power = S.to_db(power)
    return power, ch_names


def make_info_for_topo(ch_names):
    """Crea un mne.Info con montaje 10-20 para los canales dados."""
    info = mne.create_info(ch_names, sfreq=1000.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(montage, on_missing="ignore", match_case=False)
    return info


def plot_band_topo(ax, values, ch_names, title="", vlim=(None, None),
                   cmap="RdBu_r", show_names=False):
    """
    Dibuja un topomap en el eje matplotlib 'ax'.
    Devuelve el objeto de imagen (para colorbar).
    """
    info = make_info_for_topo(ch_names)
    # Conserva solo canales con posicion valida.
    pos_ch = info.ch_names
    vals = np.asarray([values[ch_names.index(c)] for c in pos_ch])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im, _ = mne.viz.plot_topomap(
            vals, info, axes=ax, show=False, cmap=cmap,
            vlim=vlim, names=pos_ch if show_names else None,
            contours=6)
    ax.set_title(title, fontsize=10)
    return im


def shared_vlim(list_of_values, symmetric=False):
    """Limites de color compartidos entre varios topomaps (NS vs SD)."""
    allv = np.concatenate([np.asarray(v, dtype=float) for v in list_of_values])
    allv = allv[~np.isnan(allv)]
    if allv.size == 0:
        return (None, None)
    lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    return (lo, hi)
