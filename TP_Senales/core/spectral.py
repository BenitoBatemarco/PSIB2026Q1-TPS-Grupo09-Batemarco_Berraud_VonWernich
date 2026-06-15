"""
Analisis espectral: PSD por metodo de Welch y potencia por banda.

Replica del paper:
  - PSD por Welch.
  - Potencia ABSOLUTA log-transformada en dB:  P_dB = 10 * log10(uV^2 / Hz).
    (El paper usa 1 dB = 10*log(uV^2). Trabajamos en densidad por Hz.)
  - El paper calcula el espectro de Cz; aqui se generaliza a cualquier canal
    o conjunto de canales representativos.

Tambien se calcula potencia RELATIVA por banda (proporcion respecto a la
potencia total 0.5-45 Hz), util para comparaciones intra-sujeto.
"""
from __future__ import annotations

import numpy as np

from . import bands as B

WELCH_FFT_SEC = 4.0  # ventana de 4 s -> resolucion 0.25 Hz (coincide con epoch)


def compute_psd(epochs, fmin=B.SPECTRUM_FMIN, fmax=B.SPECTRUM_FMAX,
                n_fft_sec=WELCH_FFT_SEC):
    """
    PSD por Welch promediada sobre epochs.

    Devuelve (psd, freqs, ch_names):
      psd: array (n_channels, n_freqs) en uV^2/Hz (densidad de potencia).
    """
    sfreq = epochs.info["sfreq"]
    n_fft = int(n_fft_sec * sfreq)
    n_fft = min(n_fft, epochs.get_data().shape[-1])
    spectrum = epochs.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft,
        n_overlap=n_fft // 2, verbose="ERROR")
    psd = spectrum.get_data()          # (n_epochs, n_ch, n_freqs) en V^2/Hz
    psd = psd.mean(axis=0) * 1e12      # promedio epochs -> uV^2/Hz
    freqs = spectrum.freqs
    return psd, freqs, list(spectrum.ch_names)


def to_db(psd):
    """Convierte potencia (uV^2/Hz) a dB: 10*log10(P)."""
    return 10.0 * np.log10(np.maximum(psd, 1e-30))


def band_power(psd, freqs, band, relative=False, total_band=None):
    """
    Potencia por banda integrada sobre frecuencia (regla trapezoidal).

    psd: (n_channels, n_freqs) en uV^2/Hz.
    band: (fmin, fmax).
    relative: si True, divide por la potencia total (total_band o 0.5-45).
    Devuelve array (n_channels,) en uV^2 (absoluta) o proporcion (relativa).
    """
    fmin, fmax = band
    idx = (freqs >= fmin) & (freqs <= fmax)
    abs_power = np.trapz(psd[:, idx], freqs[idx], axis=1)
    if not relative:
        return abs_power
    if total_band is None:
        total_band = (B.SPECTRUM_FMIN, B.SPECTRUM_FMAX)
    tidx = (freqs >= total_band[0]) & (freqs <= total_band[1])
    total = np.trapz(psd[:, tidx], freqs[tidx], axis=1)
    return abs_power / np.maximum(total, 1e-30)


def all_band_powers(psd, freqs, ch_names, band_dict=None,
                    relative=False):
    """
    Calcula potencia de todas las bandas para todos los canales.
    Devuelve dict: {banda: {canal: valor}}.
    """
    if band_dict is None:
        band_dict = B.BANDS
    out = {}
    for name, rng in band_dict.items():
        vals = band_power(psd, freqs, rng, relative=relative)
        out[name] = {ch: float(v) for ch, v in zip(ch_names, vals)}
    return out


def band_power_db(psd, freqs, band):
    """Potencia absoluta de banda expresada en dB (10*log10 de la integral)."""
    p = band_power(psd, freqs, band, relative=False)
    return to_db(p)


def region_band_power(band_powers, relative_label=""):
    """
    Promedia la potencia de cada banda dentro de cada region cerebral.
    band_powers: dict {banda: {canal: valor}} (salida de all_band_powers).
    Devuelve dict {banda: {region: valor_promedio}}.
    """
    out = {}
    for band, perch in band_powers.items():
        out[band] = {}
        for region, info in B.REGIONS.items():
            vals = [perch[ch] for ch in info["channels"] if ch in perch]
            out[band][region] = float(np.mean(vals)) if vals else np.nan
    return out
