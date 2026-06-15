"""
Descubrimiento de la estructura BIDS y carga de registros EEG (.set/.fdt).

Dataset: "A Resting-state EEG Dataset for Sleep Deprivation" (ds004902).
Estructura:
    PSIB_Senales/
        sub-XX/
            ses-1/eeg/sub-XX_ses-1_task-eyesopen_eeg.set   (Normal Sleep)
            ses-2/eeg/sub-XX_ses-2_task-eyesopen_eeg.set   (Sleep Deprivation)
"""
from __future__ import annotations

import os
import glob
import re
import warnings

import numpy as np
import mne

from . import bands as B

TASK = "eyesopen"  # solo ojos abiertos, por indicacion de la profesora


def find_subjects(root: str) -> list[str]:
    """Lista los IDs de sujeto (p.ej. 'sub-01') presentes en la carpeta raiz."""
    subs = []
    for path in sorted(glob.glob(os.path.join(root, "sub-*"))):
        if os.path.isdir(path):
            subs.append(os.path.basename(path))
    return subs


def set_path(root: str, subject: str, condition: str) -> str | None:
    """
    Ruta al .set de ojos abiertos para un sujeto y condicion.
    condition: 'rested' o 'deprived'.
    """
    if condition not in B.CONDITIONS:
        raise ValueError(f"Condicion desconocida: {condition}")
    session = B.CONDITIONS[condition]["session"]
    fname = f"{subject}_{session}_task-{TASK}_eeg.set"
    path = os.path.join(root, subject, session, "eeg", fname)
    return path if os.path.exists(path) else None


def available_conditions(root: str, subject: str) -> list[str]:
    """Condiciones disponibles (con archivo presente) para un sujeto."""
    return [c for c in B.CONDITIONS if set_path(root, subject, c) is not None]


def scan_dataset(root: str) -> dict:
    """
    Inventario del dataset: sujetos, condiciones disponibles y completitud.
    Devuelve un dict resumido (no carga las senales).
    """
    subjects = find_subjects(root)
    info = {"root": root, "subjects": {}, "n_subjects": len(subjects)}
    for sub in subjects:
        info["subjects"][sub] = available_conditions(root, sub)
    info["complete"] = [s for s, c in info["subjects"].items()
                        if set(c) == set(B.CONDITIONS)]
    return info


def load_raw(root: str, subject: str, condition: str, preload: bool = True):
    """
    Carga un registro como mne.io.Raw. Aplica el montaje 10-20 estandar para
    habilitar topomaps. No filtra ni preprocesa (eso lo hace preprocessing.py).

    Es TOLERANTE a archivos .fdt truncados/corruptos (p.ej. sub-01 ses-2, cuya
    grabacion termina antes de los 300 s): en ese caso recupera las muestras
    integras directamente del .fdt.
    """
    path = set_path(root, subject, condition)
    if path is None:
        raise FileNotFoundError(
            f"No existe registro {TASK} para {subject} / {condition}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    except RuntimeError:
        raw = _load_truncated(path)
    _apply_montage(raw)
    return raw


def _load_truncated(set_path_):
    """
    Recupera un Raw a partir de un .set cuyo .fdt esta truncado. Lee la
    cabecera (canales, sfreq) sin datos y carga las muestras completas del .fdt.
    El .fdt de EEGLAB esta en orden (n_times, n_channels) float32 little-endian.
    """
    import os
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info_raw = mne.io.read_raw_eeglab(set_path_, preload=False,
                                          verbose="ERROR")
    info = info_raw.info
    n_ch = len(info["ch_names"])
    fdt = os.path.splitext(set_path_)[0] + ".fdt"
    floats = np.fromfile(fdt, dtype="<f4")
    n_times = floats.size // n_ch
    data = floats[: n_times * n_ch].reshape(n_times, n_ch).T.astype("float64")
    data *= 1e-6  # uV (EEGLAB) -> V (MNE)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    warnings.warn(
        f"{os.path.basename(set_path_)}: .fdt truncado; recuperados "
        f"{n_times} muestras ({n_times / info['sfreq']:.1f} s).")
    return raw


def _apply_montage(raw) -> None:
    """Asigna el montaje standard_1020 para posicionar electrodos en topomaps."""
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        # Igualar mayusculas/minusculas con el montaje estandar.
        rename = {}
        mon_names = {n.lower(): n for n in montage.ch_names}
        for ch in raw.ch_names:
            if ch.lower() in mon_names and mon_names[ch.lower()] != ch:
                rename[ch] = mon_names[ch.lower()]
        if rename:
            raw.rename_channels(rename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage(montage, on_missing="warn", match_case=False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"No se pudo aplicar el montaje: {exc}")


def subject_number(subject: str) -> int:
    """Extrae el numero de sujeto para ordenamientos ('sub-07' -> 7)."""
    m = re.search(r"(\d+)", subject)
    return int(m.group(1)) if m else -1
