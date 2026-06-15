"""
Definicion de bandas de frecuencia, regiones cerebrales y canales
representativos, siguiendo la metodologia del paper (Xiang et al., 2024,
Scientific Data, ds004902).

El paper grafica topografias de THETA, ALPHA y BETA. Mantenemos esas tres
bandas como nucleo del analisis. Se incluye DELTA como banda opcional (no se
usa por defecto) por si se desea explorar el rango lento.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bandas de frecuencia (Hz). Las tres centrales replican el paper.
# ---------------------------------------------------------------------------
BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Banda opcional disponible pero no incluida por defecto en la replica estricta.
OPTIONAL_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
}

# Rango total de analisis espectral (coincide con el filtro del paper).
SPECTRUM_FMIN = 0.5
SPECTRUM_FMAX = 45.0

# ---------------------------------------------------------------------------
# Regiones cerebrales (montaje 10-20 extendido, 61 canales del dataset).
# Cada region lista sus canales y un canal "representativo" de la linea media.
# ---------------------------------------------------------------------------
REGIONS: dict[str, dict] = {
    "Frontal": {
        "channels": ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "AF7", "AF8",
                     "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
        "representative": "Fz",
    },
    "Central": {
        "channels": ["FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
                     "Cz", "C1", "C2", "C3", "C4", "C5", "C6"],
        "representative": "Cz",
    },
    "Temporal": {
        "channels": ["FT7", "FT8", "T7", "T8", "TP7", "TP8", "TP9", "TP10"],
        "representative": "T7",
    },
    "Parietal": {
        "channels": ["CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
                     "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        "representative": "Pz",
    },
    "Occipital": {
        "channels": ["PO3", "PO4", "PO7", "PO8", "POz", "Oz", "O1", "O2"],
        "representative": "Oz",
    },
}

# Canal representativo principal por region (orden anterior -> posterior).
REPRESENTATIVE_CHANNELS: list[str] = [
    REGIONS[r]["representative"] for r in
    ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]
]

# ---------------------------------------------------------------------------
# Canales posteriores. Indicacion de la profesora: priorizar canales
# posteriores, menos contaminados por artefactos oculares (EOG), ya que no
# se permite ICA. Estas son las metricas "de confianza".
# ---------------------------------------------------------------------------
POSTERIOR_CHANNELS: list[str] = [
    "Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8",
    "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
]

# Canales frontales usados para detectar parpadeos (blinks) por amplitud.
FRONTAL_EOG_PROXY: list[str] = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "AF7", "AF8"]

# Mapeo de condiciones: sesion BIDS -> condicion experimental.
# En ds004902, ses-1 = Normal Sleep, ses-2 = Sleep Deprivation.
CONDITIONS: dict[str, dict] = {
    "rested": {"session": "ses-1", "label": "Descansado (NS)", "short": "NS"},
    "deprived": {"session": "ses-2", "label": "Privación de sueño (SD)", "short": "SD"},
}


def region_of(channel: str):
    """Devuelve la region a la que pertenece un canal, o None."""
    for region, info in REGIONS.items():
        if channel in info["channels"]:
            return region
    return None


def active_bands(include_delta: bool = False) -> dict:
    """Diccionario de bandas activas. Por defecto solo theta/alpha/beta."""
    bands = dict(BANDS)
    if include_delta:
        return {**OPTIONAL_BANDS, **bands}
    return bands
