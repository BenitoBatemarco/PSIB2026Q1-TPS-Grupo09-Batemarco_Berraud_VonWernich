"""
Clasificacion NS vs SD 

Estrategia (analisis exploratorio + fronteras de decision manuales):
  1. Se calculan varias METRICAS por registro (sujeto x condicion).
  2. Para cada metrica se mide su capacidad de SEPARACION entre clases:
       - diferencia de medias y d de Cohen,
       - AUC empirica (= probabilidad de que un SD supere a un NS al azar;
         es un estadistico descriptivo, no un clasificador entrenado),
       - solapamiento de distribuciones.
  3. Se definen UMBRALES MANUALES e interpretables:
       - punto medio entre las medianas de cada clase (regla principal), y
       - punto de corte que maximiza (sensibilidad+especificidad) -1
         (J de Youden) reportado como referencia.
  4. Se evalua la clasificacion resultante (exactitud, sens, espec) y se
     reporta una regla INTRA-SUJETO que aprovecha el diseno pareado.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bands as B
from . import spectral as S

NS, SD = "rested", "deprived"


# --------------------------------------------------------------------------- #
# 1. Extraccion de metricas por registro
# --------------------------------------------------------------------------- #
def _mean_over(perch, channels):
    vals = [perch[c] for c in channels if c in perch]
    return float(np.mean(vals)) if vals else np.nan


def features_for_result(res) -> dict:
    """Metricas (features) de un AnalysisResult (un registro)."""
    abs_bp = res.band_powers(relative=False)
    rel_bp = res.band_powers(relative=True)
    post = B.POSTERIOR_CHANNELS
    occ = B.REGIONS["Occipital"]["channels"]

    f = {}
    # Potencia absoluta (uV^2) en canales posteriores (menos artefacto ocular).
    f["post_theta_abs"] = _mean_over(abs_bp["theta"], post)
    f["post_alpha_abs"] = _mean_over(abs_bp["alpha"], post)
    f["post_beta_abs"] = _mean_over(abs_bp["beta"], post)
    # En dB (como el paper).
    f["post_alpha_dB"] = 10 * np.log10(max(f["post_alpha_abs"], 1e-30))
    f["post_theta_dB"] = 10 * np.log10(max(f["post_theta_abs"], 1e-30))
    # Potencia relativa (%).
    f["post_theta_rel"] = _mean_over(rel_bp["theta"], post)
    f["post_alpha_rel"] = _mean_over(rel_bp["alpha"], post)
    # Occipital y Cz (el paper usa Cz).
    f["occ_alpha_abs"] = _mean_over(abs_bp["alpha"], occ)
    f["Cz_alpha_abs"] = abs_bp["alpha"].get("Cz", np.nan)
    f["Cz_theta_abs"] = abs_bp["theta"].get("Cz", np.nan)
    # Razones espectrales.
    f["alpha_theta_ratio"] = (f["post_alpha_abs"] /
                              max(f["post_theta_abs"], 1e-30))
    f["theta_beta_ratio"] = (f["post_theta_abs"] /
                             max(f["post_beta_abs"], 1e-30))
    # Potencia total (proxy de energia global) en posteriores.
    f["post_total_abs"] = (f["post_theta_abs"] + f["post_alpha_abs"] +
                           f["post_beta_abs"])
    # Potencia ABSOLUTA TOTAL (broadband 0.5-45 Hz). El paper afirma que la
    # potencia absoluta durante SD es generalmente mayor que en NS.
    full = (B.SPECTRUM_FMIN, B.SPECTRUM_FMAX)
    bp_full = S.band_power(res.psd, res.freqs, full)  # por canal (uV2)
    f["total_abs_global"] = float(np.mean(bp_full))   # promedio de canales
    cz = res.ch_names.index("Cz") if "Cz" in res.ch_names else 0
    f["total_abs_cz"] = float(bp_full[cz])            # en Cz (como el paper)
    return f


def build_feature_table(analyzer, subjects, progress=None) -> pd.DataFrame:
    """Tabla de metricas: una fila por (sujeto, condicion)."""
    from . import io as IO
    rows = []
    n = len(subjects)
    for i, sub in enumerate(subjects):
        for cond in (NS, SD):
            if IO.set_path(analyzer.root, sub, cond) is None:
                continue
            res = analyzer.analyze(sub, cond)
            row = {"subject": sub, "condition": cond}
            row.update(features_for_result(res))
            rows.append(row)
        if progress:
            progress(i + 1, n, sub)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2-3. Separabilidad y umbrales manuales
# --------------------------------------------------------------------------- #
def empirical_auc(a, b) -> float:
    """
    AUC empirica: P(x_SD > x_NS). 1 = separacion perfecta (SD mayor),
    0 = separacion perfecta invertida, 0.5 = sin separacion.
    a = valores NS, b = valores SD.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    greater = (b[:, None] > a[None, :]).sum()
    ties = (b[:, None] == a[None, :]).sum()
    return (greater + 0.5 * ties) / (a.size * b.size)


def cohen_d(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) /
                 (na + nb - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


def _classify(values, threshold, sd_is_high):
    """Predice SD/NS aplicando el umbral segun la direccion."""
    values = np.asarray(values)
    if sd_is_high:
        return np.where(values > threshold, SD, NS)
    return np.where(values < threshold, SD, NS)


def _metrics_at(values, labels, threshold, sd_is_high):
    pred = _classify(values, threshold, sd_is_high)
    labels = np.asarray(labels)
    is_sd = labels == SD
    is_ns = labels == NS
    sens = np.mean(pred[is_sd] == SD) if is_sd.any() else np.nan  # detectar SD
    spec = np.mean(pred[is_ns] == NS) if is_ns.any() else np.nan
    acc = np.mean(pred == labels)
    return acc, sens, spec


def youden_threshold(values, labels, sd_is_high):
    """Punto de corte que maximiza sensibilidad+especificidad-1 (referencia)."""
    vals = np.unique(np.asarray(values, dtype=float))
    if vals.size < 2:
        return np.nan, np.nan
    cands = (vals[:-1] + vals[1:]) / 2
    best_t, best_j = cands[0], -1
    for t in cands:
        _, se, sp = _metrics_at(values, labels, t, sd_is_high)
        j = (se + sp) - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t), float(best_j)


def evaluate_feature(df: pd.DataFrame, feature: str) -> dict:
    """Separabilidad y umbrales de UNA metrica."""
    a = df.loc[df.condition == NS, feature].dropna().values
    b = df.loc[df.condition == SD, feature].dropna().values
    vals = df[feature].dropna().values
    labels = df.loc[df[feature].notna(), "condition"].values
    if a.size == 0 or b.size == 0:
        return {}

    sd_is_high = b.mean() >= a.mean()
    auc = empirical_auc(a, b)
    # Umbral principal: punto medio entre medianas (interpretable, robusto).
    thr_med = float((np.median(a) + np.median(b)) / 2)
    acc_m, sens_m, spec_m = _metrics_at(vals, labels, thr_med, sd_is_high)
    # Umbral de referencia: J de Youden.
    thr_y, j = youden_threshold(vals, labels, sd_is_high)
    acc_y, sens_y, spec_y = _metrics_at(vals, labels, thr_y, sd_is_high)

    return {
        "feature": feature,
        "media_NS": float(a.mean()), "media_SD": float(b.mean()),
        "mediana_NS": float(np.median(a)), "mediana_SD": float(np.median(b)),
        "cohen_d": cohen_d(a, b),
        "AUC": auc,
        "separabilidad_abs": abs(auc - 0.5) * 2,  # 0..1
        "direccion": "SD>NS" if sd_is_high else "SD<NS",
        "umbral_mediana": thr_med,
        "acc_mediana": acc_m, "sens_mediana": sens_m, "spec_mediana": spec_m,
        "umbral_youden": thr_y,
        "acc_youden": acc_y, "sens_youden": sens_y, "spec_youden": spec_y,
    }


def rank_features(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Ranking de metricas por capacidad de separacion (|AUC-0.5|)."""
    if features is None:
        features = [c for c in df.columns if c not in ("subject", "condition")]
    rows = [evaluate_feature(df, f) for f in features]
    rows = [r for r in rows if r]
    out = pd.DataFrame(rows).sort_values("separabilidad_abs", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Regla intra-sujeto (aprovecha el diseno pareado)
# --------------------------------------------------------------------------- #
def within_subject_rule(df: pd.DataFrame, feature: str) -> dict:
    """
    Regla pareada: para cada sujeto, se predice como SD el registro con MAYOR
    (o menor, segun direccion) valor de la metrica. Mide el % de pares bien
    ordenados (clasificacion intra-sujeto sin umbral global).
    """
    piv = df.pivot_table(index="subject", columns="condition", values=feature)
    piv = piv.dropna()
    if piv.empty:
        return {}
    sd_is_high = df.loc[df.condition == SD, feature].mean() >= \
        df.loc[df.condition == NS, feature].mean()
    if sd_is_high:
        correct = (piv[SD] > piv[NS]).sum()
    else:
        correct = (piv[SD] < piv[NS]).sum()
    n = len(piv)
    return {"feature": feature, "n_sujetos": int(n),
            "correctos": int(correct), "exactitud": correct / n,
            "direccion": "SD>NS" if sd_is_high else "SD<NS"}


# --------------------------------------------------------------------------- #
# 5. Apoyo para mostrar conclusiones en la interfaz (no tabla)
# --------------------------------------------------------------------------- #
# Metricas clave a comentar por sujeto, con nombre legible, unidad y nota.
METRIC_INFO = {
    "total_abs_cz": ("Potencia absoluta total (Cz)", "uV2",
                     "el paper indica que la potencia absoluta aumenta con la "
                     "privacion de sueno"),
    "total_abs_global": ("Potencia absoluta total (promedio de canales)", "uV2",
                         "energia global del EEG; deberia subir con la "
                         "privacion segun el paper"),
    "Cz_theta_abs": ("Theta en Cz", "uV2",
                     "marcador principal de privacion de sueno (linea media, "
                     "poco afectada por los ojos)"),
    "post_theta_abs": ("Theta posterior", "uV2",
                       "actividad lenta de somnolencia en canales limpios"),
    "post_alpha_abs": ("Alpha posterior", "uV2",
                       "relajacion; mas variable entre sujetos"),
    "alpha_theta_ratio": ("Razon alpha/theta", "",
                          "baja cuando crece la somnolencia (theta sube)"),
    "theta_beta_ratio": ("Razon theta/beta", "",
                         "sube con la presion de sueno"),
}
KEY_METRICS = list(METRIC_INFO.keys())


def group_directions(df: pd.DataFrame, features=None) -> dict:
    """Direccion de la tendencia grupal por metrica: 'SD>NS' o 'SD<NS'."""
    if features is None:
        features = KEY_METRICS
    out = {}
    for f in features:
        if f in df.columns:
            a = df.loc[df.condition == NS, f].mean()
            b = df.loc[df.condition == SD, f].mean()
            out[f] = "SD>NS" if b >= a else "SD<NS"
    return out


def subject_values(df: pd.DataFrame, subject: str) -> dict:
    """Valores NS y SD de cada metrica para un sujeto: {feat: (ns, sd)}."""
    sdf = df[df.subject == subject]
    out = {}
    for f in [c for c in df.columns if c not in ("subject", "condition")]:
        ns = sdf.loc[sdf.condition == NS, f]
        sd = sdf.loc[sdf.condition == SD, f]
        out[f] = (float(ns.iloc[0]) if len(ns) else np.nan,
                  float(sd.iloc[0]) if len(sd) else np.nan)
    return out


def subject_verdict(df: pd.DataFrame, subject: str, feature: str) -> dict:
    """
    Veredicto intra-sujeto para un sujeto y metrica: compara su NS vs SD y dice
    si sigue la tendencia grupal (clasificacion 'bien') o no ('mal').
    """
    direction = group_directions(df, [feature]).get(feature, "SD>NS")
    vals = subject_values(df, subject).get(feature, (np.nan, np.nan))
    ns, sd = vals
    if np.isnan(ns) or np.isnan(sd):
        return {"subject": subject, "feature": feature, "ns": ns, "sd": sd,
                "ok": None, "direction": direction}
    ok = (sd > ns) if direction == "SD>NS" else (sd < ns)
    return {"subject": subject, "feature": feature, "ns": ns, "sd": sd,
            "ok": bool(ok), "direction": direction}


def all_verdicts(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Veredicto bien/mal de cada sujeto para la metrica dada."""
    subs = sorted(df.subject.unique())
    rows = [subject_verdict(df, s, feature) for s in subs]
    return pd.DataFrame(rows)


def predict(value, threshold, sd_is_high):
    """Predice la clase (NS/SD) de UNA senal aplicando un umbral."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if sd_is_high:
        return SD if value > threshold else NS
    return SD if value < threshold else NS


def recording_predictions(df: pd.DataFrame, subject: str, feature: str) -> dict:
    """
    Para un sujeto, clasifica POR SEPARADO sus dos senales (la de descanso y la
    de privacion) usando el umbral global de la metrica, y dice si la prediccion
    es correcta. Devuelve umbral, direccion y filas {true, value, pred, correct}.
    """
    ev = evaluate_feature(df, feature)
    if not ev:
        return {}
    thr = ev["umbral_mediana"]
    sd_is_high = ev["direccion"] == "SD>NS"
    ns, sd = subject_values(df, subject).get(feature, (np.nan, np.nan))
    rows = []
    for true_label, val in [(NS, ns), (SD, sd)]:
        pred = predict(val, thr, sd_is_high)
        rows.append({"true": true_label, "value": val, "pred": pred,
                     "correct": (pred == true_label) if pred is not None
                     else None})
    return {"feature": feature, "threshold": thr,
            "direction": ev["direccion"], "acc_global": ev["acc_mediana"],
            "rows": rows}


# Nombres legibles de TODAS las metricas (para la pestana de metricas).
FEATURE_LABELS = {
    "total_abs_cz": "Potencia absoluta total (Cz)",
    "total_abs_global": "Potencia absoluta total (promedio de canales)",
    "Cz_theta_abs": "Theta en Cz",
    "Cz_alpha_abs": "Alpha en Cz",
    "post_theta_abs": "Theta posterior",
    "post_alpha_abs": "Alpha posterior",
    "post_beta_abs": "Beta posterior",
    "post_theta_dB": "Theta posterior (dB)",
    "post_alpha_dB": "Alpha posterior (dB)",
    "post_theta_rel": "Theta posterior relativa",
    "post_alpha_rel": "Alpha posterior relativa",
    "occ_alpha_abs": "Alpha occipital",
    "alpha_theta_ratio": "Razon alpha/theta",
    "theta_beta_ratio": "Razon theta/beta",
    "post_total_abs": "Potencia total posterior",
}


def feature_label(feat):
    if feat in METRIC_INFO:
        return METRIC_INFO[feat][0]
    return FEATURE_LABELS.get(feat, feat)


def all_recording_predictions(df: pd.DataFrame, feature: str) -> dict:
    """
    Clasifica CADA registro (60 senales) con el umbral global de la metrica y
    compara con la condicion real. Devuelve exactitud y la tabla detallada.
    """
    ev = evaluate_feature(df, feature)
    if not ev:
        return {}
    thr = ev["umbral_mediana"]
    sd_is_high = ev["direccion"] == "SD>NS"
    rows = []
    for _, r in df.iterrows():
        val = r[feature]
        pred = predict(val, thr, sd_is_high)
        rows.append({"subject": r["subject"], "condition": r["condition"],
                     "value": val, "pred": pred,
                     "correct": (pred == r["condition"]) if pred is not None
                     else None})
    tab = pd.DataFrame(rows)
    acc = float(tab["correct"].mean()) if len(tab) else 0.0
    return {"feature": feature, "threshold": thr, "direction": ev["direccion"],
            "acc": acc, "n": int(tab["correct"].notna().sum()),
            "ok": int(tab["correct"].sum()), "table": tab}
