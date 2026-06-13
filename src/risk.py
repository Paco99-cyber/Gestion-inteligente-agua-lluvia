import numpy as np


def minmax_normalize(array):
    """
    Normaliza un array entre 0 y 1 ignorando NaN.
    """
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)

    if arr_max == arr_min:
        return np.zeros_like(array, dtype="float64")

    norm = (array - arr_min) / (arr_max - arr_min)
    return norm


def invert_and_normalize_slope(slope_array):
    """
    Invierte la pendiente para que baja pendiente = mayor riesgo.
    """
    slope_norm = minmax_normalize(slope_array)
    slope_inv = 1.0 - slope_norm
    return slope_inv


def log_normalize_flow_acc(flow_acc):
    """
    Normaliza acumulación usando log(1+x) para reducir extremos.
    """
    flow_log = np.log1p(flow_acc)
    flow_norm = minmax_normalize(flow_log)
    return flow_norm


def calculate_runoff_risk(runoff_q, flow_acc, slope,
                          w_runoff=0.4, w_flow=0.4, w_slope=0.2):
    """
    Calcula índice de riesgo de escorrentía entre 0 y 1.
    """
    runoff_norm = minmax_normalize(runoff_q)
    flow_norm = log_normalize_flow_acc(flow_acc)
    slope_inv = invert_and_normalize_slope(slope)

    risk = (
        w_runoff * runoff_norm +
        w_flow * flow_norm +
        w_slope * slope_inv
    )

    return risk


def classify_risk(risk_array):
    """
    Clasifica el riesgo en 4 clases:
    1 = Bajo
    2 = Medio
    3 = Alto
    4 = Muy alto
    """
    classes = np.zeros_like(risk_array, dtype="uint8")

    classes[(risk_array >= 0.00) & (risk_array < 0.25)] = 1
    classes[(risk_array >= 0.25) & (risk_array < 0.50)] = 2
    classes[(risk_array >= 0.50) & (risk_array < 0.75)] = 3
    classes[(risk_array >= 0.75)] = 4

    return classes


def print_risk_stats(risk_array, risk_classes):
    """
    Imprime estadísticas básicas del mapa de riesgo.
    """
    print("Riesgo mínimo:", np.nanmin(risk_array))
    print("Riesgo máximo:", np.nanmax(risk_array))
    print("Riesgo medio:", np.nanmean(risk_array))

    unique, counts = np.unique(risk_classes, return_counts=True)
    print("Distribución de clases:")
    for u, c in zip(unique, counts):
        print(f"  Clase {u}: {c} celdas")