import numpy as np


def create_cn_raster_like(reference_array, cn_value=75):
    """
    Crea un raster CN uniforme para pruebas.
    """
    cn_array = np.full(reference_array.shape, cn_value, dtype="float64")
    return cn_array


def calculate_s_from_cn(cn_array):
    """
    Calcula S (almacenamiento potencial máximo) a partir del Curve Number.
    Unidades: mm
    """
    cn_array = cn_array.astype("float64")

    # Evitar divisiones inválidas
    cn_array = np.where((cn_array <= 0) | (cn_array > 100), np.nan, cn_array)

    s_array = (25400.0 / cn_array) - 254.0
    return s_array


def calculate_ia(s_array, lambda_ia=0.2):
    """
    Calcula la abstracción inicial Ia.
    """
    ia_array = lambda_ia * s_array
    return ia_array


def calculate_runoff(p_mm, s_array, ia_array):
    """
    Calcula la escorrentía directa Q en mm usando SCS-CN.
    """
    q_array = np.zeros_like(s_array, dtype="float64")

    condition = p_mm > ia_array

    q_array[condition] = ((p_mm - ia_array[condition]) ** 2) / (
        p_mm - ia_array[condition] + s_array[condition]
    )

    q_array[~condition] = 0.0

    return q_array


def print_runoff_stats(q_array):
    """
    Imprime estadísticas básicas de la escorrentía calculada.
    """
    print("Escorrentía mínima (mm):", np.nanmin(q_array))
    print("Escorrentía máxima (mm):", np.nanmax(q_array))
    print("Escorrentía media (mm):", np.nanmean(q_array))

def create_cn_spatial_simple(dem_array):
    """
    Genera un CN espacial simple basado en elevación.
    """
    cn_array = np.zeros_like(dem_array, dtype="float64")

    # calcular percentiles
    p33 = np.nanpercentile(dem_array, 33)
    p66 = np.nanpercentile(dem_array, 66)

    # asignación de CN
    cn_array[dem_array <= p33] = 80   # zona baja
    cn_array[(dem_array > p33) & (dem_array <= p66)] = 70  # media
    cn_array[dem_array > p66] = 60   # alta

    return cn_array
def create_cn_spatial_simple(dem_array):
    """
    Genera un CN espacial simple basado en elevación.
    """
    cn_array = np.zeros_like(dem_array, dtype="float64")

    p33 = np.nanpercentile(dem_array, 33)
    p66 = np.nanpercentile(dem_array, 66)

    cn_array[dem_array <= p33] = 80
    cn_array[(dem_array > p33) & (dem_array <= p66)] = 70
    cn_array[dem_array > p66] = 60

    return cn_array