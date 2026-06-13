import numpy as np


def generate_intervention_map(risk_classes):
    intervention = np.zeros_like(risk_classes, dtype="uint8")

    intervention[risk_classes == 1] = 1
    intervention[risk_classes == 2] = 2
    intervention[risk_classes == 3] = 3
    intervention[risk_classes == 4] = 4

    return intervention


def get_priority_zones(risk_classes, hotspots):
    priority = np.zeros_like(risk_classes, dtype="uint8")
    priority[(risk_classes >= 3) | (hotspots == 1)] = 1
    return priority


def generate_smart_intervention_map(
    slope,
    flow_acc,
    runoff_volume,
    priority_zones
):
    intervention = np.zeros_like(priority_zones, dtype="uint8")

    flow_p90 = np.nanpercentile(flow_acc, 90)
    flow_p97 = np.nanpercentile(flow_acc, 97)
    volume_p97 = np.nanpercentile(runoff_volume, 97)

    low_slope = 5
    moderate_slope = 12

    conservation_mask = (
        (priority_zones == 1) &
        (slope <= moderate_slope) &
        (flow_acc < flow_p97)
    )
    intervention[conservation_mask] = 1

    infiltration_mask = (
        (priority_zones == 1) &
        (slope > low_slope) &
        (slope <= moderate_slope) &
        (flow_acc < flow_p90)
    )
    intervention[infiltration_mask] = 2

    drainage_mask = (
        (flow_acc >= flow_p97) &
        (slope > moderate_slope)
    )
    intervention[drainage_mask] = 3

    storage_mask = (
        (runoff_volume >= volume_p97) &
        (slope <= low_slope)
    )
    intervention[storage_mask] = 4

    return intervention.astype("uint8")

def get_priority_zones(risk_classes, hotspots):
    """
    Define localización prioritaria.
    1 = zona prioritaria
    0 = no prioritaria
    """
    priority = np.zeros_like(risk_classes, dtype="uint8")

    priority[(risk_classes == 4) | (hotspots == 1)] = 1

    return priority


def estimate_runoff_volume(runoff_q, cell_size):
    """
    Calcula volumen preliminar de escorrentía por celda.

    runoff_q: escorrentía en mm
    cell_size: tamaño de celda en metros

    Volumen = Q(mm) * área(m²) / 1000
    """
    cell_area = cell_size * cell_size
    volume = (runoff_q * cell_area) / 1000.0

    return volume


def total_volume_in_hotspots(volume, hotspots):
    """
    Calcula volumen total dentro de zonas críticas.
    """
    total_volume = np.nansum(volume[hotspots == 1])
    return total_volume


def total_volume_in_priority_zones(volume, priority):
    """
    Calcula volumen total dentro de zonas prioritarias.
    """
    total_volume = np.nansum(volume[priority == 1])
    return total_volume


def print_intervention_stats(intervention):
    """
    Imprime distribución de intervenciones.
    """
    unique, counts = np.unique(intervention, return_counts=True)

    print("Distribución de intervenciones:")
    for u, c in zip(unique, counts):
        if u == 1:
            label = "Monitoreo"
        elif u == 2:
            label = "Infiltración"
        elif u == 3:
            label = "Drenaje"
        elif u == 4:
            label = "Intervención prioritaria"
        else:
            label = "Sin clasificar"

        print(f"{label}: {c} celdas")


def print_priority_stats(priority):
    """
    Imprime estadísticas de localización prioritaria.
    """
    total_cells = priority.size
    priority_cells = int(np.sum(priority == 1))
    percentage = (priority_cells / total_cells) * 100

    print("Celdas prioritarias:", priority_cells)
    print("Porcentaje prioritario (%):", percentage)


def print_volume_stats(volume, hotspots, priority):
    """
    Imprime estadísticas del volumen preliminar.
    """
    total_volume = np.nansum(volume)
    hotspot_volume = total_volume_in_hotspots(volume, hotspots)
    priority_volume = total_volume_in_priority_zones(volume, priority)

    print("Volumen total estimado (m³):", total_volume)
    print("Volumen total en hotspots (m³):", hotspot_volume)
    print("Volumen total en zonas prioritarias (m³):", priority_volume)

def generate_smart_intervention_map(
    slope,
    flow_acc,
    runoff_volume,
    priority_zones
):
    """
    Modelo jerárquico territorial de intervención.

    Clases:
    0 = Sin intervención prioritaria
    1 = Conservación hídrica
    2 = Infiltración / recarga
    3 = Drenaje controlado
    4 = Retención / almacenamiento puntual
    5 = Manejo urbano
    """

    intervention = np.zeros_like(priority_zones, dtype="uint8")

    # Umbrales robustos
    flow_p90 = np.nanpercentile(flow_acc, 90)
    flow_p97 = np.nanpercentile(flow_acc, 97)
    flow_p99 = np.nanpercentile(flow_acc, 99)

    volume_p90 = np.nanpercentile(runoff_volume, 90)
    volume_p97 = np.nanpercentile(runoff_volume, 97)

    # Pendientes
    low_slope = 5
    moderate_slope = 12
    high_slope = 20

    # ============================
    # NIVEL 1: CONSERVACIÓN BASE
    # ============================

    conservation_mask = (
        (priority_zones == 1) &
        (slope <= moderate_slope) &
        (flow_acc < flow_p97)
    )

    intervention[conservation_mask] = 1

    # ============================
    # NIVEL 2: INFILTRACIÓN / RECARGA
    # ============================

    infiltration_mask = (
        (priority_zones == 1) &
        (slope > low_slope) &
        (slope <= moderate_slope) &
        (flow_acc < flow_p90) &
        (runoff_volume < volume_p90)
    )

    intervention[infiltration_mask] = 2

    # ============================
    # NIVEL 3: DRENAJE CONTROLADO
    # ============================

    drainage_mask = (
        (flow_acc >= flow_p99) &
        (slope >= moderate_slope)
    )

    intervention[drainage_mask] = 3

    # ============================
    # NIVEL 4: RETENCIÓN PUNTUAL
    # ============================

    storage_mask = (
        (runoff_volume >= volume_p97) &
        (slope <= low_slope) &
        (flow_acc >= flow_p90)
    )

    intervention[storage_mask] = 4

    # ============================
    # PRIORIDAD FINAL:
    # Retención tiene prioridad sobre conservación/infiltración,
    # drenaje queda como corredor de conducción.
    # ============================

    intervention[storage_mask] = 4
    intervention[drainage_mask & ~storage_mask] = 3

    return intervention

def print_smart_intervention_stats(smart_intervention):
    """
    Imprime estadísticas del mapa automático de intervención.
    """
    unique, counts = np.unique(smart_intervention, return_counts=True)

    print("Distribución de intervención inteligente:")
    for u, c in zip(unique, counts):
        if u == 0:
            label = "Sin intervención prioritaria"
        elif u == 1:
            label = "Infiltración"
        elif u == 2:
            label = "Drenaje controlado"
        elif u == 3:
            label = "Almacenamiento / retención"
        else:
            label = "Desconocido"

        print(f"{label}: {c} celdas")

from scipy.ndimage import label
import numpy as np


from scipy.ndimage import label, binary_dilation
import numpy as np


def generate_intervention_zones(smart_intervention, min_pixels=5, buffer_pixels=20):
    """
    Agrupa píxeles de intervención en zonas conectadas.

    Aplica una expansión espacial tipo buffer raster para que las zonas
    no queden como líneas demasiado finas.

    Parámetros:
    - smart_intervention: raster de intervención sugerida
    - min_pixels: tamaño mínimo de zona
    - buffer_pixels: expansión alrededor de píxeles críticos

    Retorna:
    - zones: raster con ID único por zona
    - final_num_zones: número de zonas
    """

    intervention_mask = smart_intervention > 0

    # Expandir zonas críticas para hacerlas visibles y operativas
    if buffer_pixels > 0:
        intervention_mask = binary_dilation(
            intervention_mask,
            iterations=buffer_pixels
        )

    zones, num_zones = label(intervention_mask)

    cleaned_zones = np.zeros_like(zones, dtype="int32")
    new_id = 1

    for zone_id in range(1, num_zones + 1):
        mask = zones == zone_id
        pixel_count = np.sum(mask)

        if pixel_count >= min_pixels:
            cleaned_zones[mask] = new_id
            new_id += 1

    final_num_zones = new_id - 1

    return cleaned_zones, final_num_zones


def classify_intervention_zones(zones, smart_intervention):
    """
    Asigna un tipo dominante de intervención a cada zona.

    0 = sin zona
    1 = infiltración
    2 = drenaje controlado
    3 = almacenamiento / retención
    """

    zone_class = np.zeros_like(zones, dtype="uint8")

    zone_ids = np.unique(zones)
    zone_ids = zone_ids[zone_ids > 0]

    for zone_id in zone_ids:
        mask = zones == zone_id

        values = smart_intervention[mask]
        values = values[values > 0]

        if values.size == 0:
            continue

        unique, counts = np.unique(values, return_counts=True)
        dominant_class = unique[np.argmax(counts)]

        zone_class[mask] = dominant_class

    return zone_class


def print_intervention_zones_stats(zones, zone_class, volume=None, resolution=None):
    """
    Imprime resumen de zonas de intervención.
    """

    zone_ids = np.unique(zones)
    zone_ids = zone_ids[zone_ids > 0]

    print("Número de zonas de intervención:", len(zone_ids))

    for zone_id in zone_ids:
        mask = zones == zone_id
        pixel_count = int(np.sum(mask))

        intervention_type = int(np.nanmax(zone_class[mask]))

        if intervention_type == 1:
            label_name = "Infiltración"
        elif intervention_type == 2:
            label_name = "Drenaje controlado"
        elif intervention_type == 3:
            label_name = "Almacenamiento / retención"
        else:
            label_name = "Sin clasificar"

        print(f"Zona {zone_id}: {label_name} | píxeles: {pixel_count}")

        if resolution is not None:
            area_m2 = pixel_count * resolution * resolution
            print(f"  Área aproximada: {area_m2:,.2f} m²")

        if volume is not None:
            zone_volume = np.nansum(volume[mask])
            print(f"  Volumen estimado: {zone_volume:,.2f} m³")