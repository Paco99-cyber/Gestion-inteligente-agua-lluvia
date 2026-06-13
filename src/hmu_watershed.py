import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
from rasterio.features import shapes
from shapely.geometry import shape
from scipy import ndimage
from collections import deque

def create_pour_points(flow_acc, percentile=97, min_distance_pixels=60, max_points=50):
    valid = flow_acc[np.isfinite(flow_acc) & (flow_acc > 0)]
    threshold = np.nanpercentile(valid, percentile)

    streams = flow_acc >= threshold

    local_max = flow_acc == ndimage.maximum_filter(
        flow_acc,
        size=min_distance_pixels
    )

    points = streams & local_max

    coords = np.argwhere(points)

    if len(coords) == 0:
        print("No se encontraron pour points.")
        return []

    values = flow_acc[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]

    coords = coords[order][:max_points]

    pour_points = [(int(r), int(c), i + 1) for i, (r, c) in enumerate(coords)]

    print("Pour points generados:", len(pour_points))

    return pour_points


def delineate_hmu_watersheds(flow_dir, pour_points):
    """
    Delinea microcuencas funcionales aguas arriba de cada pour point.
    Usa propagación inversa D8.
    """

    h, w = flow_dir.shape
    labels = np.zeros((h, w), dtype="int32")

    # D8 tipo PySheds
    d8 = {
        64: (-1, 0),    # N
        128: (-1, 1),   # NE
        1: (0, 1),      # E
        2: (1, 1),      # SE
        4: (1, 0),      # S
        8: (1, -1),     # SW
        16: (0, -1),    # W
        32: (-1, -1),   # NW
    }

    # Para saber qué vecinos drenan hacia una celda
    reverse_dirs = []

    for code, (dr, dc) in d8.items():
        reverse_dirs.append((code, dr, dc))

    for r0, c0, hmu_id in pour_points:
        if r0 < 0 or r0 >= h or c0 < 0 or c0 >= w:
            continue

        queue = deque()
        queue.append((r0, c0))

        while queue:
            r, c = queue.popleft()

            if labels[r, c] != 0:
                continue

            labels[r, c] = hmu_id

            # Buscar vecinos que fluyen hacia esta celda
            for code, dr, dc in reverse_dirs:
                nr = r - dr
                nc = c - dc

                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue

                if labels[nr, nc] != 0:
                    continue

                if int(flow_dir[nr, nc]) == code:
                    queue.append((nr, nc))

    print("HMU watershed labels:", np.unique(labels))

    return labels

def vectorize_hmu_watersheds(
    hmu_labels,
    intervention_map,
    volume_map,
    reference_raster_path,
    output_gpkg,
    output_csv,
    min_area_ha=0.5
):
    with rasterio.open(reference_raster_path) as src:
        transform = src.transform
        crs = src.crs

    mask = hmu_labels > 0
    polygons = []
    records = []

    labels = {
        1: "Conservación hídrica",
        2: "Infiltración / recarga",
        3: "Drenaje controlado",
        4: "Retención / almacenamiento puntual",
        5: "Manejo urbano",
    }

    acciones = {
        1: "Conservación/restauración hídrica",
        2: "Zanjas de infiltración / recarga",
        3: "Drenaje controlado",
        4: "Reservorio / almacenamiento temporal",
        5: "Manejo urbano SUDS",
    }

    for geom, value in shapes(
        hmu_labels.astype("int32"),
        mask=mask,
        transform=transform
    ):
        geom_shape = shape(geom)
        hmu_id = int(value)

        hmu_mask = hmu_labels == hmu_id

        area_m2 = geom_shape.area
        area_ha = area_m2 / 10000

        if area_ha < min_area_ha:
            continue

        values = intervention_map[hmu_mask]
        values = values[values > 0]

        if len(values) > 0:
            classes, counts = np.unique(values, return_counts=True)
            dominant = int(classes[np.argmax(counts)])
        else:
            dominant = 0

        volumen_m3 = float(np.nansum(volume_map[hmu_mask]))

        records.append({
            "hmu_id": hmu_id,
            "class_id": dominant,
            "tipo_hmu": labels.get(dominant, "Sin intervención"),
            "area_m2": area_m2,
            "area_ha": area_ha,
            "volumen_m3": volumen_m3,
            "accion_recomendada": acciones.get(dominant, "Sin acción definida")
        })

        polygons.append(geom_shape)

    gdf = gpd.GeoDataFrame(records, geometry=polygons, crs=crs)

    if gdf.empty:
        print("No se generaron HMU watershed.")
        return gdf

    gdf["prioridad"] = "Baja"

    q50 = gdf["volumen_m3"].quantile(0.50)
    q75 = gdf["volumen_m3"].quantile(0.75)
    q90 = gdf["volumen_m3"].quantile(0.90)

    gdf.loc[gdf["volumen_m3"] >= q50, "prioridad"] = "Media"
    gdf.loc[gdf["volumen_m3"] >= q75, "prioridad"] = "Alta"
    gdf.loc[gdf["volumen_m3"] >= q90, "prioridad"] = "Muy alta"

    gdf.to_file(output_gpkg, layer="hmu_watershed", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("HMU watershed guardadas en:", output_gpkg)
    print("CSV HMU watershed guardado en:", output_csv)
    print("Número HMU watershed:", len(gdf))
    print(gdf["area_ha"].describe())

    return gdf