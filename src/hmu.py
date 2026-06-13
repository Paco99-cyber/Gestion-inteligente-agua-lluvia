import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
print("HMU.PY CARGADO")

def classify_hmu_type(values):
    """
    Define intervención dominante dentro de una HMU.
    """
    values = values[values > 0]

    if len(values) == 0:
        return "Sin intervención"

    unique, counts = np.unique(values, return_counts=True)
    dominant = unique[np.argmax(counts)]

    labels = {
        1: "Conservación hídrica",
        2: "Infiltración / recarga",
        3: "Drenaje controlado",
        4: "Retención / almacenamiento puntual",
        5: "Manejo urbano"
    }

    return labels.get(int(dominant), "Sin clasificar")


from scipy import ndimage

def generate_hmu_map(flow_acc, intervention_map, min_acc_percentile=95, expansion_pixels=8):
    """
    Genera HMU preliminares usando corredores de acumulación
    y expansión territorial alrededor de ellos.
    """

    valid = flow_acc[np.isfinite(flow_acc) & (flow_acc > 0)]
    threshold = np.nanpercentile(valid, min_acc_percentile)

    corridors = flow_acc >= threshold

    expanded = ndimage.binary_dilation(
        corridors,
        structure=np.ones((3, 3)),
        iterations=expansion_pixels
    )

    hmu_map = np.where(expanded, intervention_map, 0).astype("uint8")

    return hmu_map


def vectorize_hmu(hmu_map, reference_transform, crs, intervention_map, volume_map, output_gpkg, output_csv):
    """
    Vectoriza HMU y genera tabla territorial.
    """

    mask = hmu_map > 0

    polygons = []
    records = []

    for geom, value in shapes(
        hmu_map.astype("uint8"),
        mask=mask,
        transform=reference_transform
    ):
        geom_shape = shape(geom)
        class_id = int(value)

        records.append({
            "class_id": class_id,
            "tipo_hmu": classify_hmu_type(np.array([class_id])),
        })

        polygons.append(geom_shape)

    gdf = gpd.GeoDataFrame(
        records,
        geometry=polygons,
        crs=crs
    )

    if gdf.empty:
        print("No se generaron HMU.")
        return gdf

    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000

    # Filtrar HMU pequeñas
    MIN_HMU_HA = 1.0
    gdf = gdf[gdf["area_ha"] >= MIN_HMU_HA].copy()

    if gdf.empty:
        print("No hay HMU mayores al área mínima.")
        return gdf

    # Crear ID
    gdf["hmu_id"] = range(1, len(gdf) + 1)

    # Prioridad simple por área
    gdf["prioridad"] = "Media"
    gdf.loc[gdf["area_ha"] >= 5, "prioridad"] = "Alta"
    gdf.loc[gdf["area_ha"] >= 10, "prioridad"] = "Muy alta"

    gdf.to_file(output_gpkg, layer="hmu", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("HMU guardadas en:", output_gpkg)
    print("CSV HMU guardado en:", output_csv)
    print("Número de HMU:", len(gdf))

    return gdf