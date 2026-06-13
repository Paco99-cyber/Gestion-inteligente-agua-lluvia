import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def intervention_label(value):
    if value == 1:
        return "Conservación hídrica"
    elif value == 2:
        return "Infiltración / recarga"
    elif value == 3:
        return "Drenaje controlado"
    elif value == 4:
        return "Retención / almacenamiento puntual"
    elif value == 5:
        return "Manejo urbano"
    else:
        return "Sin intervención"


def classify_priority(volume_m3):
    """
    Clasifica prioridad según volumen preliminar.
    """
    if volume_m3 >= 5000:
        return "Muy alta"
    elif volume_m3 >= 2000:
        return "Alta"
    elif volume_m3 >= 500:
        return "Media"
    else:
        return "Baja"

def vectorize_intervention_zones(
    zones_class_path,
    volume_path,
    output_gpkg,
    output_csv,
    min_area_ha=0.005
):
    """
    Vectoriza zonas de intervención y genera tabla resumen.
    """

    MIN_AREA_HA = min_area_ha

    with rasterio.open(zones_class_path) as src:
        zones_class = src.read(1)
        transform = src.transform
        crs = src.crs
        pixel_area = abs(src.res[0] * src.res[1])

        mask = zones_class > 0

        polygons = []
        values = []

        for geom, value in shapes(
            zones_class.astype("uint8"),
            mask=mask,
            transform=transform
        ):
            polygons.append(shape(geom))
            values.append(int(value))

    with rasterio.open(volume_path) as vol_src:
        volume = vol_src.read(1)

    mean_volume_per_pixel = np.nanmean(volume[volume > 0])

    records = []

    for i, (geom, value) in enumerate(zip(polygons, values), start=1):
        area_m2 = geom.area
        area_ha = area_m2 / 10000.0

        estimated_pixels = area_m2 / pixel_area
        volumen_m3 = estimated_pixels * mean_volume_per_pixel

        records.append({
            "zone_id": i,
            "class_id": value,
            "tipo_intervencion": intervention_label(value),
            "area_m2": area_m2,
            "area_ha": area_ha,
            "volumen_m3": volumen_m3
        })

    gdf = gpd.GeoDataFrame(
        records,
        geometry=polygons,
        crs=crs
    )

    # Eliminar zonas sin intervención
    gdf = gdf[gdf["class_id"] > 0].copy()

    print("Zonas antes del filtro:", len(gdf))
    print(gdf["area_ha"].describe())

    # Filtrar microzonas
    zonas_antes = len(gdf)

    gdf = gdf[gdf["area_ha"] >= MIN_AREA_HA].copy()

    print("Zonas después del filtro:", len(gdf))

    if gdf.empty:
        print("No hay zonas válidas después del filtrado.")
        return gdf

    # Prioridad por volumen
    gdf["prioridad"] = gdf["volumen_m3"].apply(classify_priority)

   
    # Recalcular área después del dissolve
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000

    print("Área mínima:", gdf["area_ha"].min())
    print("Área máxima:", gdf["area_ha"].max())
    print("Área media:", gdf["area_ha"].mean())
    print(gdf["area_ha"].describe())

    # Recalcular prioridad después del dissolve
    gdf["prioridad"] = gdf["volumen_m3"].apply(classify_priority)

    # Crear ID final
    gdf["zone_id"] = range(1, len(gdf) + 1)

    # Guardar GeoPackage
    gdf.to_file(output_gpkg, layer="intervention_zones", driver="GPKG")

    # Guardar CSV sin geometría
    table = gdf.drop(columns="geometry")
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Zonas antes del filtro:", zonas_antes)
    print("Zonas después del filtro:", len(gdf))

    print("Número final de zonas:", len(gdf))

    print("GeoPackage guardado en:", output_gpkg)
    print("CSV resumen guardado en:", output_csv)
    print("Número de zonas exportadas:", len(gdf))

    return gdf
