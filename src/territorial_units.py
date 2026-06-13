import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def create_territorial_units(landcover, slope, flow_acc):
    """
    Clasifica el 100% de la cuenca en unidades territoriales funcionales.
    """

    tu = np.zeros_like(landcover, dtype="uint8")

    # ESA WorldCover
    TREE_COVER = 10
    SHRUBLAND = 20
    GRASSLAND = 30
    CROPLAND = 40
    BUILTUP = 50
    BARE = 60
    SNOW_ICE = 70
    WATER = 80
    WETLAND = 90

    # 1. Conservación hídrica
    natural = (
        (landcover == TREE_COVER) |
        (landcover == SHRUBLAND) |
        (landcover == GRASSLAND) |
        (landcover == WETLAND)
    )
    tu[natural] = 1

    # 2. Producción agrohidrológica
    tu[landcover == CROPLAND] = 3

    # 3. Manejo urbano
    tu[landcover == BUILTUP] = 4

    # 4. Drenaje / regulación en zonas de alta acumulación
    valid_acc = flow_acc[np.isfinite(flow_acc) & (flow_acc > 0)]
    acc_threshold = np.nanpercentile(valid_acc, 95)
    tu[flow_acc >= acc_threshold] = 5

    # 5. Recarga / infiltración en pendientes suaves no urbanas
    recharge = (
        (slope <= 12) &
        (tu == 1)
    )
    tu[recharge] = 2

    # Agua, nieve/hielo y nodata quedan sin TU activa
    tu[(landcover == WATER) | (landcover == SNOW_ICE)] = 0
    # 6. Suelo desnudo / áreas abiertas: regulación y control erosivo
    tu[(landcover == BARE) & (tu == 0)] = 5

    # 7. Nieve/hielo: conservación hídrica especial
    tu[(landcover == SNOW_ICE) & (tu == 0)] = 1

    # 8. Relleno final: todo píxel válido no clasificado queda como conservación base
    tu[(tu == 0) & (landcover > 0) & (landcover != WATER)] = 1

    return tu


def tu_label(class_id):
    labels = {
        1: "Conservación hídrica",
        2: "Recarga / infiltración",
        3: "Producción agrohidrológica",
        4: "Manejo urbano",
        5: "Drenaje / regulación",
    }
    return labels.get(int(class_id), "Sin clasificación")


def vectorize_territorial_units(
    tu_map,
    reference_raster_path,
    output_gpkg,
    output_csv,
    min_area_ha=5.0
):
    with rasterio.open(reference_raster_path) as src:
        transform = src.transform
        crs = src.crs

    mask = tu_map > 0

    polygons = []
    records = []

    for geom, value in shapes(
        tu_map.astype("uint8"),
        mask=mask,
        transform=transform
    ):
        geom_shape = shape(geom)
        class_id = int(value)

        area_m2 = geom_shape.area
        area_ha = area_m2 / 10000

        if area_ha < min_area_ha:
            continue

        records.append({
            "tu_id": len(records) + 1,
            "class_id": class_id,
            "tipo_tu": tu_label(class_id),
            "area_m2": area_m2,
            "area_ha": area_ha,
        })

        polygons.append(geom_shape)

    gdf = gpd.GeoDataFrame(records, geometry=polygons, crs=crs)

    if gdf.empty:
        print("No se generaron Territorial Units.")
        return gdf

    gdf.to_file(output_gpkg, layer="territorial_units", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Territorial Units guardadas en:", output_gpkg)
    print("CSV Territorial Units guardado en:", output_csv)
    print("Número de TU:", len(gdf))
    print(gdf["tipo_tu"].value_counts())
    print(gdf["area_ha"].describe())

    return gdf