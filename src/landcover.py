import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def load_landcover_aligned(landcover_path, reference_path):
    """
    Carga ESA WorldCover y lo alinea al raster de referencia.
    Usa vecino más cercano para conservar clases categóricas.
    """

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    with rasterio.open(landcover_path) as src:
        landcover_aligned = np.zeros(ref_shape, dtype="uint8")

        reproject(
            source=rasterio.band(src, 1),
            destination=landcover_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )

    return landcover_aligned


def print_landcover_classes(landcover):
    """
    Imprime clases ESA WorldCover presentes en el área.
    """

    labels = {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }

    values, counts = np.unique(landcover, return_counts=True)

    print("Clases ESA WorldCover detectadas:")
    for value, count in zip(values, counts):
        if value == 0:
            continue
        label = labels.get(int(value), "Clase desconocida")
        print(f"  {int(value)} - {label}: {count} celdas")

def apply_landcover_rules(intervention_map, landcover):
    """
    Corrige el mapa de intervención usando ESA WorldCover.

    Clases finales del PMV:
    0 = Sin intervención prioritaria
    1 = Conservación hídrica
    2 = Infiltración / recarga
    3 = Drenaje controlado
    4 = Retención / almacenamiento puntual
    5 = Manejo urbano
    """

    corrected = intervention_map.copy().astype("uint8")

    # Limpiar cualquier valor que no sea clase PMV
    valid_classes = [0, 1, 2, 3, 4, 5]
    corrected[~np.isin(corrected, valid_classes)] = 0
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
    MANGROVE = 95
    MOSS_LICHEN = 100

    active = corrected > 0

    # 1. Agua y nieve/hielo: excluir intervención
    corrected[(landcover == WATER) | (landcover == SNOW_ICE)] = 0

    # 2. Urbano/construido: manejo urbano, no reservorios abiertos
    corrected[(landcover == BUILTUP) & (corrected > 0)] = 5

    # 3. Humedales y manglar: conservación hídrica
    corrected[
        ((landcover == WETLAND) | (landcover == MANGROVE)) & active
    ] = 1

    # 4. Cobertura natural: conservación hídrica
    natural_mask = (
        (landcover == TREE_COVER)
        | (landcover == SHRUBLAND)
        | (landcover == GRASSLAND)
        | (landcover == MOSS_LICHEN)
    )

    # Conservación domina sobre infiltración y drenaje
    corrected[
    natural_mask &
    (
        (corrected == 2) |
        (corrected == 3) |
        (corrected == 4)
    )
    ] = 1

    # 5. Cultivos: permitir infiltración o retención puntual
    # Si el modelo proponía drenaje en cultivo, se convierte a retención puntual.
    corrected[(landcover == CROPLAND) & (corrected == 3)] = 4

    # 6. Suelo desnudo: drenaje/control erosivo
    corrected[(landcover == BARE) & active] = 3
    # ====================================
    # CONSERVACIÓN DOMINA SOBRE DRENAJE
    # EN COBERTURAS NATURALES
    # ====================================

    natural_mask = (
        (landcover == TREE_COVER)
        | (landcover == SHRUBLAND)
        | (landcover == GRASSLAND)
        | (landcover == MOSS_LICHEN)
    )

    corrected[
        natural_mask & (corrected == 3)
    ] = 1

    corrected[
        natural_mask & (corrected == 4)
    ] = 1
    return corrected