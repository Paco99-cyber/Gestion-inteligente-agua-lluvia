import numpy as np
import rasterio
from pathlib import Path

from config import (
    LAMBDA_IA,
    RISK_WEIGHTS,
    RISK_THRESHOLDS,
    MIN_HOTSPOT_AREA,
    DEFAULT_RESOLUTION,
)

from io_utils import read_raster, print_raster_info, save_raster

from terrain import (
    read_dem_array,
    print_dem_stats,
    calculate_slope,
    print_slope_stats,
    calculate_flow,
    print_flow_stats,
    extract_drainage_network,
    print_drainage_stats,
    identify_hotspots,
    print_hotspot_stats,
)

from hydrology import (
    create_cn_spatial_simple,
    calculate_s_from_cn,
    calculate_ia,
    calculate_runoff,
    print_runoff_stats,
)

from risk import (
    calculate_runoff_risk,
    classify_risk,
    print_risk_stats,
)

from intervention import (
    generate_intervention_map,
    get_priority_zones,
    estimate_runoff_volume,
    total_volume_in_hotspots,
    total_volume_in_priority_zones,
    print_intervention_stats,
    print_priority_stats,
    print_volume_stats,
    generate_smart_intervention_map,
    print_smart_intervention_stats,
    generate_intervention_zones,
    classify_intervention_zones,
    print_intervention_zones_stats,
)
from zone_export import vectorize_intervention_zones
from hmu import generate_hmu_map, vectorize_hmu

from landcover import (
    load_landcover_aligned,
    print_landcover_classes,
    apply_landcover_rules
)
from masking import apply_basin_mask
from rasterio.features import sieve
from scipy import ndimage
from hmu_watershed import (
    create_pour_points,
    delineate_hmu_watersheds,
    vectorize_hmu_watersheds
)

from territorial_units import create_territorial_units, vectorize_territorial_units
from pmu import generate_pmu

def smooth_intervention_classes(array, iterations=1):
    """
    Agrupa píxeles cercanos por clase usando morfología raster.
    No crea nuevas clases; solo compacta manchas existentes.
    """

    cleaned = np.zeros_like(array, dtype="uint8")

    for class_id in [1, 2, 3, 4, 5]:
        mask = array == class_id

        # Cierra pequeños huecos y conecta píxeles cercanos
        mask_clean = ndimage.binary_closing(
            mask,
            structure=np.ones((3, 3)),
            iterations=iterations
        )


        cleaned[mask_clean] = class_id

    return cleaned
# =========================
# CONFIGURACIÓN GENERAL
# =========================

print("El código está funcionando correctamente")
print("Lambda Ia:", LAMBDA_IA)
print("Pesos de riesgo:", RISK_WEIGHTS)
print("Umbrales:", RISK_THRESHOLDS)
print("Área mínima de hotspot:", MIN_HOTSPOT_AREA)
print("Resolución base:", DEFAULT_RESOLUTION)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

# Crear carpeta outputs si no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dem_path = RAW_DIR / "dem.tif"

print("Ruta DEM:", dem_path)
print("Existe DEM:", dem_path.exists())

if not dem_path.exists():
    raise FileNotFoundError(f"No se encontró el DEM en: {dem_path}")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

LANDCOVER_PATH = DATA_DIR / "raw" / "worldcover.tif"
BASIN_PATH = DATA_DIR / "raw" / "limite_boundary.shp"
# =========================
# LECTURA DEM
# =========================

dem = read_raster(dem_path)
print_raster_info(dem)

dem_array = read_dem_array(dem)
print_dem_stats(dem_array)

landcover = load_landcover_aligned(
    LANDCOVER_PATH,
    dem_path
)
print("DEM shape:", dem.shape)
print("Landcover shape:", landcover.shape)
print("Clases WorldCover alineadas:", np.unique(landcover))
print_landcover_classes(landcover)

print("Cobertura alineada correctamente")

# =========================
# PENDIENTE
# =========================

resolution = dem.res[0]
slope = calculate_slope(dem_array, resolution)
print_slope_stats(slope)

slope_output = OUTPUT_DIR / "slope.tif"
save_raster(slope_output, slope, dem, dtype="float32", nodata=np.nan)
print("Raster de pendiente guardado en:", slope_output)


# =========================
# FLUJO Y ACUMULACIÓN
# =========================

flow_dir, flow_acc = calculate_flow(str(dem_path))
print_flow_stats(flow_acc)

flowacc_output = OUTPUT_DIR / "flow_acc.tif"
save_raster(flowacc_output, flow_acc, dem, dtype="float32", nodata=np.nan)
print("Raster de acumulación guardado en:", flowacc_output)


# =========================
# RED DE DRENAJE
# =========================

drainage_threshold = 200
drainage = extract_drainage_network(flow_acc, drainage_threshold)

print("Umbral de drenaje:", drainage_threshold)
print_drainage_stats(drainage)

drainage_output = OUTPUT_DIR / "drainage_network.tif"
save_raster(drainage_output, drainage, dem, dtype="uint8", nodata=0)
print("Raster de drenaje guardado en:", drainage_output)


# =========================
# HOTSPOTS / ZONAS CRÍTICAS
# =========================

acc_threshold = 40
slope_threshold = 8

hotspots = identify_hotspots(flow_acc, slope, acc_threshold, slope_threshold)

print("Umbral acumulación hotspots:", acc_threshold)
print("Umbral pendiente hotspots:", slope_threshold)
print_hotspot_stats(hotspots)

hotspot_output = OUTPUT_DIR / "hotspots.tif"
save_raster(hotspot_output, hotspots, dem, dtype="uint8", nodata=0)
print("Raster de hotspots guardado en:", hotspot_output)


# =========================
# SCS-CN / ESCORRENTÍA
# =========================

p_event = 50  # lluvia del evento en mm

# CN espacial simple
cn_array = create_cn_spatial_simple(dem_array)

cn_output = OUTPUT_DIR / "cn_map.tif"
save_raster(cn_output, cn_array, dem, dtype="float32", nodata=np.nan)
print("Raster CN guardado en:", cn_output)

# Cálculo SCS-CN
s_array = calculate_s_from_cn(cn_array)
ia_array = calculate_ia(s_array, lambda_ia=LAMBDA_IA)
q_array = calculate_runoff(p_event, s_array, ia_array)

print("Lluvia del evento (mm):", p_event)
print_runoff_stats(q_array)

runoff_output = OUTPUT_DIR / "runoff_q.tif"
save_raster(runoff_output, q_array, dem, dtype="float32", nodata=np.nan)
print("Raster de escorrentía guardado en:", runoff_output)


# =========================
# MAPA DE RIESGO DE ESCORRENTÍA
# =========================

risk_array = calculate_runoff_risk(
    runoff_q=q_array,
    flow_acc=flow_acc,
    slope=slope,
    w_runoff=0.4,
    w_flow=0.4,
    w_slope=0.2,
)

risk_classes = classify_risk(risk_array)
print_risk_stats(risk_array, risk_classes)

risk_output = OUTPUT_DIR / "runoff_risk.tif"
save_raster(risk_output, risk_array, dem, dtype="float32", nodata=np.nan)
print("Raster de riesgo guardado en:", risk_output)

risk_class_output = OUTPUT_DIR / "runoff_risk_classes.tif"
save_raster(risk_class_output, risk_classes, dem, dtype="uint8", nodata=0)
print("Raster de clases de riesgo guardado en:", risk_class_output)

# =========================
# INTERVENCIÓN
# =========================

intervention = generate_intervention_map(risk_classes)
print_intervention_stats(intervention)

intervention_output = OUTPUT_DIR / "intervention_map.tif"
save_raster(intervention_output, intervention, dem, dtype="uint8", nodata=0)
print("Mapa de intervención guardado en:", intervention_output)


# =========================
# VOLUMEN PRELIMINAR REQUERIDO
# =========================

volume = estimate_runoff_volume(q_array, resolution)

# ============================
# LOCALIZACIÓN PRIORITARIA
# ============================

high_flow = flow_acc >= np.nanpercentile(flow_acc, 90)
high_volume = volume >= np.nanpercentile(volume, 90)
high_risk = risk_classes >= 3

priority = (high_flow & high_volume).astype("uint8")

print_priority_stats(priority)

priority_output = OUTPUT_DIR / "priority_zones.tif"
save_raster(priority_output, priority, dem, dtype="uint8", nodata=0)

print("Mapa de localización prioritaria guardado en:", priority_output)

print_volume_stats(volume, hotspots, priority)

volume_output = OUTPUT_DIR / "runoff_volume.tif"
save_raster(volume_output, volume, dem, dtype="float32", nodata=np.nan)
print("Raster de volumen preliminar guardado en:", volume_output)

high_flow = flow_acc >= np.nanpercentile(flow_acc, 98)

priority = (high_flow & high_volume).astype("uint8")
# =========================
# MAPA AUTOMÁTICO DE INTERVENCIÓN
# =========================

smart_intervention = generate_smart_intervention_map(
    slope=slope,
    flow_acc=flow_acc,
    runoff_volume=volume,
    priority_zones=priority,
)

print_smart_intervention_stats(smart_intervention)
print("Valores únicos:", np.unique(smart_intervention))

smart_intervention_output = OUTPUT_DIR / "smart_intervention_map.tif"
save_raster(
    smart_intervention_output,
    smart_intervention,
    dem,
    dtype="uint8",
    nodata=0
)

print("Mapa automático de intervención guardado en:", smart_intervention_output)
# =========================
# ZONAS DE INTERVENCIÓN AGRUPADAS
# =========================

intervention_zones, num_zones = generate_intervention_zones(
    smart_intervention,
    min_pixels=20,
    buffer_pixels=12
)

intervention_zones_class = classify_intervention_zones(
    intervention_zones,
    landcover
)

print_intervention_zones_stats(
    intervention_zones,
    intervention_zones_class,
    volume=volume,
    resolution=resolution
)

zones_output = OUTPUT_DIR / "intervention_zones.tif"

print("Clase 1:", np.sum(intervention_zones_class == 1))
print("Clase 3:", np.sum(intervention_zones_class == 3))
print("Clase 4:", np.sum(intervention_zones_class == 4))
print("Clase 5:", np.sum(intervention_zones_class == 5))

save_raster(
    zones_output,
    intervention_zones,
    dem,
    dtype="int32",
    nodata=0
)
print("Mapa de zonas de intervención guardado en:", zones_output)
print("Clases ANTES de reglas:")
print(np.unique(intervention_zones_class))

zones_class_output = OUTPUT_DIR / "intervention_zones_class.tif"

print("Clases ANTES de reglas:")
print("smart_intervention:", np.unique(smart_intervention))
print("Coberturas:", np.unique(landcover))

intervention_zones_class = apply_landcover_rules(
    smart_intervention,
    landcover
)

print("DESPUÉS DE REGLAS")
print("intervention_zones_class:", np.unique(intervention_zones_class))


print("PIXELES POR CLASE")
for c in [1,3,4,5]:
    print(
        f"Clase {c}:",
        np.count_nonzero(intervention_zones_class == c)
    )


intervention_zones_class = apply_basin_mask(
    intervention_zones_class,
    dem_path,
    BASIN_PATH
)

intervention_zones_class = smooth_intervention_classes(
    intervention_zones_class,
    iterations=1
)

print("Filtro morfológico aplicado. Valores:", np.unique(intervention_zones_class))

print("Filtro sieve aplicado. Valores:", np.unique(intervention_zones_class))
save_raster(
    zones_class_output,
    intervention_zones_class,
    dem,
    dtype="uint8",
    nodata=0
)
print("Mapa de clases de zonas de intervención guardado en:", zones_class_output)
print("Valores únicos intervención:", np.unique(intervention_zones_class))
intervention_zones_class = apply_landcover_rules(
    intervention_zones_class,
    landcover
)

print("Corrección territorial aplicada")

# ============================
# TERRITORIAL UNITS - 100% CUENCA
# ============================

tu_map = create_territorial_units(
    landcover=landcover,
    slope=slope,
    flow_acc=flow_acc
)

tu_map = apply_basin_mask(
    tu_map,
    dem_path,
    BASIN_PATH
)

import geopandas as gpd
from rasterio.features import geometry_mask

with rasterio.open(dem_path) as src:
    basin = gpd.read_file(BASIN_PATH).to_crs(src.crs)

    basin_mask = geometry_mask(
        basin.geometry,
        out_shape=src.shape,
        transform=src.transform,
        invert=True
    )

total_pixels = np.count_nonzero(basin_mask)
zero_pixels = np.count_nonzero((tu_map == 0) & basin_mask)

print("\nDIAGNÓSTICO TERRITORIAL UNITS - MÁSCARA REAL")
print("Pixeles reales dentro de cuenca:", total_pixels)
print("Pixeles clase 0 dentro de cuenca:", zero_pixels)
print(
    "Porcentaje clase 0 real:",
    round(zero_pixels / total_pixels * 100, 2),
    "%"
)

tu_output = OUTPUT_DIR / "territorial_units.tif"

save_raster(
    tu_output,
    tu_map,
    dem,
    dtype="uint8",
    nodata=0
)

print("Mapa Territorial Units guardado en:", tu_output)
print("Valores TU:", np.unique(tu_map))

tu_gpkg = OUTPUT_DIR / "territorial_units.gpkg"
tu_csv = OUTPUT_DIR / "territorial_units_summary.csv"

vectorize_territorial_units(
    tu_map=tu_map,
    reference_raster_path=dem_path,
    output_gpkg=tu_gpkg,
    output_csv=tu_csv,
    min_area_ha=10.0
)

# ============================
# HMU REALES POR WATERSHED
# ============================

pour_points = create_pour_points(
    flow_acc=flow_acc,
    percentile=97,
    min_distance_pixels=60,
    max_points=50
)

hmu_watershed = delineate_hmu_watersheds(
    flow_dir=flow_dir,
    pour_points=pour_points
)

hmu_watershed = apply_basin_mask(
    hmu_watershed,
    dem_path,
    BASIN_PATH
)

hmu_watershed_output = OUTPUT_DIR / "hmu_watershed_map.tif"

save_raster(
    hmu_watershed_output,
    hmu_watershed,
    dem,
    dtype="int32",
    nodata=0
)



# =========================
# VECTORIAL / TABLA DE ZONAS
# =========================

zones_class_path = OUTPUT_DIR / "intervention_zones_class.tif"
volume_path = OUTPUT_DIR / "runoff_volume.tif"

zones_gpkg_output = OUTPUT_DIR / "intervention_zones.gpkg"
zones_csv_output = OUTPUT_DIR / "intervention_zones_summary.csv"

vectorize_intervention_zones(
    zones_class_path=OUTPUT_DIR / "intervention_zones_class.tif",
    volume_path=OUTPUT_DIR / "runoff_volume.tif",
    output_gpkg=OUTPUT_DIR / "intervention_zones.gpkg",
    output_csv=OUTPUT_DIR / "intervention_zones_summary.csv"
)
# ============================
# HMU - HYDROLOGIC MANAGEMENT UNITS
# ============================

hmu_map = generate_hmu_map(
    flow_acc=flow_acc,
    intervention_map=intervention_zones_class,
    min_acc_percentile=95,
    expansion_pixels=8
)

# ============================
# HMU REALES - MICROCUENCAS FUNCIONALES
# ============================

pour_points = create_pour_points(
    flow_acc=flow_acc,
    percentile=97,
    min_distance_pixels=60,
    max_points=50
)

hmu_watershed = delineate_hmu_watersheds(
    flow_dir=flow_dir,
    pour_points=pour_points
)

hmu_watershed = apply_basin_mask(
    hmu_watershed,
    dem_path,
    BASIN_PATH
)

hmu_watershed_output = OUTPUT_DIR / "hmu_watershed_map.tif"

save_raster(
    hmu_watershed_output,
    hmu_watershed,
    dem,
    dtype="int32",
    nodata=0
)

print("Mapa HMU watershed guardado en:", hmu_watershed_output)
print("Valores únicos flow_dir:", np.unique(flow_dir))
print("Valores únicos hmu_watershed:", np.unique(hmu_watershed))
print("Pixeles HMU watershed:", np.count_nonzero(hmu_watershed > 0))

hmu_watershed_gpkg = OUTPUT_DIR / "hmu_watershed.gpkg"
hmu_watershed_csv = OUTPUT_DIR / "hmu_watershed_summary.csv"

vectorize_hmu_watersheds(
    hmu_labels=hmu_watershed,
    intervention_map=intervention_zones_class,
    volume_map=volume,
    reference_raster_path=dem_path,
    output_gpkg=hmu_watershed_gpkg,
    output_csv=hmu_watershed_csv,
    min_area_ha=0.5
)

# ============================
# PMU - PLANNING MANAGEMENT UNITS
# ============================

tu_gpkg = OUTPUT_DIR / "territorial_units.gpkg"
hmu_watershed_gpkg = OUTPUT_DIR / "hmu_watershed.gpkg"

pmu_gpkg = OUTPUT_DIR / "pmu.gpkg"
pmu_csv = OUTPUT_DIR / "pmu_summary.csv"

if hmu_watershed_gpkg.exists():
    generate_pmu(
        tu_gpkg=tu_gpkg,
        hmu_gpkg=hmu_watershed_gpkg,
        output_gpkg=pmu_gpkg,
        output_csv=pmu_csv
    )
else:
    print("No se generó hmu_watershed.gpkg. Se omite generación de PMU.")