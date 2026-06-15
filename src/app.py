from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
import pandas as pd

import folium
from streamlit_folium import st_folium
import geopandas as gpd
PUBLIC_WEB_MODE = True
# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Gestión inteligente de agua lluvia",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
PMU_GPKG = OUTPUT_DIR / "pmu.gpkg"
PMU_CSV = OUTPUT_DIR / "pmu_summary.csv"
TU_GPKG = OUTPUT_DIR / "territorial_units.gpkg"

# =========================
# FUNCIONES AUXILIARES
# =========================

@st.cache_data
def read_raster(raster_path, mask_nodata=True):
    with rasterio.open(raster_path) as src:
        array = src.read(1).astype("float32")
        nodata = src.nodata

        if mask_nodata and nodata is not None:
            array[array == nodata] = np.nan

        return array

def get_raster_stats(array):
    return {
        "mínimo": float(np.nanmin(array)),
        "máximo": float(np.nanmax(array)),
        "media": float(np.nanmean(array)),
    }


def normalize_array(array):
    arr = array.copy()

    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.uint8)

    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)

    if arr_max == arr_min:
        out = np.zeros_like(arr, dtype=np.uint8)
        out[valid] = 180
        return out

    norm = (arr - arr_min) / (arr_max - arr_min)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)


def render_continuous(array):
    return normalize_array(array)


def render_classes(array, color_map):
    arr = np.nan_to_num(array, nan=0).astype("uint8")

    h, w = arr.shape

    # Fondo gris claro por defecto, no negro
    rgb = np.full((h, w, 3), 235, dtype=np.uint8)

    for value, color in color_map.items():
        rgb[arr == value] = color

    return rgb
def create_base_map(center_lat=0, center_lon=0, zoom_start=13):
    """
    Crea mapa base con Esri World Imagery.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri World Imagery",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)

    return m
def get_intervention_color(tipo):
    if tipo == "Conservación hídrica":
        return "#009600"
    elif tipo == "Infiltración / recarga":
        return "#0066ff"
    elif tipo == "Drenaje controlado":
        return "#ff8c00"
    elif tipo == "Retención / almacenamiento puntual":
        return "#dc0000"
    elif tipo == "Manejo urbano":
        return "#800080"
    else:
        return "#999999"

def load_tu(gpkg_path):
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    return gdf
@st.cache_data
def load_pmu(gpkg_path):
    gdf = gpd.read_file(gpkg_path)

    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf.geometry.simplify(
        tolerance=0.0001,
        preserve_topology=True
    )
    return gdf

def tu_color(tipo):
    colors = {
        "Conservación hídrica": "#2ca25f",
        "Recarga / infiltración": "#41b6c4",
        "Producción agrohidrológica": "#fed976",
        "Manejo urbano": "#756bb1",
        "Drenaje / regulación": "#fb6a4a",
    }
    return colors.get(tipo, "#cccccc")

def priority_color(priority):

    colors = {
        "Muy alta": "#d73027",
        "Alta": "#fc8d59",
        "Media": "#fee08b",
        "Baja": "#91cf60"
    }

    return colors.get(priority, "#808080")

def update_pmu_dynamic_values(pmu, rain_mm):
    """
    Actualiza volumen, prioridad y ranking PMU según escenario de lluvia.
    Usa el volumen base de pmu.gpkg como referencia.
    """

    pmu_dyn = pmu.copy()

    # Escenario base actual del modelo
    base_rain_mm = 25

    factor_lluvia = rain_mm / base_rain_mm

    pmu_dyn["volumen_m3_dyn"] = (
        pmu_dyn["volumen_m3"] * factor_lluvia
    )

    p50 = pmu_dyn["volumen_m3_dyn"].quantile(0.50)
    p70 = pmu_dyn["volumen_m3_dyn"].quantile(0.70)
    p90 = pmu_dyn["volumen_m3_dyn"].quantile(0.90)

    def classify_priority(vol):
        if vol >= p90:
            return "Muy alta"
        elif vol >= p70:
            return "Alta"
        elif vol >= p50:
            return "Media"
        else:
            return "Baja"

    pmu_dyn["prioridad_pmu_dyn"] = (
        pmu_dyn["volumen_m3_dyn"].apply(classify_priority)
    )

    priority_score = {
        "Muy alta": 4,
        "Alta": 3,
        "Media": 2,
        "Baja": 1
    }

    pmu_dyn["priority_score_dyn"] = (
        pmu_dyn["prioridad_pmu_dyn"].map(priority_score)
    )

    pmu_dyn["ranking_score_dyn"] = (
        pmu_dyn["priority_score_dyn"] * 1000
        + pmu_dyn["volumen_m3_dyn"]
    )

    pmu_dyn = pmu_dyn.sort_values(
        by="ranking_score_dyn",
        ascending=False
    )

    return pmu_dyn

def estimate_hydrologic_benefit(pmu_df):
    """
    Estima beneficio hidrológico preliminar por tipo de intervención.
    Usa volumen dinámico PMU y eficiencias referenciales.
    """

    pmu_benefit = pmu_df.copy()

    efficiency = {
        "Conservación/restauración hídrica": 0.15,
        "Zanjas de infiltración / recarga hídrica": 0.35,
        "Reservorio / almacenamiento temporal": 0.55,
        "Agricultura regenerativa, curvas a nivel y zanjas de infiltración": 0.25,
        "Drenaje controlado y estabilización de cauces": 0.20,
    }

    pmu_benefit["eficiencia_estimada"] = (
        pmu_benefit["accion_pmu"]
        .map(efficiency)
        .fillna(0.15)
    )

    pmu_benefit["volumen_gestionado_m3"] = (
        pmu_benefit["volumen_m3_dyn"]
        * pmu_benefit["eficiencia_estimada"]
    )

    total_volume = pmu_benefit["volumen_m3_dyn"].sum()
    managed_volume = pmu_benefit["volumen_gestionado_m3"].sum()

    if total_volume > 0:
        reduction_percent = managed_volume / total_volume * 100
    else:
        reduction_percent = 0

    return pmu_benefit, managed_volume, reduction_percent  

def compare_scenarios(pmu):

    escenarios = {
        "Moderado": 25,
        "Fuerte": 50,
        "Extremo": 100,
        "Crítico": 140
    }

    resultados = []

    for nombre, lluvia in escenarios.items():

        pmu_tmp = update_pmu_dynamic_values(
            pmu,
            lluvia
        )

        _, managed_volume, reduction_percent = (
            estimate_hydrologic_benefit(pmu_tmp)
        )

        resultados.append({
            "escenario": nombre,
            "lluvia_mm": lluvia,
            "volumen_pmu_m3":
                pmu_tmp["volumen_m3_dyn"].sum(),
            "volumen_gestionable_m3":
                managed_volume,
            "reduccion_pct":
                reduction_percent
        })

    return pd.DataFrame(resultados)

def estimate_cost_benefit(pmu_benefit):
    """
    Estima costo-beneficio preliminar por tipo de intervención.
    Valores referenciales para PMV.
    """

    pmu_cost = pmu_benefit.copy()

    # Costos referenciales
    # Conservación/agricultura: USD/ha
    # Zanjas/reservorios/drenaje: USD/m3 gestionado
    cost_rules = {
        "Conservación/restauración hídrica": {
            "type": "area",
            "cost": 800
        },
        "Agricultura regenerativa, curvas a nivel y zanjas de infiltración": {
            "type": "area",
            "cost": 1200
        },
        "Zanjas de infiltración / recarga hídrica": {
            "type": "volume",
            "cost": 2.5
        },
        "Reservorio / almacenamiento temporal": {
            "type": "volume",
            "cost": 6.0
        },
        "Drenaje controlado y estabilización de cauces": {
            "type": "volume",
            "cost": 4.0
        },
    }

    def calc_cost(row):
        rule = cost_rules.get(row["accion_pmu"], {"type": "area", "cost": 1000})

        if rule["type"] == "area":
            return row["area_ha"] * rule["cost"]
        else:
            return row["volumen_gestionado_m3"] * rule["cost"]

    pmu_cost["costo_estimado_usd"] = pmu_cost.apply(calc_cost, axis=1)

    pmu_cost["costo_por_m3_usd"] = (
        pmu_cost["costo_estimado_usd"] /
        pmu_cost["volumen_gestionado_m3"].replace(0, np.nan)
    )

    total_cost = pmu_cost["costo_estimado_usd"].sum()
    total_managed = pmu_cost["volumen_gestionado_m3"].sum()

    if total_managed > 0:
        cost_per_m3 = total_cost / total_managed
    else:
        cost_per_m3 = 0

    return pmu_cost, total_cost, cost_per_m3
# =========================
# FUNCIONES HIDROLÓGICAS
# =========================

def calculate_s_from_cn(cn_array):
    cn_array = cn_array.astype("float64")
    cn_array = np.where((cn_array <= 0) | (cn_array > 100), np.nan, cn_array)
    s_array = (25400.0 / cn_array) - 254.0
    return s_array


def calculate_ia(s_array, lambda_ia=0.2):
    return lambda_ia * s_array


def calculate_runoff(p_mm, s_array, ia_array):
    q_array = np.zeros_like(s_array, dtype="float64")

    condition = p_mm > ia_array
    q_array[condition] = ((p_mm - ia_array[condition]) ** 2) / (
        p_mm - ia_array[condition] + s_array[condition]
    )
    q_array[~condition] = 0.0

    return q_array


def minmax_normalize(array):
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)

    if arr_max == arr_min:
        return np.zeros_like(array, dtype="float64")

    return (array - arr_min) / (arr_max - arr_min)


def invert_and_normalize_slope(slope_array):
    slope_norm = minmax_normalize(slope_array)
    slope_inv = 1.0 - slope_norm
    return slope_inv


def log_normalize_flow_acc(flow_acc):
    flow_log = np.log1p(flow_acc)
    flow_norm = minmax_normalize(flow_log)
    return flow_norm


def calculate_runoff_risk(runoff_q, flow_acc, slope,
                          w_runoff=0.4, w_flow=0.4, w_slope=0.2):
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
    classes = np.zeros_like(risk_array, dtype="uint8")
    classes[(risk_array >= 0.00) & (risk_array < 0.25)] = 1
    classes[(risk_array >= 0.25) & (risk_array < 0.50)] = 2
    classes[(risk_array >= 0.50) & (risk_array < 0.75)] = 3
    classes[(risk_array >= 0.75)] = 4
    return classes


def generate_intervention_map(risk_classes):
    intervention = np.zeros_like(risk_classes, dtype="uint8")
    intervention[risk_classes == 1] = 1
    intervention[risk_classes == 2] = 2
    intervention[risk_classes == 3] = 3
    intervention[risk_classes == 4] = 4
    return intervention


# =========================
# CARGA DE BASES
# =========================

cn_path = OUTPUT_DIR / "cn_map.tif"
flow_acc_path = OUTPUT_DIR / "flow_acc.tif"
slope_path = OUTPUT_DIR / "slope.tif"
hotspots_path = OUTPUT_DIR / "hotspots.tif"
drainage_path = OUTPUT_DIR / "drainage_network.tif"
priority_path = OUTPUT_DIR / "priority_zones.tif"
volume_path = OUTPUT_DIR / "runoff_volume.tif"
smart_intervention_path = OUTPUT_DIR / "smart_intervention_map.tif"
intervention_zones_class_path = OUTPUT_DIR / "intervention_zones_class.tif"
zones_summary_path = OUTPUT_DIR / "intervention_zones_summary.csv"
zones_gpkg_path = OUTPUT_DIR / "intervention_zones.gpkg"
zones_gdf = gpd.read_file(zones_gpkg_path) if zones_gpkg_path.exists() else None
territorial_units_path = OUTPUT_DIR / "territorial_units.tif"

territorial_units = read_raster(
    territorial_units_path,
    mask_nodata=False
) if territorial_units_path.exists() else None

intervention_zones_class = read_raster(
    intervention_zones_class_path,
    mask_nodata=False
) if intervention_zones_class_path.exists() else None

required_files = {
    "CN": cn_path,
    "Flow accumulation": flow_acc_path,
    "Pendiente": slope_path,
    "Hotspots": hotspots_path,
    "Red de drenaje": drainage_path,
    "Zonas prioritarias": priority_path,
    "Volumen": volume_path,
}
missing = [name for name, path in required_files.items() if not path.exists()]

if missing:
    st.error(
        "Faltan rasters base para la versión dinámica del dashboard: "
        + ", ".join(missing)
    )
    st.stop()

cn_array = read_raster(cn_path)
flow_acc = read_raster(flow_acc_path)
slope = read_raster(slope_path)

hotspots = read_raster(hotspots_path, mask_nodata=False) if hotspots_path.exists() else None
drainage = read_raster(drainage_path, mask_nodata=False) if drainage_path.exists() else None
priority_zones = read_raster(priority_path, mask_nodata=False) if priority_path.exists() else None
runoff_volume = read_raster(volume_path) if volume_path.exists() else None
intervention_zones = read_raster(
    intervention_zones_class_path,
    mask_nodata=False
) if intervention_zones_class_path.exists() else None

intervention_zones_class = read_raster(
    intervention_zones_class_path,
    mask_nodata=False
) if intervention_zones_class_path.exists() else None
smart_intervention = read_raster(
    smart_intervention_path,
    mask_nodata=False
) if smart_intervention_path.exists() else None
zones_summary = pd.read_csv(zones_summary_path) if zones_summary_path.exists() else None

# =========================
# SIDEBAR
# =========================

st.sidebar.title("PMV")
st.sidebar.subheader("Gestión inteligente de agua lluvia")

scenario = st.sidebar.selectbox(
    "Escenario de lluvia",
    [
        "Moderado",
        "Fuerte",
        "Extremo",
        "Crítico",
        "Personalizado"
    ]
)
if scenario == "Moderado":
    p_event = 25
    lambda_ia = 0.20

elif scenario == "Fuerte":
    p_event = 50
    lambda_ia = 0.20

elif scenario == "Extremo":
    p_event = 100
    lambda_ia = 0.10

elif scenario == "Crítico":
    p_event = 140
    lambda_ia = 0.05

elif scenario == "Personalizado":
    p_event = st.sidebar.number_input(
        "Lluvia evento personalizada (mm)",
        min_value=1.0,
        max_value=300.0,
        value=25.0,
        step=1.0
    )
    lambda_ia = st.sidebar.number_input(
        "Lambda Ia personalizada",
        min_value=0.0,
        max_value=0.5,
        value=0.20,
        step=0.01
    )
rain_mm = p_event

st.sidebar.markdown("### Parámetros")
st.sidebar.write(f"**Escenario:** {scenario}")
st.sidebar.write(f"**Lluvia evento:** {p_event} mm")
st.sidebar.write(f"**Lambda Ia:** {lambda_ia:.2f}")

selected_layer = st.sidebar.selectbox(
    "Selecciona la capa a visualizar",
    [
    "Mapa CN",
    "Pendiente",
    "Acumulación de flujo",
    "Escorrentía dinámica",
    "Riesgo continuo dinámico",
    "Clases de riesgo dinámicas",
    "Mapa de intervención dinámico",
    "Localización prioritaria",
    "Volumen preliminar",
    "Intervención sugerida",
    "Hotspots",
    "Red de drenaje",
    "Zonas de intervención",
    "Unidades Territoriales"
]
)

# =========================
# CÁLCULO DINÁMICO
# =========================

s_array = calculate_s_from_cn(cn_array)
ia_array = calculate_ia(s_array, lambda_ia=lambda_ia)
runoff_q = calculate_runoff(p_event, s_array, ia_array)

risk_array = calculate_runoff_risk(
    runoff_q=runoff_q,
    flow_acc=flow_acc,
    slope=slope,
    w_runoff=0.4,
    w_flow=0.4,
    w_slope=0.2
)

risk_classes = classify_risk(risk_array)
intervention = generate_intervention_map(risk_classes)

# =========================
# CAPAS DISPONIBLES
# =========================

categorical_colors = {
    "Mapa CN": {
        60: (102, 194, 165),
        70: (252, 141, 98),
        80: (228, 26, 28),
    },

    "Clases de riesgo dinámicas": {
        1: (102, 194, 165),
        2: (255, 255, 153),
        3: (253, 174, 97),
        4: (215, 25, 28),
    },

    "Mapa de intervención dinámico": {
        1: (0, 150, 0),
        2: (230, 230, 0),
        3: (230, 160, 40),
        4: (220, 0, 0),
    },

    "Localización prioritaria": {
        0: (240, 240, 240),
        1: (180, 0, 0),
    },

    "Intervención sugerida": {
    0: (235, 235, 235),
    1: (0, 150, 0),
    2: (0, 102, 255),
    3: (255, 140, 0),
    4: (220, 0, 0),
    5: (128, 0, 128),
    },

    "Hotspots": {
        0: (240, 240, 240),
        1: (220, 0, 0),
    },

    "Red de drenaje": {
        0: (240, 240, 240),
        1: (0, 90, 255),
    },
    "Zonas de intervención": {
    0: (235, 235, 235),   # sin intervención
    1: (0, 150, 0),       # verde conservación
    2: (0, 102, 255),     # azul infiltración
    3: (255, 140, 0),     # naranja drenaje
    4: (220, 0, 0),       # rojo retención
    5: (128, 0, 128),     # morado urbano
    },
    "Unidades Territoriales": {
    0: (235, 235, 235),
    1: (44, 162, 95),     # Conservación hídrica
    2: (65, 182, 196),    # Recarga / infiltración
    3: (254, 217, 118),   # Producción agrohidrológica
    4: (117, 107, 177),   # Manejo urbano
    5: (251, 106, 74),    # Drenaje / regulación
    },
}

layer_arrays = {
    "Mapa CN": cn_array,
    "Pendiente": slope,
    "Acumulación de flujo": flow_acc,
    "Escorrentía dinámica": runoff_q,
    "Riesgo continuo dinámico": risk_array,
    "Clases de riesgo dinámicas": risk_classes,
    "Mapa de intervención dinámico": intervention,
    "Localización prioritaria": priority_zones,
    "Volumen preliminar": runoff_volume,
    "Hotspots": hotspots,
    "Intervención sugerida": smart_intervention,
    "Red de drenaje": drainage,
    "Zonas de intervención": intervention_zones_class,
    "Unidades Territoriales": territorial_units,
 }

layer_desc = {
    "Mapa CN": "Distribución espacial simplificada del Curve Number.",
    "Pendiente": "Pendiente del terreno en grados.",
    "Acumulación de flujo": "Número relativo de celdas que aportan flujo a cada punto.",
    "Escorrentía dinámica": "Escorrentía recalculada en tiempo real según la lluvia del evento.",
    "Riesgo continuo dinámico": "Índice continuo de riesgo de escorrentía.",
    "Clases de riesgo dinámicas": "Clasificación discreta del riesgo en 4 clases.",
    "Mapa de intervención dinámico": "Traducción del riesgo a acciones de intervención.",
    "Hotspots": "Zonas críticas por acumulación alta y pendiente baja.",
    "Red de drenaje": "Rutas naturales de concentración de flujo.",
    "Localización prioritaria": "Zonas donde se recomienda priorizar acciones de manejo.",
    "Intervención sugerida": "Clasificación automática del tipo de intervención: infiltración, drenaje o almacenamiento.",
    "Volumen preliminar": "Volumen preliminar de escorrentía estimado por celda en m³.",
    "Zonas de intervención": "Zonas agrupadas de intervención, clasificadas por tipo dominante.",
    "Unidades Territoriales": "Clasificación funcional del 100% de la cuenca: conservación, recarga, producción, manejo urbano y drenaje/regulación.",
}

# =========================
# PANEL PRINCIPAL
# =========================

st.title("Sistema de apoyo a la planificación territorial y manejo de aguas pluviales")
st.caption("Dashboard interactivo con lluvia dinámica para análisis hidrológico espacial.")

col1, col2 = st.columns([2, 1])

current_array = layer_arrays[selected_layer]

with col1:
    st.subheader(selected_layer)

    if current_array is None:
        st.warning("La capa seleccionada no está disponible.")
    else:
        if selected_layer in categorical_colors:
            img = render_classes(current_array, categorical_colors[selected_layer])
        else:
            img = render_continuous(current_array)
        
        st.write("Valores únicos de la capa:", np.unique(current_array[~np.isnan(current_array)])[:20])

        st.image(img, caption=selected_layer, use_container_width=True)

with col2:
    st.subheader("Resumen")
    st.write(layer_desc[selected_layer])

    if current_array is not None:
        stats = get_raster_stats(current_array)
        st.markdown("### Estadísticas")
        st.write(f"**Mínimo:** {stats['mínimo']:.3f}")
        st.write(f"**Máximo:** {stats['máximo']:.3f}")
        st.write(f"**Media:** {stats['media']:.3f}")
    if runoff_volume is not None and hotspots is not None and priority_zones is not None:
        total_volume = np.nansum(runoff_volume)
        hotspot_volume = np.nansum(runoff_volume[hotspots == 1])
        priority_volume = np.nansum(runoff_volume[priority_zones == 1])

        st.markdown("### Volumen preliminar")
        st.write(f"**Volumen total:** {total_volume:,.2f} m³")
        st.write(f"**Volumen en hotspots:** {hotspot_volume:,.2f} m³")
        st.write(f"**Volumen en zonas prioritarias:** {priority_volume:,.2f} m³")
    st.markdown("### Interpretación")
    if selected_layer == "Escorrentía dinámica":
        st.info("A mayor lluvia, mayor escorrentía estimada.")
    elif selected_layer == "Riesgo continuo dinámico":
        st.info("Integra escorrentía, acumulación de flujo y pendiente.")
    elif selected_layer == "Mapa de intervención dinámico":
        st.info("Permite priorizar decisiones según el escenario de lluvia.")
    elif selected_layer == "Clases de riesgo dinámicas":
        st.info("Resume el riesgo en 4 niveles de fácil lectura.")
    if selected_layer == "Pendiente":
        st.info("Las pendientes bajas favorecen concentración y anegamiento.")
    elif selected_layer == "Acumulación de flujo":
        st.info("Valores altos indican mayor convergencia del flujo superficial.")
    elif selected_layer == "Escorrentía dinámica":
        st.info("A mayor lluvia, mayor escorrentía estimada.")
    elif selected_layer == "Riesgo continuo dinámico":
        st.info("Integra escorrentía, acumulación de flujo y pendiente.")
    elif selected_layer == "Mapa de intervención dinámico":
        st.info("Permite priorizar decisiones según el escenario de lluvia.")
    elif selected_layer == "Clases de riesgo dinámicas":
        st.info("Resume el riesgo en 4 niveles de fácil lectura.")
    elif selected_layer == "Localización prioritaria":
        st.info("Marca las zonas donde se debe priorizar la intervención.")
    elif selected_layer == "Volumen preliminar":
        st.info("Representa el volumen estimado de escorrentía por celda en m³.")
    elif selected_layer == "Intervención sugerida":
        st.info("Azul = infiltración, naranja = drenaje controlado, rojo = almacenamiento o retención.")
    elif selected_layer == "Zonas de intervención":
        st.info("Mapa agrupado por zonas: azul = infiltración, naranja = drenaje, rojo = almacenamiento.")
    if selected_layer in ["Intervención sugerida", "Zonas de intervención"]:
        st.markdown("### Leyenda de intervención")
    if selected_layer == "Unidades Territoriales":
        st.markdown("### Unidades Territoriales")

        st.markdown(
        """
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div><span style="background-color:#2ca25f; padding:6px 14px; border-radius:4px;"></span> Conservación hídrica</div>
            <div><span style="background-color:#41b6c4; padding:6px 14px; border-radius:4px;"></span> Recarga / infiltración</div>
            <div><span style="background-color:#fed976; padding:6px 14px; border-radius:4px;"></span> Producción agrohidrológica</div>
            <div><span style="background-color:#756bb1; padding:6px 14px; border-radius:4px;"></span> Manejo urbano</div>
            <div><span style="background-color:#fb6a4a; padding:6px 14px; border-radius:4px;"></span> Drenaje / regulación</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def generate_executive_summary(zones_summary, rainfall_mm, lambda_ia,
                               total_volume, hotspot_volume, priority_volume):

    total_zones = len(zones_summary)
    total_area_ha = zones_summary["area_ha"].sum()

    dominant_type = (
        zones_summary.groupby("tipo_intervencion")["area_ha"]
        .sum()
        .idxmax()
    )

    high_priority_zones = zones_summary[
        zones_summary["prioridad"].isin(["Muy alta", "Alta"])
    ]

    n_high_priority = len(high_priority_zones)
    high_priority_volume = high_priority_zones["volumen_m3"].sum()

    summary = f"""
    Para un evento de lluvia de **{rainfall_mm} mm** y un valor de **λ = {lambda_ia:.2f}**, 
    el PMV identificó **{total_zones} zonas de intervención**, con una superficie aproximada 
    de **{total_area_ha:,.2f} ha**.

    El volumen preliminar total estimado fue de **{total_volume:,.2f} m³**, 
    de los cuales **{hotspot_volume:,.2f} m³** se concentran en hotspots y 
    **{priority_volume:,.2f} m³** en zonas prioritarias.

    La intervención dominante corresponde a **{dominant_type}**, lo que indica el tipo de 
    solución más relevante para el escenario analizado.

    Se identificaron **{n_high_priority} zonas de prioridad alta o muy alta**, que concentran 
    aproximadamente **{high_priority_volume:,.2f} m³**. Estas zonas deberían considerarse como 
    primera fase de intervención dentro de una estrategia de manejo inteligente de aguas lluvia.
    """

    return summary
if zones_summary is not None:
    executive_summary = generate_executive_summary(
        zones_summary,
        rainfall_mm=p_event,
        lambda_ia=lambda_ia,
        total_volume=total_volume,
        hotspot_volume=hotspot_volume,
        priority_volume=priority_volume
    )

st.markdown("---")
st.markdown(
    "**Estado del PMV:** demo funcional con escenario dinámico de lluvia para manejo inteligente de aguas lluvia."
)

map_center_lat = st.sidebar.number_input(
    "Centro latitud",
    value=-0.2,
    format="%.6f",
    key="map_center_lat"
)

map_center_lon = st.sidebar.number_input(
    "Centro longitud",
    value=-78.3,
    format="%.6f",
    key="map_center_lon"
)

zoom_level = st.sidebar.slider(
    "Zoom mapa base",
    5,
    18,
    10,
    key="zoom_base"
)

m = create_base_map(
    center_lat=map_center_lat,
    center_lon=map_center_lon,
    zoom_start=zoom_level
)

if zones_gdf is not None:
    zones_gdf_map = zones_gdf.to_crs(epsg=4326)

    for _, row in zones_gdf_map.iterrows():
        color = get_intervention_color(row["tipo_intervencion"])

        popup_text = f"""
        <b>Zona:</b> {row['zone_id']}<br>
        <b>Intervención:</b> {row['tipo_intervencion']}<br>
        <b>Área:</b> {row['area_ha']:.2f} ha<br>
        <b>Volumen:</b> {row['volumen_m3']:.2f} m³<br>
        <b>Prioridad:</b> {row['prioridad']}<br>
"""

        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": color,
                "weight": 1,
                "fillOpacity": 0.45,
            },
            popup=folium.Popup(popup_text, max_width=300),
            name=f"Zona {row['zone_id']}"
        ).add_to(m)
else:
    st.warning("No se encontró intervention_zones.gpkg para superponer.")

if TU_GPKG.exists():
    tu = load_tu(TU_GPKG)
    folium.GeoJson(
        tu,
        #name="Territorial Units",
        style_function=lambda feature: {
            "fillColor": tu_color(feature["properties"].get("tipo_tu")),
            "color": "#555555",
            "weight": 0.2,
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tipo_tu", "area_ha"],
            aliases=["Tipo TU", "Área (ha)"],
            localize=True  
        )
        ).add_to(m)

if "executive_summary" not in locals():
    executive_summary = (
        f"Para el escenario **{scenario}** ({p_event} mm), "
        "no se encontraron zonas de intervención vectoriales para esta corrida. "
        "El análisis continúa con Territorial Units, HMU watershed y PMU."
    )

st.markdown("### Resumen ejecutivo automático")
st.markdown(executive_summary)


# =========================
# PMU - VISOR TERRITORIAL PRINCIPAL
# =========================

st.markdown("## Visor de Unidades Prioritarias de Planificación y Gestión (PMU)")
st.caption(
    "Unidades Territoriales como contexto territorial y Unidades Prioritarias de Planificación y Gestión "
    "como zonas priorizadas de intervención."
)

if not PMU_GPKG.exists():
    st.warning("No se encontró pmu.gpkg. Ejecuta primero main.py.")
else:
    pmu = load_pmu(PMU_GPKG)
    pmu = update_pmu_dynamic_values(
    pmu,
    rain_mm
    )

priority_score = {
    "Muy alta": 4,
    "Alta": 3,
    "Media": 2,
    "Baja": 1
}

pmu["priority_score"] = pmu["prioridad_pmu_dyn"].map(priority_score)

pmu["ranking_score"] = (
    pmu["priority_score"] * 1000
    + pmu["volumen_m3_dyn"]
)

pmu_ranked = pmu.sort_values(
    by="ranking_score",
    ascending=False
)

top10_pmu = pmu.head(10)
# =========================
# RANKING PMU
# =========================

st.markdown("## Ranking de Unidades Prioritarias de Planificación y Gestión")

if not top10_pmu.empty:
    pmu_1 = top10_pmu.iloc[0]

    col_a, col_b, col_c = st.columns(3)

    col_a.metric(
        "PMU más prioritaria",
        f"PMU {pmu_1['pmu_id']}",
        pmu_1["prioridad_pmu"]
    )

    col_b.metric(
        "Volumen asociado",
        f"{pmu_1['volumen_m3']:,.0f} m³"
    )

    col_c.metric(
        "Área",
        f"{pmu_1['area_ha']:,.2f} ha"
    )

    ranking_view = top10_pmu[
        [
            "pmu_id",
            "prioridad_pmu_dyn",
            "accion_pmu",
            "volumen_m3_dyn",
            "area_ha",
            "tipo_tu",
            "tipo_hmu",
        ]
    ].copy()

    ranking_view.insert(
        0,
        "ranking",
        range(1, len(ranking_view) + 1)
    )

    st.dataframe(
        ranking_view,
        use_container_width=True
    )
else:
    st.warning("No existen PMU para generar ranking.")


# =========================
# PMU - VISOR TERRITORIAL PRINCIPAL
# =========================

st.markdown("## Visor territorial de Unidades Prioritarias de Planificación y Gestión")
st.caption(
    "Unidades Territoriales como contexto territorial y Unidades Prioritarias de Planificación y Gestión "
    "como zonas priorizadas de intervención."
)

priority_score = {
    "Muy alta": 4,
    "Alta": 3,
    "Media": 2,
    "Baja": 1
}

pmu["priority_score"] = pmu["prioridad_pmu_dyn"].map(priority_score)

pmu["ranking_score"] = (
    pmu["priority_score"] * 1000
    + pmu["volumen_m3_dyn"]
)

pmu_ranked = pmu.sort_values(
    by="ranking_score",
    ascending=False
)

top10_pmu = pmu_ranked.head(10)
    
    # =========================
    # FILTROS PMU
    # =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
        prioridades = sorted(pmu["prioridad_pmu_dyn"].dropna().unique())
        prioridad_sel = st.multiselect(
            "Prioridad",
            prioridades,
            default=prioridades
        )

with col2:
        acciones = sorted(pmu["accion_pmu"].dropna().unique())
        accion_sel = st.multiselect(
            "Acción recomendada",
            acciones,
            default=acciones
        )

with col3:
        min_vol = float(pmu["volumen_m3_dyn"].min())
        max_vol = float(pmu["volumen_m3_dyn"].max())

        vol_range = st.slider(
            "Volumen (m³)",
            min_value=min_vol,
            max_value=max_vol,
            value=(min_vol, max_vol)
        )

with col4:
        min_area = float(pmu["area_ha"].min())
        max_area = float(pmu["area_ha"].max())

        area_range = st.slider(
            "Área (ha)",
            min_value=min_area,
            max_value=max_area,
            value=(min_area, max_area)
        )

pmu_filtrada = pmu[
    (pmu["prioridad_pmu_dyn"].isin(prioridad_sel)) &
    (pmu["accion_pmu"].isin(accion_sel)) &
    (pmu["volumen_m3_dyn"].between(vol_range[0], vol_range[1])) &
    (pmu["area_ha"].between(area_range[0], area_range[1]))
].copy()
pmu_benefit, managed_volume, reduction_percent = estimate_hydrologic_benefit(
    pmu_filtrada
)
pmu_cost, total_cost, cost_per_m3 = estimate_cost_benefit(
    pmu_benefit
)
# =========================
# MÉTRICAS
# =========================

c1, c2, c3, c4 = st.columns(4)

c1.metric("PMU filtradas", len(pmu_filtrada))
c2.metric("Volumen total PMU", f"{pmu_filtrada['volumen_m3'].sum():,.0f} m³")
c3.metric("Área total PMU", f"{pmu_filtrada['area_ha'].sum():,.2f} ha")

if not pmu_filtrada.empty:
    c4.metric(
        "Prioridad dominante",
        pmu_filtrada["prioridad_pmu_dyn"].mode().iloc[0]
    )
else:
    c4.metric("Prioridad dominante", "Sin datos")

# =========================
# MAPA
# =========================

if not pmu_filtrada.empty:
    bounds = pmu.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap"
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri World Imagery",
        overlay=False,
        control=True
    ).add_to(m)

    if TU_GPKG.exists():
        tu = load_tu(TU_GPKG)

        folium.GeoJson(
            tu,
        name="Territorial Units",
        show=False,
        style_function=lambda feature: {
        "fillColor": tu_color(
            feature["properties"].get("tipo_tu")
        ),
        "color": "#555555",
        "weight": 0.2,
        "fillOpacity": 0.25,
        },
        tooltip=folium.GeoJsonTooltip(
        fields=["tipo_tu", "area_ha"],
        aliases=["Tipo TU", "Área (ha)"],
        localize=True
        )
            ).add_to(m)

    if zones_gdf is not None:
        zones_gdf_map = zones_gdf.to_crs(epsg=4326)

        for _, row in zones_gdf_map.iterrows():
            color = get_intervention_color(row["tipo_intervencion"])

            popup_text = f"""
            <b>Zona:</b> {row['zone_id']}<br>
            <b>Intervención:</b> {row['tipo_intervencion']}<br>
            <b>Área:</b> {row['area_ha']:.2f} ha<br>
            <b>Volumen:</b> {row['volumen_m3']:.2f} m³<br>
            <b>Prioridad:</b> {row['prioridad']}
            """

            folium.GeoJson(
                row.geometry,
                #name="Zonas de intervención",
                show=False,
                style_function=lambda feature, color=color: {
                    "fillColor": color,
                    "color": color,
                    "weight": 1,
                    "fillOpacity": 0.35,
                },
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

    pmu_mapa = pmu_filtrada.copy()

    if len(pmu_mapa) > 10:
        pmu_mapa = pmu_mapa.sort_values(
            by="volumen_m3",
            ascending=False
        ).head(10)

    pmu_mapa["geometry"] = pmu_mapa["geometry"].simplify(
    0.00002,
    preserve_topology=True
    )

    st.caption(f"Mostrando {len(pmu_mapa)} PMU en el mapa.")

    for _, row in pmu_mapa.iterrows():
        popup = f"""
        <b>PMU:</b> {row['pmu_id']}<br>
        <b>Prioridad:</b> {row["prioridad_pmu_dyn"]}<br>
        <b>Acción:</b> {row['accion_pmu']}<br>
        <b>Tipo TU:</b> {row['tipo_tu']}<br>
        <b>Tipo HMU:</b> {row['tipo_hmu']}<br>
        <b>Área:</b> {row['area_ha']:.2f} ha<br>
        <b>Volumen:</b> {row["volumen_m3_dyn"]:.2f} m³
        """

        folium.GeoJson(
            row.geometry,
            name="PMU",
            style_function=lambda feature, color=priority_color(row["prioridad_pmu_dyn"]): {
                "fillColor": color,
                "color": "#111111",
                "weight": 1.4,
                "fillOpacity": 0.3,
            },
            popup=folium.Popup(popup, max_width=350)
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(
        m,
        width=850,
        height=500,
        key="visor_pmu_principal"
    )

else:
    st.warning("No existen PMU con los filtros seleccionados.")

st.markdown("### Leyenda de Unidades Prioritarias de Planificación y Gestión")

st.markdown(
    """
    <div style="display:flex; gap:18px; flex-wrap:wrap; align-items:center;">
        <div><span style="background-color:#2ca25f; padding:6px 16px; border-radius:4px;"></span> Baja</div>
        <div><span style="background-color:#fdd049; padding:6px 16px; border-radius:4px;"></span> Media</div>
        <div><span style="background-color:#fdae61; padding:6px 16px; border-radius:4px;"></span> Alta</div>
        <div><span style="background-color:#d73027; padding:6px 16px; border-radius:4px;"></span> Muy alta</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# BENEFICIO HIDROLÓGICO ESTIMADO
# =========================

st.markdown("## Beneficio hidrológico estimado")

b1, b2, b3 = st.columns(3)

b1.metric(
    "Volumen PMU escenario",
    f"{pmu_filtrada['volumen_m3_dyn'].sum():,.0f} m³"
)

b2.metric(
    "Volumen gestionable estimado",
    f"{managed_volume:,.0f} m³"
)

b3.metric(
    "Reducción potencial",
    f"{reduction_percent:.1f} %"
)

st.info(
    "El beneficio hidrológico se estima aplicando eficiencias referenciales "
    "según el tipo de intervención propuesta para cada Unidades Prioritarias de Planificación y Gestión (PMU). "
    "Este cálculo es preliminar y sirve para comparación entre escenarios."
    "Versión web optimizada: se muestran las 10 unidades prioritarias principales para mejorar el rendimiento en la nube."
)
benefit_by_action = (
    pmu_benefit
    .groupby("accion_pmu")
    .agg(
        pmu_count=("pmu_id", "count"),
        volumen_total_m3=("volumen_m3_dyn", "sum"),
        volumen_gestionado_m3=("volumen_gestionado_m3", "sum"),
        eficiencia_promedio=("eficiencia_estimada", "mean"),
        area_total_ha=("area_ha", "sum"),
    )
    .reset_index()
)

st.markdown("## Comparación de escenarios")
st.markdown("### Beneficio por tipo de intervención")
scenario_df = compare_scenarios(pmu)

st.dataframe(
    scenario_df,
    use_container_width=True
)
import plotly.express as px
fig = px.bar(
    scenario_df,
    x="escenario",
    y="volumen_gestionable_m3",
    color="escenario",
    title="Volumen gestionable por escenario"
)

if not PUBLIC_WEB_MODE:
    # aquí va la sección pesada
    st.plotly_chart(
    fig,
    use_container_width=True
    )
    st.dataframe(
    benefit_by_action,
    use_container_width=True
    )
    dominant_action = (
    benefit_by_action
    .sort_values("volumen_gestionado_m3", ascending=False)
    .iloc[0]
)

st.success(
    f"Para el escenario **{scenario}** ({rain_mm} mm), "
    f"el sistema estima que las Unidades Prioritarias de Planificación y Gestión (PMU) seleccionadas podrían gestionar aproximadamente "
    f"**{managed_volume:,.0f} m³** de escorrentía, equivalente a una reducción potencial "
    f"del **{reduction_percent:.1f}%** sobre el volumen PMU analizado. "
    f"La acción con mayor aporte estimado es **{dominant_action['accion_pmu']}**."
)
# =========================
# INDICADOR COSTO-BENEFICIO
# =========================

st.markdown("## Indicador económico costo-beneficio")

e1, e2, e3 = st.columns(3)

e1.metric(
    "Costo estimado",
    f"${total_cost:,.0f}"
)

e2.metric(
    "Costo por m³ gestionado",
    f"${cost_per_m3:,.2f}/m³"
)

e3.metric(
    "Volumen gestionado",
    f"{managed_volume:,.0f} m³"
)

st.info(
    "Los costos son referenciales para el PMV. "
    "Deben ajustarse con precios locales, diseño definitivo, mano de obra, "
    "materiales y condiciones de sitio."
)

cost_by_action = (
    pmu_cost
    .groupby("accion_pmu")
    .agg(
        pmu_count=("pmu_id", "count"),
        area_total_ha=("area_ha", "sum"),
        volumen_gestionado_m3=("volumen_gestionado_m3", "sum"),
        costo_estimado_usd=("costo_estimado_usd", "sum"),
        costo_promedio_m3=("costo_por_m3_usd", "mean"),
    )
    .reset_index()
)

st.markdown("### Costo-beneficio por tipo de intervención")

st.dataframe(
    cost_by_action,
    use_container_width=True
)

best_action = (
    cost_by_action
    .sort_values("costo_promedio_m3", ascending=True)
    .iloc[0]
)

st.success(
    f"Para el escenario **{scenario}**, el costo total estimado de intervención "
    f"es de aproximadamente **${total_cost:,.0f}**, con un costo medio de "
    f"**${cost_per_m3:,.2f}/m³ gestionado**. "
    f"La intervención con mejor relación costo-beneficio preliminar es "
    f"**{best_action['accion_pmu']}**."
)

# =========================
# TABLA PMU
# =========================

st.markdown("### Tabla Unidades Prioritarias de Planificación y Gestión (PMU)")

st.dataframe(
    pmu_filtrada[
        [
            "pmu_id",
            "tipo_tu",
            "tipo_hmu",
            "area_ha",
            "volumen_m3_dyn",
            "prioridad_pmu_dyn",
            "accion_pmu",
        ]
    ].rename(columns={
        "volumen_m3_dyn": "volumen_escenario_m3",
        "prioridad_pmu_dyn": "prioridad_escenario"
    }),
    use_container_width=True
)

st.markdown("### Tabla de zonas de intervención")

if zones_summary is not None:

    
    priority_order = {
        "Muy alta": 1,
        "Alta": 2,
        "Media": 3,
        "Baja": 4
    }

    zones_summary["orden_prioridad"] = (
        zones_summary["prioridad"].map(priority_order)
    )

    zones_summary_view = zones_summary.sort_values(
        by=["orden_prioridad", "volumen_m3"],
        ascending=[True, False]
    )

    st.dataframe(
        zones_summary_view[[
            "zone_id",
            "tipo_intervencion",
            "area_ha",
            "volumen_m3",
            "prioridad"
        ]],
        use_container_width=True
    )
# =========================
# DESCARGA PMU FILTRADAS
# =========================

st.markdown("### Descargar Unidades Prioritarias de Planificación y Gestión filtradas")

pmu_export = pmu_filtrada.copy()

# CSV sin geometría
csv_data = (
    pmu_export
    .drop(columns="geometry")
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Descargar PMU filtradas CSV",
    data=csv_data,
    file_name=f"pmu_filtradas_{scenario}.csv",
    mime="text/csv"
)

# GeoJSON con geometría
pmu_geojson = pmu_export.to_crs(epsg=4326).to_json()

st.download_button(
    label="Descargar PMU filtradas GeoJSON",
    data=pmu_geojson,
    file_name=f"pmu_filtradas_{scenario}.geojson",
    mime="application/geo+json"
)
# =========================
# GLOSARIO
# =========================

st.markdown("## Glosario")

with st.expander("Ver glosario de términos del PMV"):
    st.markdown("""
### Unidades Territoriales (TU)
Sectores del territorio clasificados según cobertura, pendiente y comportamiento hidrológico general.

### Unidades de Gestión Hidrológica (HMU)
Áreas que comparten una dinámica similar de drenaje, acumulación de flujo y respuesta frente a la lluvia.

### Unidades Prioritarias de Planificación y Gestión (PMU)
Zonas priorizadas donde se recomienda implementar acciones de manejo de aguas lluvia.

### Hotspots - Zonas Críticas de Escorrentía
Áreas donde el modelo identifica mayor concentración de escorrentía, acumulación de flujo o posible anegamiento. Son puntos clave para priorizar intervención.

### Beneficio Hidrológico Estimado
Volumen de agua que podría ser gestionado mediante las intervenciones propuestas.

### Escenario de lluvia
Condición simulada de precipitación. En el PMV se usan escenarios moderado, fuerte, extremo, crítico y personalizado.

### Costo-beneficio
Relación entre el costo estimado de implementar una intervención y el volumen de agua que podría gestionarse.

""")