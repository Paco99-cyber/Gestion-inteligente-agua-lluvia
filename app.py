from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
import pandas as pd

import folium
from streamlit_folium import st_folium
import geopandas as gpd
# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Gestión inteligente de agua lluvia",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
st.write("Ruta outputs:", OUTPUT_DIR)
st.write("Archivos encontrados:", [p.name for p in OUTPUT_DIR.glob("*.tif")])


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
    "Escenario hidrológico",
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

else:
    p_event = st.sidebar.slider(
        "Lluvia del evento (mm)",
        min_value=10,
        max_value=150,
        value=50
    )

    lambda_ia = st.sidebar.slider(
        "Lambda Ia",
        min_value=0.05,
        max_value=0.30,
        value=0.20
    )
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
]
)

st.sidebar.markdown("### Parámetros")
st.sidebar.write(f"**Lluvia evento:** {p_event} mm")
st.sidebar.write(f"**Lambda Ia:** {lambda_ia:.2f}")

st.sidebar.markdown("### Ruta de trabajo")
st.sidebar.write(str(OUTPUT_DIR))


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
}

# =========================
# PANEL PRINCIPAL
# =========================

st.title("Dashboard demo - Manejo inteligente de aguas lluvia")
st.caption("PMV interactivo con lluvia dinámica para análisis hidrológico espacial.")

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

    st.markdown(
    """
    <div style="display:flex; flex-direction:column; gap:8px;">
        <div><span style="background-color:#009600; padding:6px 14px; border-radius:4px;"></span> Conservación hídrica</div>
        <div><span style="background-color:#0066ff; padding:6px 14px; border-radius:4px;"></span> Infiltración / recarga</div>
        <div><span style="background-color:#ff8c00; padding:6px 14px; border-radius:4px;"></span> Drenaje controlado</div>
        <div><span style="background-color:#dc0000; padding:6px 14px; border-radius:4px;"></span> Retención / almacenamiento puntual</div>
        <div><span style="background-color:#800080; padding:6px 14px; border-radius:4px;"></span> Manejo urbano</div>
    </div>
    """,
    unsafe_allow_html=True
 )

st.markdown("---")
st.markdown(
    "**Estado del PMV:** demo funcional con escenario dinámico de lluvia para manejo inteligente de aguas lluvia."
)
st.markdown("### Tabla de zonas de intervención")

if zones_summary is not None:

    st.markdown("### Tabla de zonas de intervención")

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

zones_gdf = gpd.read_file(zones_gpkg_path) if zones_gpkg_path.exists() else None

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
st.markdown("### Resumen ejecutivo automático")
st.markdown(executive_summary)

# =========================
# PMU - VISOR TERRITORIAL PRINCIPAL
# =========================

st.markdown("## Visor territorial PMU")
st.caption(
    "Territorial Units como contexto territorial y Planning Management Units "
    "como zonas priorizadas de intervención."
)

if not PMU_GPKG.exists():
    st.warning("No se encontró pmu.gpkg. Ejecuta primero main.py.")
else:
    pmu = load_pmu(PMU_GPKG)

    # =========================
    # FILTROS PMU
    # =========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        prioridades = sorted(pmu["prioridad_pmu"].dropna().unique())
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
        min_vol = float(pmu["volumen_m3"].min())
        max_vol = float(pmu["volumen_m3"].max())

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
        (pmu["prioridad_pmu"].isin(prioridad_sel)) &
        (pmu["accion_pmu"].isin(accion_sel)) &
        (pmu["volumen_m3"].between(vol_range[0], vol_range[1])) &
        (pmu["area_ha"].between(area_range[0], area_range[1]))
    ].copy()

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
            pmu_filtrada["prioridad_pmu"].mode().iloc[0]
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

        # =========================
        # TERRITORIAL UNITS
        # =========================

        if TU_GPKG.exists():
            tu = load_tu(TU_GPKG)

            folium.GeoJson(
                tu,
                name="Territorial Units",
                style_function=lambda feature: {
                    "fillColor": tu_color(feature["properties"].get("tipo_tu")),
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

        # =========================
        # ZONAS DE INTERVENCIÓN
        # =========================

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
                    name="Zonas de intervención",
                    style_function=lambda feature, color=color: {
                        "fillColor": color,
                        "color": color,
                        "weight": 1,
                        "fillOpacity": 0.35,
                    },
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m)

        # =========================
        # PMU FILTRADAS
        # =========================

        for _, row in pmu_filtrada.iterrows():
            popup = f"""
            <b>PMU:</b> {row['pmu_id']}<br>
            <b>Prioridad:</b> {row['prioridad_pmu']}<br>
            <b>Acción:</b> {row['accion_pmu']}<br>
            <b>Tipo TU:</b> {row['tipo_tu']}<br>
            <b>Tipo HMU:</b> {row['tipo_hmu']}<br>
            <b>Área:</b> {row['area_ha']:.2f} ha<br>
            <b>Volumen:</b> {row['volumen_m3']:.2f} m³
            """

            folium.GeoJson(
                row.geometry,
                name="PMU",
                style_function=lambda feature, color=priority_color(row["prioridad_pmu"]): {
                    "fillColor": color,
                    "color": "#111111",
                    "weight": 1.2,
                    "fillOpacity": 0.75,
                },
                popup=folium.Popup(popup, max_width=350)
            ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        st_folium(
            m,
            width=1200,
            height=650,
            key="visor_pmu_principal"
        )

    else:
        st.warning("No existen PMU con los filtros seleccionados.")

    # =========================
    # TABLA PMU
    # =========================

    st.markdown("### Tabla PMU")

    st.dataframe(
        pmu_filtrada[
            [
                "pmu_id",
                "tipo_tu",
                "tipo_hmu",
                "area_ha",
                "volumen_m3",
                "prioridad_pmu",
                "accion_pmu",
            ]
        ],
        use_container_width=True
    )