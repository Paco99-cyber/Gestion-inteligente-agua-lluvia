import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import rasterio
import numpy as np


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="Gestión Inteligente de Agua Lluvia",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

PMU_PATH = OUTPUTS_DIR / "pmu.gpkg"
PMU_SUMMARY_PATH = OUTPUTS_DIR / "pmu_summary.csv"
TU_RASTER_PATH = OUTPUTS_DIR / "territorial_units.tif"

LLUVIA_BASE_MM = 50.0


# =====================================================
# FUNCIONES
# =====================================================

@st.cache_data
def cargar_pmu(path):
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf = gdf.to_crs(epsg=4326)

    gdf["geometry"] = gdf.geometry.simplify(
        0.00008,
        preserve_topology=True
    )

    return gdf


@st.cache_data
def cargar_csv(path):
    return pd.read_csv(path)


def asignar_eficiencia(accion):
    eficiencias = {
        "Conservación/restauración hídrica": 0.15,
        "Zanjas de infiltración / recarga hídrica": 0.35,
        "Reservorio / almacenamiento temporal": 0.55,
        "Agricultura regenerativa": 0.25,
        "Drenaje controlado": 0.20,
    }
    return eficiencias.get(accion, 0.15)


def calcular_costo(row):
    accion = row["accion_pmu"]

    if accion == "Conservación/restauración hídrica":
        return row["area_ha"] * 800

    if accion == "Agricultura regenerativa":
        return row["area_ha"] * 1200

    if accion == "Zanjas de infiltración / recarga hídrica":
        return row["volumen_gestionable_m3"] * 2.5

    if accion == "Reservorio / almacenamiento temporal":
        return row["volumen_gestionable_m3"] * 6.0

    if accion == "Drenaje controlado":
        return row["volumen_gestionable_m3"] * 4.0

    return 0


def render_tu_raster(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)

    # Colores RGBA por clase
    colores = {
        1: [44, 162, 95, 255],     # Conservación hídrica
        2: [65, 182, 196, 255],    # Recarga / infiltración
        3: [254, 224, 139, 255],   # Producción agrohidrológica
        4: [117, 107, 177, 255],   # Manejo urbano
        5: [251, 106, 74, 255],    # Drenaje / regulación
    }

    img = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

    for valor, color in colores.items():
        img[arr == valor] = color

    img[arr == 0] = [255, 255, 255, 0]

    return img


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Escenario hidrológico")

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
    p_event = 25.0
    lambda_ia = 0.20

elif scenario == "Fuerte":
    p_event = 50.0
    lambda_ia = 0.20

elif scenario == "Extremo":
    p_event = 100.0
    lambda_ia = 0.10

elif scenario == "Crítico":
    p_event = 140.0
    lambda_ia = 0.05

else:
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


# =====================================================
# TÍTULO
# =====================================================

st.title("Gestión Inteligente de Agua Lluvia")
st.caption("Dashboard light - sistema de apoyo a decisiones")


# =====================================================
# VALIDACIÓN Y CARGA DE DATOS
# =====================================================

if not PMU_PATH.exists():
    st.error(f"No se encontró el archivo: {PMU_PATH}")
    st.stop()

if not PMU_SUMMARY_PATH.exists():
    st.error(f"No se encontró el archivo: {PMU_SUMMARY_PATH}")
    st.stop()

if not TU_RASTER_PATH.exists():
    st.warning(f"No se encontró el ráster de Unidades Territoriales: {TU_RASTER_PATH}")

pmu = cargar_pmu(PMU_PATH)
pmu_summary = cargar_csv(PMU_SUMMARY_PATH)

required_cols = ["area_ha", "volumen_m3", "accion_pmu", "prioridad_pmu"]
missing = [c for c in required_cols if c not in pmu_summary.columns]

if missing:
    st.error(f"Faltan columnas en pmu_summary.csv: {missing}")
    st.stop()


# =====================================================
# CÁLCULOS LIVIANOS
# =====================================================

factor_lluvia = rain_mm / LLUVIA_BASE_MM

pmu_summary["volumen_escenario_m3"] = (
    pmu_summary["volumen_m3"] * factor_lluvia
)

pmu_summary["eficiencia"] = (
    pmu_summary["accion_pmu"]
    .apply(asignar_eficiencia)
)

pmu_summary["volumen_gestionable_m3"] = (
    pmu_summary["volumen_escenario_m3"]
    * pmu_summary["eficiencia"]
)

pmu_summary["reduccion_pct"] = (
    pmu_summary["eficiencia"] * 100
)

pmu_summary["costo_estimado_usd"] = (
    pmu_summary.apply(calcular_costo, axis=1)
)

pmu_count = len(pmu_summary)
area_total = pmu_summary["area_ha"].sum()
volumen_potencial_total = pmu_summary["volumen_escenario_m3"].sum()
volumen_gestionable_total = pmu_summary["volumen_gestionable_m3"].sum()

reduccion_potencial = (
    volumen_gestionable_total / volumen_potencial_total * 100
    if volumen_potencial_total > 0
    else 0
)

costo_total = pmu_summary["costo_estimado_usd"].sum()

usd_por_m3 = (
    costo_total / volumen_gestionable_total
    if volumen_gestionable_total > 0
    else 0
)

accion_dominante = (
    pmu_summary["accion_pmu"].value_counts().idxmax()
    if not pmu_summary.empty
    else "N/D"
)

prioridad_dominante = (
    pmu_summary["prioridad_pmu"].value_counts().idxmax()
    if not pmu_summary.empty
    else "N/D"
)


# =====================================================
# RESUMEN EJECUTIVO
# =====================================================

st.header("Resumen ejecutivo")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Escenario", scenario)
c2.metric("Lluvia evento (mm)", f"{rain_mm:.0f}")
c3.metric("λ Ia/S", f"{lambda_ia:.2f}")
c4.metric("PMU identificadas", pmu_count)

c5, c6, c7, c8 = st.columns(4)

c5.metric("Área prioritaria (ha)", f"{area_total:,.2f}")
c6.metric("Volumen potencial (m³)", f"{volumen_potencial_total:,.0f}")
c7.metric("Volumen gestionable total (m³)", f"{volumen_gestionable_total:,.0f}")
c8.metric("Reducción potencial (%)", f"{reduccion_potencial:.1f}")

c9, c10, c11 = st.columns(3)

c9.metric("Costo estimado (USD)", f"{costo_total:,.0f}")
c10.metric("USD / m³ gestionado", f"{usd_por_m3:.2f}")
c11.metric("Prioridad dominante", prioridad_dominante)

st.info(f"Acción dominante recomendada: **{accion_dominante}**")

# =====================================================
# GLOSARIO / INTERPRETACIÓN
# =====================================================

with st.expander("¿Cómo interpretar este dashboard?"):
    st.markdown("""
    **PMU - Unidad Prioritaria de Manejo**  
    Área identificada por el modelo como prioritaria para intervenir. Combina información territorial, hidrológica y de riesgo.

    **TU - Unidad Territorial**  
    Clasificación del territorio según su función hidrológica dominante: conservación, infiltración, producción, manejo urbano o regulación.

    **HMU - Unidad Hidrológica de Manejo**  
    Área delimitada por la topografía y las rutas naturales del agua. Representa una zona que drena hacia un mismo punto.

    **Volumen potencial**  
    Cantidad estimada de escorrentía generada por el evento de lluvia seleccionado dentro de las PMU.

    **Volumen gestionable total**  
    Parte del volumen potencial que podría captarse, infiltrarse, almacenarse o regularse mediante las intervenciones recomendadas.

    **Reducción potencial**  
    Porcentaje del volumen potencial que podría ser gestionado con las medidas propuestas.

    **Costo por m³ gestionado**  
    Indicador referencial que estima cuánto costaría gestionar un metro cúbico de agua lluvia. Se calcula como costo estimado total dividido para volumen gestionable total.
    """)
# =====================================================
# SECCIONES PRINCIPALES
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Unidades Territoriales",
        "PMU - Visor Esri",
        "Beneficio hidrológico",
        "Indicador costo-beneficio"
    ]
)


# =====================================================
# TAB 1: UNIDADES TERRITORIALES
# =====================================================

with tab1:
    st.subheader("Unidades Territoriales")

    col_mapa, col_leyenda = st.columns([3, 1])

    with col_mapa:
        if TU_RASTER_PATH.exists():
            tu_img = render_tu_raster(TU_RASTER_PATH)
            st.image(tu_img, use_container_width=True)
        else:
            st.error("No se puede mostrar territorial_units.tif")

    with col_leyenda:
        st.markdown("### Leyenda")
        st.markdown(
            """
            <div style="display:flex;align-items:center;margin-bottom:8px;">
                <div style="width:28px;height:28px;background:#2ca25f;border-radius:4px;margin-right:8px;"></div>
                Conservación hídrica
            </div>
            <div style="display:flex;align-items:center;margin-bottom:8px;">
                <div style="width:28px;height:28px;background:#41b6c4;border-radius:4px;margin-right:8px;"></div>
                Recarga / infiltración
            </div>
            <div style="display:flex;align-items:center;margin-bottom:8px;">
                <div style="width:28px;height:28px;background:#fee08b;border-radius:4px;margin-right:8px;"></div>
                Producción agrohidrológica
            </div>
            <div style="display:flex;align-items:center;margin-bottom:8px;">
                <div style="width:28px;height:28px;background:#756bb1;border-radius:4px;margin-right:8px;"></div>
                Manejo urbano
            </div>
            <div style="display:flex;align-items:center;margin-bottom:8px;">
                <div style="width:28px;height:28px;background:#fb6a4a;border-radius:4px;margin-right:8px;"></div>
                Drenaje / regulación
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# TAB 2: PMU VISOR ESRI
# =====================================================

with tab2:
    st.subheader("PMU prioritarias - Visor Esri")

    centro = pmu.geometry.unary_union.centroid

    m = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=13,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satelital",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)

    campos_pmu = [
        c for c in [
            "pmu_id",
            "prioridad_pmu",
            "accion_pmu",
            "area_ha",
            "volumen_m3"
        ]
        if c in pmu.columns
    ]

    folium.GeoJson(
        pmu,
        name="PMU",
        tooltip=folium.GeoJsonTooltip(
            fields=campos_pmu
        ) if campos_pmu else None,
        style_function=lambda feature: {
            "fillColor": "#ff7800",
            "color": "#ff0000",
            "weight": 2,
            "fillOpacity": 0.35,
        }
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m, width=1200, height=650)


# =====================================================
# TAB 3: BENEFICIO HIDROLÓGICO
# =====================================================

with tab3:
    st.subheader("Beneficio hidrológico estimado")

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Volumen PMU escenario",
        f"{volumen_potencial_total:,.0f} m³"
    )

    b2.metric(
        "Volumen gestionable estimado",
        f"{volumen_gestionable_total:,.0f} m³"
    )

    b3.metric(
        "Reducción potencial",
        f"{reduccion_potencial:.1f} %"
    )

    st.info(
        "El beneficio hidrológico se estima aplicando eficiencias referenciales "
        "según el tipo de intervención propuesta para cada Unidad Prioritaria de Manejo (PMU). "
        "Este cálculo es preliminar y sirve para comparar escenarios de lluvia."
    )

    st.success(
        f"Para el escenario **{scenario}** ({rain_mm:.0f} mm), "
        f"el sistema estima que las PMU identificadas podrían gestionar aproximadamente "
        f"**{volumen_gestionable_total:,.0f} m³** de escorrentía, equivalente a una reducción potencial "
        f"del **{reduccion_potencial:.1f}%** sobre el volumen PMU analizado."
    )


# =====================================================
# TAB 4: INDICADOR COSTO-BENEFICIO
# =====================================================

with tab4:
    st.subheader("Indicador económico costo-beneficio")

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Costo estimado",
        f"${costo_total:,.0f}"
    )

    e2.metric(
        "Costo por m³ gestionado",
        f"${usd_por_m3:,.2f}/m³"
    )

    e3.metric(
        "Volumen gestionado",
        f"{volumen_gestionable_total:,.0f} m³"
    )

    st.info(
        "Los costos son referenciales para el PMV. "
        "Deben ajustarse con precios locales, diseño definitivo, mano de obra, "
        "materiales y condiciones específicas del sitio."
    )

    st.success(
        f"Para el escenario **{scenario}**, el costo total estimado de intervención "
        f"es de aproximadamente **${costo_total:,.0f}**, con un costo medio de "
        f"**${usd_por_m3:,.2f}/m³ gestionado**."
    )

