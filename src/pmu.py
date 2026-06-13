import geopandas as gpd


def recommend_pmu_action(tipo_tu, tipo_hmu, volumen_m3, area_ha):
    """
    Motor simple de recomendación territorial.
    """

    if "Manejo urbano" in tipo_tu or "Manejo urbano" in tipo_hmu:
        return "SUDS urbano: pavimento permeable, jardines de lluvia, parques inundables"

    if "Retención" in tipo_hmu:
        return "Reservorio / almacenamiento temporal"

    if "Drenaje" in tipo_tu or "Drenaje" in tipo_hmu:
        return "Drenaje controlado y estabilización de cauces"

    if "Recarga" in tipo_tu:
        return "Zanjas de infiltración / recarga hídrica"

    if "Producción" in tipo_tu:
        return "Agricultura regenerativa, curvas a nivel y zanjas de infiltración"

    if "Conservación" in tipo_tu:
        return "Conservación/restauración hídrica"

    return "Manejo territorial general"


def classify_pmu_priority(volumen, p50, p70, p90):
    if volumen >= p90:
        return "Muy alta"
    elif volumen >= p70:
        return "Alta"
    elif volumen >= p50:
        return "Media"
    else:
        return "Baja"


def generate_pmu(
    tu_gpkg,
    hmu_gpkg,
    output_gpkg,
    output_csv
):
    """
    Cruza Territorial Units con HMU watershed para generar PMU.
    """

    tu = gpd.read_file(tu_gpkg)
    hmu = gpd.read_file(hmu_gpkg)

    if tu.crs != hmu.crs:
        hmu = hmu.to_crs(tu.crs)

    pmu = gpd.overlay(
        tu,
        hmu,
        how="intersection"
    )

    if pmu.empty:
        print("No se generaron PMU.")
        return pmu

    pmu["area_m2"] = pmu.geometry.area
    pmu["area_ha"] = pmu["area_m2"] / 10000

    p50 = pmu["volumen_m3"].quantile(0.50)
    p70 = pmu["volumen_m3"].quantile(0.70)
    p90 = pmu["volumen_m3"].quantile(0.90)

    print("\nUMBRALES PMU")
    print("P50 =", round(p50, 2))
    print("P70 =", round(p70, 2))
    print("P90 =", round(p90, 2))

    pmu["prioridad_pmu"] = pmu["volumen_m3"].apply(
    lambda x: classify_pmu_priority(x, p50, p70, p90)
    )

    pmu["accion_pmu"] = pmu.apply(
        lambda row: recommend_pmu_action(
            row.get("tipo_tu", ""),
            row.get("tipo_hmu", ""),
            row.get("volumen_m3", 0),
            row.get("area_ha", 0)
        ),
        axis=1
    )

    pmu["pmu_id"] = range(1, len(pmu) + 1)

    keep_cols = [
        "pmu_id",
        "tu_id",
        "hmu_id",
        "tipo_tu",
        "tipo_hmu",
        "area_ha",
        "volumen_m3",
        "prioridad_pmu",
        "accion_pmu",
        "geometry"
    ]

    pmu = pmu[[c for c in keep_cols if c in pmu.columns]]

    pmu.to_file(output_gpkg, layer="pmu", driver="GPKG")
    pmu.drop(columns="geometry").to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("PMU guardadas en:", output_gpkg)
    print("CSV PMU guardado en:", output_csv)
    print("Número de PMU:", len(pmu))
    print(pmu["accion_pmu"].value_counts())
    print(pmu["prioridad_pmu"].value_counts())
    print("\nACCIONES PMU")
    print(pmu["accion_pmu"].value_counts())

    print("\nPRIORIDADES PMU")
    print(pmu["prioridad_pmu"].value_counts())
    return pmu