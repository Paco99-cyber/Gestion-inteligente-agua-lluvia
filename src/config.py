# Nota:
# Todos estos valores pueden ser ajustados según calibración futura
# ================================
# CONFIGURACIÓN GLOBAL DEL PROYECTO
# ================================

# Parámetro lambda para SCS-CN
LAMBDA_IA = 0.2

# Pesos del índice de riesgo
RISK_WEIGHTS = {
    "runoff": 0.30,
    "flow_accumulation": 0.30,
    "depression": 0.20,
    "slope": 0.10,
    "infiltration": 0.10
}

# Umbrales de clasificación de riesgo
RISK_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75
}

# Área mínima para considerar un hotspot (en m²)
MIN_HOTSPOT_AREA = 100

# Resolución base (puedes cambiar según datos)
DEFAULT_RESOLUTION = 10