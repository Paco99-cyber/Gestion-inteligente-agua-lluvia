import numpy as np
from pysheds.grid import Grid

print("terrain.py cargado correctamente")


def read_dem_array(dem):
    """
    Convierte el DEM en array NumPy.
    """
    dem_array = dem.read(1).astype("float64")

    nodata = dem.nodata
    if nodata is not None:
        dem_array[dem_array == nodata] = np.nan

    return dem_array


def print_dem_stats(dem_array):
    print("Forma del DEM:", dem_array.shape)
    print("Valor mínimo:", np.nanmin(dem_array))
    print("Valor máximo:", np.nanmax(dem_array))
    print("Valor medio:", np.nanmean(dem_array))


def calculate_slope(dem_array, resolution):
    """
    Calcula la pendiente en grados.
    """
    dz_dy, dz_dx = np.gradient(dem_array, resolution, resolution)
    slope_rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = np.degrees(np.arctan(slope_rise_run))
    return slope_deg


def print_slope_stats(slope):
    print("Pendiente mínima:", np.nanmin(slope))
    print("Pendiente máxima:", np.nanmax(slope))
    print("Pendiente media:", np.nanmean(slope))


def calculate_flow(dem_path):
    """
    Calcula dirección y acumulación de flujo usando PySheds.
    """
    print("Calculando flujo...")

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    filled_dem = grid.fill_depressions(dem)
    flow_dir = grid.flowdir(filled_dem)
    flow_acc = grid.accumulation(flow_dir)

    print("Flujo calculado correctamente")

    return flow_dir, flow_acc


def print_flow_stats(flow_acc):
    print("Acumulación mínima:", np.nanmin(flow_acc))
    print("Acumulación máxima:", np.nanmax(flow_acc))
    print("Acumulación media:", np.nanmean(flow_acc))


def extract_drainage_network(flow_acc, threshold):
    """
    Extrae una red de drenaje binaria a partir de la acumulación de flujo.
    """
    drainage = (flow_acc > threshold).astype("uint8")
    return drainage


def print_drainage_stats(drainage):
    """
    Imprime estadísticas básicas de la red de drenaje extraída.
    """
    total_cells = drainage.size
    drainage_cells = int(np.sum(drainage == 1))
    percentage = (drainage_cells / total_cells) * 100

    print("Celdas totales:", total_cells)
    print("Celdas de drenaje:", drainage_cells)
    print("Porcentaje drenaje (%):", percentage)


def plot_drainage_network(drainage, title="Red de drenaje preliminar"):
    """
    Visualiza la red de drenaje binaria.
    Si matplotlib no está disponible, solo muestra un aviso.
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.imshow(drainage, cmap="Blues")
        plt.title(title)
        plt.colorbar(label="Drenaje")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("No se pudo visualizar con matplotlib.")
        print("Detalle:", e)

def identify_hotspots(flow_acc, slope, acc_threshold, slope_threshold):
    """
    Identifica hotspots combinando alta acumulación y baja pendiente.

    Parámetros:
    - flow_acc: array de acumulación de flujo
    - slope: array de pendiente en grados
    - acc_threshold: umbral mínimo de acumulación
    - slope_threshold: umbral máximo de pendiente

    Retorna:
    - hotspots: raster binario (1 = hotspot, 0 = no hotspot)
    """
    hotspots = ((flow_acc > acc_threshold) & (slope < slope_threshold)).astype("uint8")
    return hotspots


def print_hotspot_stats(hotspots):
    """
    Imprime estadísticas básicas de hotspots.
    """
    total_cells = hotspots.size
    hotspot_cells = int(np.sum(hotspots == 1))
    percentage = (hotspot_cells / total_cells) * 100

    print("Celdas hotspot:", hotspot_cells)
    print("Porcentaje hotspot (%):", percentage)

