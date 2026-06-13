import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np


def apply_basin_mask(array, reference_path, basin_path):
    """
    Aplica máscara de cuenca:
    todo pixel fuera de la cuenca = 0
    """

    basin = gpd.read_file(basin_path)

    with rasterio.open(reference_path) as src:

        # reproyectar cuenca al CRS del raster
        basin = basin.to_crs(src.crs)

        mask = geometry_mask(
            basin.geometry,
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        )

    masked = np.where(mask, array, 0)

    return masked