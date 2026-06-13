import rasterio


def read_raster(path):
    """
    Lee un raster y retorna el dataset.
    """
    dataset = rasterio.open(str(path))
    return dataset


def print_raster_info(dataset):
    """
    Imprime información básica del raster.
    """
    print("CRS:", dataset.crs)
    print("Resolución:", dataset.res)
    print("Ancho:", dataset.width)
    print("Alto:", dataset.height)
    print("Número de bandas:", dataset.count)


def save_raster(output_path, array, reference_dataset, dtype="float32", nodata=None):
    """
    Guarda un array como raster usando otro raster como referencia.
    """
    profile = reference_dataset.profile.copy()
    profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata
    )

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)