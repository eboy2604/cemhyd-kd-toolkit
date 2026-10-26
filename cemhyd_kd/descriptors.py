"""Format-safe loading and multiscale descriptors for labelled 3D volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

try:
    import gudhi as gd
except ImportError:  # deferred until topology is requested
    gd = None


DESCRIPTOR_COLUMNS = (
    "Porosity",
    "CSH_Fraction",
    "Max_Pore_Connectivity",
    "Pore_Component_Count",
    "Beta_0",
    "Beta_1",
    "Beta_2",
)


def parse_shape(text: str) -> tuple[int, int, int]:
    """Parse a CLI shape such as ``100,100,100``."""
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("shape must contain three positive integers, for example 100,100,100")
    return values  # type: ignore[return-value]


def load_volume(path: str | Path, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    """Load labelled data from NPY, TIFF (optional) or CEMHYD3D raw phase files.

    Raw CEMHYD3D files are accepted only with an explicit shape; that prevents a
    plausible-looking but incorrectly reshaped microstructure from entering an
    analysis silently.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("TIFF input requires `pip install tifffile`.") from exc
        array = tifffile.imread(path)
    else:
        if shape is None:
            raise ValueError("Raw .img/.ima input requires --shape, for example 100,100,100.")
        expected = int(np.prod(shape))
        n_bytes = path.stat().st_size
        if n_bytes == expected:
            array = np.fromfile(path, dtype=np.uint8)
        elif n_bytes == expected * np.dtype(np.int32).itemsize:
            array = np.fromfile(path, dtype=np.int32)
        else:
            try:
                array = np.loadtxt(path, dtype=np.int32)
            except Exception as exc:
                raise ValueError(
                    f"Cannot read {path.name}: expected {expected} uint8/int32 values for shape {shape}."
                ) from exc
    array = np.asarray(array)
    if array.ndim == 1 and shape is not None and array.size == int(np.prod(shape)):
        array = array.reshape(shape)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Phase labels must be integer-valued.")
    return array.astype(np.int32, copy=False)


def _parse_ids(values: Iterable[int] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    ids = tuple(default if values is None else values)
    if not ids:
        raise ValueError("At least one phase ID is required.")
    return ids


def _topology(mask: np.ndarray) -> dict[str, int]:
    if gd is None:
        raise RuntimeError("Topology calculation requires `pip install gudhi`.")
    filtration = np.where(mask, 0.0, 1.0).astype(np.float64)
    complex_ = gd.CubicalComplex(top_dimensional_cells=filtration)
    complex_.persistence(homology_coeff_field=2, min_persistence=0.0)

    def alive(dim: int) -> int:
        intervals = complex_.persistence_intervals_in_dimension(dim)
        if intervals.size == 0:
            return 0
        return int(np.sum((intervals[:, 0] <= 0.5) & (intervals[:, 1] > 0.5)))

    return {"Beta_0": alive(0), "Beta_1": alive(1), "Beta_2": alive(2)}


def extract_descriptors(
    volume: np.ndarray,
    *,
    pore_ids: Iterable[int] | None = None,
    csh_ids: Iterable[int] | None = None,
    topology_space: str = "pore",
) -> dict[str, float | int]:
    """Return the seven descriptors used by the manuscript's core models."""
    pore_ids = _parse_ids(pore_ids, (0,))
    csh_ids = _parse_ids(csh_ids, (14, 20, 21, 30))
    pore = np.isin(volume, pore_ids)
    csh = np.isin(volume, csh_ids)
    total_pore = int(pore.sum())
    if total_pore:
        labels, count = ndimage.label(pore, structure=ndimage.generate_binary_structure(3, 1))
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(sizes.max()) if sizes.size else 0
        connectivity = float(largest / total_pore)
    else:
        count, largest, connectivity = 0, 0, 0.0
    if topology_space not in {"pore", "solid"}:
        raise ValueError("topology_space must be 'pore' or 'solid'.")
    topology = _topology(pore if topology_space == "pore" else ~pore)
    return {
        "Porosity": float(pore.mean()),
        "CSH_Fraction": float(csh.mean()),
        "Max_Pore_Connectivity": connectivity,
        "Pore_Component_Count": int(count),
        **topology,
    }
