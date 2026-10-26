import numpy as np

from cemhyd_kd.descriptors import DESCRIPTOR_COLUMNS, extract_descriptors, load_volume


def test_npy_volume_round_trip_and_descriptor_schema(tmp_path):
    volume = np.zeros((4, 4, 4), dtype=np.int32)
    volume[0, 0, 0] = 14
    path = tmp_path / "sample.npy"
    np.save(path, volume)
    result = extract_descriptors(load_volume(path))
    assert tuple(result) == DESCRIPTOR_COLUMNS
    assert result["Porosity"] == 63 / 64
    assert result["CSH_Fraction"] == 1 / 64


def test_raw_volume_requires_shape(tmp_path):
    path = tmp_path / "sample.img"
    np.arange(8, dtype=np.uint8).tofile(path)
    try:
        load_volume(path)
    except ValueError as error:
        assert "requires --shape" in str(error)
    else:
        raise AssertionError("Raw input should require a shape")
