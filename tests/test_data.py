import pandas as pd
import pytest

from nir_adulterants_detection.data import (
    EXPECTED_COLUMNS,
    SPECTRAL_COLUMNS,
    load_spectra_csv,
)


def _write_csv(path, *, updates=None, columns=None):
    row = {
        "Img_ID": " image-01  ",
        "Replicate": " R1 ",
        "Info": " Oregano ",
        "Class": " Or ",
        "TR/TS": " TR ",
        **dict.fromkeys(SPECTRAL_COLUMNS, "0.25"),
    }
    row.update(updates or {})
    pd.DataFrame([row], columns=columns or EXPECTED_COLUMNS).to_csv(
        path, sep=";", index=False
    )


def test_loads_valid_csv_and_normalizes_metadata_whitespace(tmp_path):
    csv_path = tmp_path / "spectra.csv"
    _write_csv(csv_path)

    data = load_spectra_csv(csv_path)

    assert tuple(data.columns) == EXPECTED_COLUMNS
    assert data.loc[0, "Img_ID"] == "image-01"
    assert data.loc[0, "Replicate"] == "R1"
    assert data.loc[0, "TR/TS"] == "TR"
    assert data.loc[0, "980"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("columns", "error"),
    [
        (EXPECTED_COLUMNS[:-1], "missing columns"),
        ((*EXPECTED_COLUMNS, "extra"), "unexpected columns"),
    ],
)
def test_rejects_missing_or_unexpected_columns(tmp_path, columns, error):
    csv_path = tmp_path / "spectra.csv"
    _write_csv(csv_path, columns=columns)

    with pytest.raises(ValueError, match=error):
        load_spectra_csv(csv_path)


def test_rejects_invalid_partition(tmp_path):
    csv_path = tmp_path / "spectra.csv"
    _write_csv(csv_path, updates={"TR/TS": "VALIDATION"})

    with pytest.raises(ValueError, match="TR/TS.*invalid values"):
        load_spectra_csv(csv_path)


def test_rejects_non_numeric_spectrum_value(tmp_path):
    csv_path = tmp_path / "spectra.csv"
    _write_csv(csv_path, updates={"1020": "not-a-number"})

    with pytest.raises(ValueError, match="1020.*non-numeric"):
        load_spectra_csv(csv_path)
