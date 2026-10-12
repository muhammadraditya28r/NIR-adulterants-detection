"""Loading and validation for the tabular NIR spectra dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

METADATA_COLUMNS = ("Img_ID", "Replicate", "Info", "Class", "TR/TS")
WAVELENGTHS_NM = tuple(range(980, 1661, 5))
SPECTRAL_COLUMNS = tuple(str(wavelength) for wavelength in WAVELENGTHS_NM)
EXPECTED_COLUMNS = (*METADATA_COLUMNS, *SPECTRAL_COLUMNS)

VALID_REPLICATES = {"R1", "R2", "R3"}
VALID_MATERIALS = {
    "Oregano",
    "Oregano - inflorescences",
    "Oregano (infl.)",  # Abbreviation used in the supplied CSV.
    "Myrtle",
    "Olive",
    "Strawberry tree",
    "Sumac",
}
VALID_CLASSES = {"Ad", "Or"}
VALID_PARTITIONS = {"TR", "TS"}


def load_spectra_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a semicolon-delimited NIR spectra CSV.

    Metadata strings are stripped of surrounding whitespace in the returned
    DataFrame. The input file is never modified.

    Raises:
        ValueError: If the file does not match the documented data contract.
    """
    data = pd.read_csv(path, sep=";", dtype="string")

    actual_columns = tuple(data.columns)
    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in actual_columns
    ]
    unexpected_columns = [
        column for column in actual_columns if column not in EXPECTED_COLUMNS
    ]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {missing_columns}")
        if unexpected_columns:
            details.append(f"unexpected columns: {unexpected_columns}")
        raise ValueError("CSV schema mismatch; " + "; ".join(details))

    if not data.columns.is_unique:
        raise ValueError("CSV schema mismatch; column names must be unique")

    if data.empty:
        raise ValueError("CSV must contain at least one spectrum row")

    for column in METADATA_COLUMNS:
        data[column] = data[column].str.strip()
        if data[column].isna().any() or data[column].eq("").any():
            raise ValueError(
                f"Metadata column '{column}' contains missing or blank values"
            )

    _validate_values(data, "Replicate", VALID_REPLICATES)
    _validate_values(data, "Info", VALID_MATERIALS)
    _validate_values(data, "Class", VALID_CLASSES)
    _validate_values(data, "TR/TS", VALID_PARTITIONS)

    for column in SPECTRAL_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Spectral column '{column}' contains missing or non-numeric values"
            )
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Spectral column '{column}' contains non-finite values")
        data[column] = values

    return data


def _validate_values(data: pd.DataFrame, column: str, allowed: set[str]) -> None:
    observed = set(data[column].dropna().unique())
    invalid = sorted(observed - allowed)
    if invalid:
        raise ValueError(
            f"Metadata column '{column}' has invalid values {invalid}; "
            f"expected values from {sorted(allowed)}"
        )
