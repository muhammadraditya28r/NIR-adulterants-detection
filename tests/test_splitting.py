import pandas as pd
import pytest

from nir_adulterants_detection.splitting import (
    GROUPED_PARTITION_COLUMN,
    REPLICATE_GROUP_COLUMN,
    add_grouped_partition,
)


@pytest.fixture
def replicated_spectra():
    rows = []
    for class_label, prefix in (("Ad", "adulterant"), ("Or", "oregano")):
        for family_number in range(4):
            family_id = f"{prefix}_{family_number:02d}"
            for replicate in ("R1", "R2", "R3"):
                rows.append(
                    {
                        "Img_ID": f"{family_id}_{replicate}",
                        "Replicate": replicate,
                        "Class": class_label,
                        # Deliberately put only R3 in the supplied test partition.
                        "TR/TS": "TS" if replicate == "R3" else "TR",
                    }
                )
    return pd.DataFrame(rows)


def test_keeps_images_and_replicate_families_in_one_partition(replicated_spectra):
    result = add_grouped_partition(replicated_spectra)

    train = result.loc[result[GROUPED_PARTITION_COLUMN] == "TR"]
    test = result.loc[result[GROUPED_PARTITION_COLUMN] == "TS"]

    assert set(train["Img_ID"]).isdisjoint(test["Img_ID"])
    assert set(train[REPLICATE_GROUP_COLUMN]).isdisjoint(test[REPLICATE_GROUP_COLUMN])
    assert (
        result.groupby(REPLICATE_GROUP_COLUMN)[GROUPED_PARTITION_COLUMN].nunique().max()
        == 1
    )


def test_same_seed_and_fold_produce_the_same_partition(replicated_spectra):
    first = add_grouped_partition(replicated_spectra, random_state=13, test_fold=2)
    second = add_grouped_partition(replicated_spectra, random_state=13, test_fold=2)

    pd.testing.assert_series_equal(
        first[GROUPED_PARTITION_COLUMN], second[GROUPED_PARTITION_COLUMN]
    )


def test_each_partition_contains_every_class(replicated_spectra):
    result = add_grouped_partition(replicated_spectra)
    expected_classes = set(replicated_spectra["Class"])

    for partition in ("TR", "TS"):
        actual_classes = set(
            result.loc[result[GROUPED_PARTITION_COLUMN] == partition, "Class"]
        )
        assert actual_classes == expected_classes


def test_preserves_supplied_partition_and_does_not_mutate_input(replicated_spectra):
    original = replicated_spectra.copy(deep=True)

    result = add_grouped_partition(replicated_spectra)

    pd.testing.assert_frame_equal(replicated_spectra, original)
    pd.testing.assert_series_equal(result["TR/TS"], original["TR/TS"])
    assert GROUPED_PARTITION_COLUMN in result


def test_rejects_image_id_with_mismatched_replicate_suffix(replicated_spectra):
    invalid_data = replicated_spectra.copy()
    invalid_data.loc[0, "Replicate"] = "R2"

    with pytest.raises(ValueError, match="matching _R1, _R2, or _R3"):
        add_grouped_partition(invalid_data)
