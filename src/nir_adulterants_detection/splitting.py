"""Group-aware partitions for replicate NIR image measurements."""

import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPLICATE_GROUP_COLUMN = "Replicate_Group"
GROUPED_PARTITION_COLUMN = "Grouped_TR_TS"
_IMAGE_ID_PATTERN = re.compile(r"^(?P<group>.+)_(?P<replicate>R[123])$")


def add_grouped_partition(
    data: pd.DataFrame,
    *,
    n_splits: int = 4,
    test_fold: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy with replicate families assigned wholly to TR or TS.

    The family key is derived by removing the final ``_R1``/``_R2``/``_R3``
    suffix from ``Img_ID``. The source ``TR/TS`` column is preserved.

    Stratification uses labels only to balance folds; it does not fit or inspect
    spectral features. The chosen fold and seed should be fixed before model
    evaluation and not selected based on model performance.
    """
    required_columns = {"Img_ID", "Replicate", "Class"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if not isinstance(test_fold, int) or not 0 <= test_fold < n_splits:
        raise ValueError("test_fold must be an integer between 0 and n_splits - 1")

    image_ids = data["Img_ID"].astype("string").str.strip()
    replicate_values = data["Replicate"].astype("string").str.strip()
    matches = image_ids.str.extract(_IMAGE_ID_PATTERN)
    invalid_ids = matches["group"].isna() | matches["replicate"].ne(replicate_values)
    if invalid_ids.any():
        examples = image_ids[invalid_ids].drop_duplicates().head(5).tolist()
        raise ValueError(
            "Img_ID must end in the matching _R1, _R2, or _R3 replicate suffix; "
            f"invalid examples: {examples}"
        )

    groups = matches["group"]
    labels = data["Class"].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("Class contains missing or blank values")

    classes_per_group = pd.DataFrame({"group": groups, "label": labels})
    class_group_counts = (
        classes_per_group.drop_duplicates().groupby("label")["group"].nunique()
    )
    if (class_group_counts < n_splits).any():
        raise ValueError(
            "Each class must have at least n_splits distinct replicate groups "
            "to appear in every stratified test fold"
        )
    if classes_per_group.groupby("group")["label"].nunique().gt(1).any():
        raise ValueError("Each replicate group must have exactly one Class label")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_indices = test_indices = None
    for fold_index, (candidate_train, candidate_test) in enumerate(
        splitter.split(data, labels, groups)
    ):
        if fold_index == test_fold:
            train_indices, test_indices = candidate_train, candidate_test
            break

    if train_indices is None or test_indices is None:
        raise ValueError("Could not construct the requested grouped split")

    train_classes = set(labels.iloc[train_indices])
    test_classes = set(labels.iloc[test_indices])
    if train_classes != set(labels) or test_classes != set(labels):
        raise ValueError("The selected fold does not contain every class in both sets")

    partition = pd.Series("TR", index=range(len(data)), dtype="string")
    partition.iloc[test_indices] = "TS"

    result = data.copy()
    result[REPLICATE_GROUP_COLUMN] = groups.to_numpy()
    result[GROUPED_PARTITION_COLUMN] = partition.to_numpy()
    return result
