from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_benchmark_v2_family_ood import (  # noqa: E402
    aggregate_error_labels,
    compute_ood_distances,
    ensure_finite_descriptor_records,
    fit_train_standardizer,
)


FEATURES = ("package_width_mm", "occupied_area_fraction", "source_base_centered_rms_K")
TRAIN = ("f001", "f002", "f003", "f004")
HELDOUT = ("f041", "f044")


def main() -> None:
    test_train_only_standardization()
    test_nearest_family_ranking()
    test_heldout_targets_do_not_change_distances()
    test_finite_descriptor_vectors()
    test_distance_outputs_are_deterministic()
    print("benchmark v2 family OOD tests passed")


def synthetic_records() -> list[dict[str, float | str]]:
    return [
        row("f001", "train", 10.0, 0.20, 1.0),
        row("f002", "train", 20.0, 0.30, 2.0),
        row("f003", "train", 30.0, 0.40, 3.0),
        row("f004", "train", 40.0, 0.50, 4.0),
        row("f041", "val", 21.0, 0.31, 2.1),
        row("f044", "test", 70.0, 0.80, 9.0),
    ]


def row(uid: str, split: str, width: float, occupied: float, centered: float) -> dict[str, float | str]:
    return {
        "family_uid": uid,
        "split": split,
        "package_width_mm": width,
        "occupied_area_fraction": occupied,
        "source_base_centered_rms_K": centered,
    }


def test_train_only_standardization() -> None:
    records = synthetic_records()
    scaler = fit_train_standardizer(records, FEATURES, TRAIN)
    train_matrix = np.asarray(
        [[float(record[name]) for name in FEATURES] for record in records if record["family_uid"] in TRAIN],
        dtype=np.float64,
    )
    assert np.allclose(scaler.mean, train_matrix.mean(axis=0))
    assert np.allclose(scaler.std, train_matrix.std(axis=0))
    all_family_mean = np.asarray(
        [[float(record[name]) for name in FEATURES] for record in records],
        dtype=np.float64,
    ).mean(axis=0)
    assert not np.allclose(scaler.mean, all_family_mean)
    shifted = synthetic_records()
    next(record for record in shifted if record["family_uid"] == "f044")["package_width_mm"] = 7000.0
    shifted_scaler = fit_train_standardizer(shifted, FEATURES, TRAIN)
    assert np.array_equal(scaler.mean, shifted_scaler.mean)
    assert np.array_equal(scaler.scale, shifted_scaler.scale)


def test_nearest_family_ranking() -> None:
    result = compute_ood_distances(
        synthetic_records(),
        FEATURES,
        TRAIN,
        HELDOUT,
        regularization=0.1,
        top_k=3,
    )
    f041 = [row for row in result["nearest_rows"] if row["heldout_family_uid"] == "f041"]
    assert f041[0]["rank"] == 1
    assert f041[0]["train_family_uid"] == "f002"
    f044 = [row for row in result["nearest_rows"] if row["heldout_family_uid"] == "f044"]
    assert f044[0]["train_family_uid"] == "f004"
    assert result["family_scores"]["f044"]["nearest_euclidean_distance"] > result["family_scores"]["f041"]["nearest_euclidean_distance"]


def test_heldout_targets_do_not_change_distances() -> None:
    records = synthetic_records()
    first = compute_ood_distances(
        records,
        FEATURES,
        TRAIN,
        HELDOUT,
        regularization=0.1,
        top_k=4,
    )
    error_rows_a = error_rows(final_f041=2.0, final_f044=8.0)
    error_rows_b = error_rows(final_f041=2000.0, final_f044=-8000.0)
    labels_a = aggregate_error_labels(error_rows_a, required_families=HELDOUT)
    labels_b = aggregate_error_labels(error_rows_b, required_families=HELDOUT)
    assert labels_a != labels_b
    second = compute_ood_distances(
        records,
        FEATURES,
        TRAIN,
        HELDOUT,
        regularization=0.1,
        top_k=4,
    )
    assert first["nearest_rows"] == second["nearest_rows"]
    assert first["family_scores"] == second["family_scores"]


def test_finite_descriptor_vectors() -> None:
    records = synthetic_records()
    ensure_finite_descriptor_records(records, FEATURES)
    invalid = synthetic_records()
    invalid[-1]["source_base_centered_rms_K"] = float("nan")
    try:
        ensure_finite_descriptor_records(invalid, FEATURES)
    except ValueError as exc:
        assert "non-finite descriptors" in str(exc)
        assert "source_base_centered_rms_K" in str(exc)
    else:
        raise AssertionError("non-finite descriptors were accepted")


def test_distance_outputs_are_deterministic() -> None:
    kwargs = {
        "records": synthetic_records(),
        "feature_names": FEATURES,
        "train_family_uids": TRAIN,
        "heldout_family_uids": HELDOUT,
        "regularization": 0.25,
        "top_k": 4,
    }
    first = compute_ood_distances(**kwargs)
    second = compute_ood_distances(**kwargs)
    assert first["nearest_rows"] == second["nearest_rows"]
    assert first["family_scores"] == second["family_scores"]
    assert np.array_equal(first["inverse_covariance"], second["inverse_covariance"])
    for uid in (*TRAIN, *HELDOUT):
        assert np.array_equal(first["standardized_by_uid"][uid], second["standardized_by_uid"][uid])


def error_rows(*, final_f041: float, final_f044: float) -> list[dict[str, str]]:
    return [
        error_row("f041", final_f041),
        error_row("f041", final_f041 + 0.2),
        error_row("f044", final_f044),
        error_row("f044", final_f044 + 0.2),
    ]


def error_row(uid: str, final_mae: float) -> dict[str, str]:
    return {
        "family_uid": uid,
        "source_superposition_mae_K": "10.0",
        "final_cnn_mae_K": str(final_mae),
        "absolute_mean_correction_error_K": "1.0",
        "centered_spatial_mae_K": "2.0",
        "peak_temperature_abs_error_K": "3.0",
    }


if __name__ == "__main__":
    main()
