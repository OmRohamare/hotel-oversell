"""End-to-end test: run every stage on synthetic data in an isolated copy of the project.

Checks the rules the notebook sets: nothing deleted at load, split by date, no group of identical
rows spans two periods, both data variants run, the frozen buffers are not changed by the proof
run, every proof run is recorded, and failures exit with a non-zero code.
Skipped if contract.py is not in the repo.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.conftest import build_raw

REPO = Path(__file__).resolve().parent.parent
N_ROWS = 6000
METHODS = ["1 Flat rule", "2 Segment table", "3 Logistic", "4 Random forest"]
VARIANTS = ["dedup", "raw"]


def run_stage(root: Path, module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", f"src.{module}", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def run_ok(root: Path, module: str, *args: str) -> None:
    result = run_stage(root, module, *args)
    assert result.returncode == 0, (
        f"{module} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def project(tmp_path_factory, contract):
    """An isolated copy of the project with a small synthetic dataset, built up to freeze_buffer."""
    root = tmp_path_factory.mktemp("hotel_project")
    shutil.copytree(
        REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    if not (root / "src" / "contract.py").exists():
        shutil.copy(Path(contract.__file__), root / "contract.py")

    params = yaml.safe_load((REPO / "params.yaml").read_text())
    params["data"].update(source="local", expected_rows=N_ROWS)
    params["models"]["random_forest"].update(n_estimators=20, n_jobs=1)
    params["bootstrap"]["n_boot"] = 100
    params["mlflow"]["enabled"] = False
    (root / "params.yaml").write_text(yaml.safe_dump(params))

    (root / "data" / "raw").mkdir(parents=True)
    build_raw(N_ROWS, seed=7).to_csv(root / "data" / "raw" / "hotels.csv", index=False)

    for stage in ["validate", "split", "backtest", "freeze_buffer"]:
        run_ok(root, stage)
    return root


def test_nothing_is_deleted_at_load(project):
    clean = pd.read_parquet(project / "data" / "interim" / "clean.parquet")
    assert len(clean) == N_ROWS
    assert {"is_dup", "row_key", "arrival_date"} <= set(clean.columns)
    assert clean.is_dup.sum() > 0


def test_dedup_is_smaller_than_raw_and_has_no_repeats(project):
    dedup = {
        p: pd.read_parquet(project / "data" / "processed" / "dedup" / f"{p}.parquet")
        for p in ["train", "choose", "proof"]
    }
    raw = {
        p: pd.read_parquet(project / "data" / "processed" / "raw" / f"{p}.parquet")
        for p in ["train", "choose", "proof"]
    }
    assert sum(len(v) for v in raw.values()) == N_ROWS
    assert sum(len(v) for v in dedup.values()) < N_ROWS
    assert not any(v.is_dup.any() for v in dedup.values())


def test_periods_are_ordered_by_date_in_both_variants(project):
    for slug in VARIANTS:
        parts = {
            p: pd.read_parquet(project / "data" / "processed" / slug / f"{p}.parquet")
            for p in ["train", "choose", "proof"]
        }
        assert parts["train"].arrival_date.max() < parts["choose"].arrival_date.min()
        assert parts["choose"].arrival_date.max() < parts["proof"].arrival_date.min()


def test_no_group_of_identical_rows_spans_two_periods(project):
    parts = {
        p: set(
            pd.read_parquet(
                project / "data" / "processed" / "raw" / f"{p}.parquet"
            ).row_key
        )
        for p in ["train", "choose", "proof"]
    }
    assert not parts["train"] & parts["choose"]
    assert not parts["train"] & parts["proof"]
    assert not parts["choose"] & parts["proof"]


def test_all_four_models_are_saved_for_each_variant(project):
    for slug in VARIANTS:
        for name in ["flat_rule", "segment_table", "logistic", "random_forest"]:
            assert (project / "models" / slug / f"{name}.joblib").exists()


def test_a_buffer_and_a_headline_model_are_frozen_for_each_variant(project):
    for slug in VARIANTS:
        frozen = json.loads(
            (project / "reports" / f"frozen_buffers_{slug}.json").read_text()
        )
        assert sorted(frozen["buffers"]) == sorted(METHODS)
        assert frozen["headline_model"] in METHODS
        for fb in frozen["buffers"].values():
            assert fb["family"] in ("constant", "spread") and fb["buffer"] >= 0


def test_variant_comparison_is_written(project):
    table = pd.read_csv(project / "reports" / "variant_comparison.csv", index_col=0)
    assert sorted(table.index) == sorted(METHODS)
    assert "buffer moved?" in table.columns


def test_proof_run_does_not_change_the_frozen_buffers(project):
    paths = [project / "reports" / f"frozen_buffers_{s}.json" for s in VARIANTS]
    before = [sha(p) for p in paths]
    run_ok(project, "prove")
    assert [sha(p) for p in paths] == before
    for slug in VARIANTS:
        assert (project / "reports" / f"poc_method_table_{slug}.csv").exists()
    runs = (project / "docs" / "PROVE_RUNS.txt").read_text().strip().splitlines()
    assert len(runs) == 1


def test_a_second_proof_run_is_recorded_and_warned_about(project):
    paths = [project / "reports" / f"frozen_buffers_{s}.json" for s in VARIANTS]
    before = [sha(p) for p in paths]
    result = run_stage(project, "prove")
    assert result.returncode == 0
    assert "already been scored" in result.stdout
    assert [sha(p) for p in paths] == before
    assert (
        len((project / "docs" / "PROVE_RUNS.txt").read_text().strip().splitlines()) == 2
    )


def test_proof_tables_have_every_method_and_savings_vs_the_flat_rule(project):
    for slug in VARIANTS:
        tbl = pd.read_csv(
            project / "reports" / f"poc_method_table_{slug}.csv", index_col="method"
        )
        for name in METHODS:
            assert name in tbl.index
        assert any(i.startswith("(ref)") for i in tbl.index)
        assert tbl.loc["1 Flat rule", "saving_vs_flat_eur"] == 0


def test_headline_table_and_decision_record_are_written(project):
    head = pd.read_csv(project / "reports" / "headline_table.csv", index_col="variant")
    assert len(head) == 2
    entry = (project / "reports" / "DECISIONS_dedup_entry.md").read_text()
    assert "Does the buffer recommendation flip?" in entry


def test_nightly_writes_a_recommendation_file(project):
    run_ok(project, "nightly", "--from", "2017-03-01", "--nights", "5")
    out = list((project / "artifacts").glob("nightly_2017-03-01_5n.csv"))
    assert len(out) == 1
    table = pd.read_csv(out[0])
    assert {"book", "expected", "rooms_to_oversell"} <= set(table.columns)
    assert (table.rooms_to_oversell >= 0).all()


def test_nightly_rejects_a_bad_date_and_an_unknown_variant(project):
    bad_date = run_stage(project, "nightly", "--from", "not-a-date")
    assert bad_date.returncode == 1 and "ERROR" in bad_date.stdout
    bad_variant = run_stage(
        project, "nightly", "--from", "2017-03-01", "--variant", "nope"
    )
    assert bad_variant.returncode == 1


def test_a_wrong_raw_shape_fails_with_a_logged_error(project):
    params_path = project / "params.yaml"
    original = params_path.read_text()
    try:
        params = yaml.safe_load(original)
        params["data"]["expected_rows"] = 5
        params_path.write_text(yaml.safe_dump(params))
        result = run_stage(project, "validate")
    finally:
        params_path.write_text(original)
    assert result.returncode == 1
    assert "Unexpected raw shape" in result.stdout


def test_prove_refuses_to_run_without_frozen_buffers(project):
    frozen_path = project / "reports" / "frozen_buffers_raw.json"
    backup = frozen_path.with_suffix(".bak")
    frozen_path.rename(backup)
    try:
        result = run_stage(project, "prove")
    finally:
        backup.rename(frozen_path)
    assert result.returncode == 1
    assert "Freeze the buffers" in result.stdout
