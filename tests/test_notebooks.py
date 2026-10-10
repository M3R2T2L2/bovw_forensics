"""Guard the generated notebooks: each must contain its own run cell and no other's."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"


def _code(name):
    nb = json.loads((NB / name).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_00_stl10_minimal():
    src = _code("00_p0_minimal_stl10.ipynb")
    assert "from bovw.sweep import load_config, run" in src and "df = run(cfg)" in src
    assert "anomaly" not in src


def test_notebook_01_stl10_full():
    src = _code("01_p0_stl10_full.ipynb")
    assert "select_best_variant" in src and "p0_stl10_full.yaml" in src
    assert "anomaly" not in src


def test_notebook_02_anomaly():
    src = _code("02_p0_anomaly_mvtec.ipynb")
    assert "df = anomaly.run(cfg)" in src and "ad_data.ensure_archive(DRIVE_TAR)" in src
    assert "plots.anomaly_vs_k" in src
    assert "df = run(cfg)" not in src and "metric_vs_k" not in src  # no STL-10 cells spliced in


def test_notebook_03_followup():
    src = _code("03_p0_anomaly_followup.ipynb")
    assert "p0_anomaly_mvtec_v2.yaml" in src and "df = anomaly.run(cfg)" in src
    assert "ad_data.ensure_archive(DRIVE_TAR)" in src and "I_ACCEPT_MVTEC_LICENCE" in src
    assert "plots.false_alarms" in src and "anomaly.followup_table" in src
    assert "df = run(cfg)" not in src and "metric_vs_k" not in src


def test_notebook_04_visa():
    src = _code("04_p0_anomaly_visa.ipynb")
    assert "p0_anomaly_visa.yaml" in src and "df = anomaly.run(cfg)" in src
    assert "ad_data.VISA_URL" in src and "anomaly.prereg_checks" in src
    assert "MVTEC" not in src and "df = run(cfg)" not in src


def test_notebook_05_discovery():
    src = _code("05_p0_defect_discovery.ipynb")
    assert "p0_defect_discovery_mvtec.yaml" in src and "df = discovery.run(cfg)" in src
    assert "ad_data.ensure_archive(DRIVE_TAR)" in src and "discovery.cluster_category" in src
    assert "anomaly.run(cfg)" not in src and "df = run(cfg)" not in src


def test_notebook_06_3cad():
    src = _code("06_p0_anomaly_3cad.ipynb")
    assert "p0_anomaly_3cad.yaml" in src and "df = anomaly.run(cfg)" in src
    assert "THREECAD_GDRIVE_ID" in src and "anomaly.prereg_checks" in src and "count_mvtec_style" in src
    assert "VISA_URL" not in src and "df = run(cfg)" not in src


def test_notebook_07_hybrid():
    src = _code("07_p0_hybrid.ipynb")
    assert "p0_hybrid_{ds}.yaml" in src and "anomaly.hybrid_compare" in src
    assert "ensure_archive" not in src and "df = run(cfg)" not in src


def test_notebook_08_hybrid_3cad():
    src = _code("08_p0_hybrid_3cad.ipynb")
    assert "p0_hybrid_3cad.yaml" in src and "anomaly.prereg_checks_hybrid" in src
    assert "THREECAD_GDRIVE_ID" in src and "torch.cuda.is_available" in src
    assert "df = run(cfg)" not in src


def test_notebook_09_review():
    src = _code("09_p0_review_controls.ipynb")
    for k in ["p0_review_{ds}.yaml", "image_level_table", "review.benchmark", "p0_review_discovery_mvtec.yaml",
              "p0_review_stl10.yaml", "stl10_stability"]:
        assert k in src, k
