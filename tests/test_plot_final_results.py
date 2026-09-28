import yaml

from src import plot_final_results
from tests.test_paired_analysis import CONFIG, _write_results


def test_all_four_figures_are_written(tmp_path, monkeypatch):
    results, figures = tmp_path / "results", tmp_path / "figures"
    _write_results(results, CONFIG["datasets"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "plot", "--results-dir", str(results), "--config", str(config_path),
        "--figures-dir", str(figures), "--n-bootstrap", "50",
    ])
    plot_final_results.main()
    names = {path.name for path in figures.iterdir()}
    for stem in ("fig1_mismatch_cost_by_rate", "fig2_conditions_by_rate",
                 "fig3_per_dataset_effect", "fig4_classifier_selection"):
        assert {f"{stem}.png", f"{stem}.pdf"} <= names
