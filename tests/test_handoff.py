from pathlib import Path
from scripts.smoke_test_handoff import run_smoke_test


def test_handoff_smoke_test_execution():
    report = run_smoke_test()
    assert report["status"] == "PASSED"
    assert report["train_rows"] == 14594
    assert report["val_rows"] == 3000
    assert report["vocab_size"] > 0
    assert report["embedding_dim"] == 16
    assert report["recommended_p95_seq_length"] == 22
    assert Path(report["smoke_test_prediction_file"]).exists()
    assert "macro_f1" in report["smoke_test_metrics"]

