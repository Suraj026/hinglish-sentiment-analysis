import pytest

from backend.model_loader import ID_TO_LABEL, ModelRegistry


def test_prediction_uses_trained_neural_checkpoint():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nltk")
    registry = ModelRegistry()

    registry.initialize()
    result = registry.predict("This product is very useful", model_name="bilstm_conv")

    assert isinstance(registry._models["bilstm_conv"], torch.nn.Module)
    assert result.sentiment in ID_TO_LABEL.values()
    assert sum(
        (
            result.scores.negative,
            result.scores.neutral,
            result.scores.positive,
        )
    ) == pytest.approx(1.0, abs=0.0002)


def test_new_registry_does_not_report_hardcoded_usage():
    registry = ModelRegistry()

    assert registry.get_stats()["total_analyzed"] == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This movie is absolutely fantastic", "positive"),
        ("This is terrible", "negative"),
        ("This service is disappointing", "negative"),
        ("Bahut badhiya service", "positive"),
    ],
)
def test_real_world_examples_are_not_neutral(text, expected):
    registry = ModelRegistry()
    registry.initialize()

    assert registry.predict(text, "bilstm_conv").sentiment == expected
