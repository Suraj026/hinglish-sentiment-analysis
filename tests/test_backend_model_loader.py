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
