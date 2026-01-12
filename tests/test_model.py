from pathlib import Path
from tests import _PATH_DATA, _PROJECT_ROOT
import torch
from src.ml_ops.model import Model
import pytest

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = Model()

    dummy_input = torch.randn(batch_size, 1, 28, 28)

    assert model(dummy_input).shape == (batch_size,10), "Wrong dimensions of output"