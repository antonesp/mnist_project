from pathlib import Path

import pytest
import torch

from src.ml_ops.model import Model
from tests import _PATH_DATA, _PROJECT_ROOT


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = Model()

    dummy_input = torch.randn(batch_size, 1, 28, 28)

    print("the new yaml file did something")

    assert model(dummy_input).shape == (batch_size,10), "Wrong dimensions of output"