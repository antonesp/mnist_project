from pathlib import Path
from tests import _PATH_DATA
import torch
from src.ml_ops.data import corrupt_mnist
import pytest
import os

@pytest.mark.skipif(not os.path.exists("ml_ops/data"), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    base_dir = Path(__file__).parent.parent

    N_train = 30000 # This is dependant on the dataset and should be adjusted to fit
    N_test = 5000 # This is also dependant on the data


    train_set, test_set = corrupt_mnist()

    assert len(train_set) == N_train, "Unepected amount of trainning data"
    assert len(test_set) == N_test, "Unexpected amount of test data"


    for x,y in train_set:
        assert x.shape == torch.Size([1,28,28]), "Unepected shape of trainning data"
        assert y in range(0,10), "Invalid label in training data" 
    
    for x,y in test_set:
        assert x.shape == torch.Size([1,28,28]), "Unepected shape of test data"
        assert y in range(0,10), "Invalid label in test data" 
    

    train_targets = torch.unique(train_set.tensors[1])
    test_targets = torch.unique(test_set.tensors[1])

    assert (train_targets == torch.arange(0,10)).all()
    assert (test_targets == torch.arange(0,10)).all()
