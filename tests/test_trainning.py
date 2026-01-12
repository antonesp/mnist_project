from pathlib import Path
from tests import  _PROJECT_ROOT
from src.ml_ops.train import train



def test_train():
    try:
        train(epochs=1,model_name = "models/test_model.pth")
    except:
        print("Trainning did not finish corretly")

    model_path = Path("models/test_model.pth")
    print(model_path)
    assert model_path.is_file(), "Model not saved correctly"


