import pytest # type: ignore
import torch 
from src.Camvid_segmentation.components.models import UNet1
from src.Camvid_segmentation.components.data_transformation import Transform
from ..config.config import read_yaml 

config = read_yaml("../../../config/config.yaml") 
model_config = config["Model_config"]


def test_model(): 
    model = UNet1(model_config["num_classes"])
    input_1 = torch.randn(1, 3, 224, 224)
    input_2 = torch.randn(1, 3, 512, 512)
    output = model(input_1)
    output_2 = model(input_2)

    assert model is not None
    assert output.shape == torch.Size([1, 12, 224, 224])
    assert output_2.shape == torch.Size([1, 12, 512, 512])




