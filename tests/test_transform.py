from src.Camvid_segmentation.components.data_transformation import Transform
import pytest # type: ignore
def test_trasform(): 
    transform = Transform("train")
    transform_1 = transform.get_transforms()
    transform = Transform("val")
    transform_2 = transform.get_transforms()
    transform = Transform("test")
    transform_3 = transform.get_transforms()

    assert transform_1 is not None 
    assert transform_2 is not None 
    assert transform_3 is not None 
    