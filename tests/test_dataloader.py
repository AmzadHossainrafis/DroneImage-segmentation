import Camvid_segmentation
from Camvid_segmentation.components.data_loader import DataLoader

def test_data_loader():
    data_loader = DataLoader()
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None