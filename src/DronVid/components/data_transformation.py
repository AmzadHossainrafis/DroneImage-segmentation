import albumentations as A 
# from utils.common import read_yaml 
# from utils.logger import logger
from DronVid.components.utils.common import read_yaml
from DronVid.components.utils.logger import logger
config = read_yaml("config/config.yaml") 




'''
TODO: 
    transform class for training and validation data -done 
    all the transforms must be user defined -done

'''

class Transform(object):
    """
    Transform class for training, validation, and testing data 

    return:
        image: transformed image 
        mask: transformed mask 
    """
    def __init__(self,) -> None:
        self.transform_config = config["Transform_config"]
        

    def get_transforms(self, transform_type):
        if transform_type == 'train':
            return self.train_transforms()
        elif transform_type == 'val':
            return self.val_transforms()
        elif transform_type == 'test':
            return self.test_transforms()
        else:
            raise ValueError(f'Invalid transform type {self.transform_type}')

    def train_transforms(self):
        """
        Please add all the transforms here or comment out the transforms that you don't want to use 

        """

        logger.info("Training transforms returned")

        return A.Compose([
            A.Resize(self.transform_config["image_size"], self.transform_config["image_size"]),
            A.HorizontalFlip(self.transform_config["horizontal_flip_prob"]),
            A.VerticalFlip(self.transform_config["vertical_flip_prob"]),
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=20, shift_limit=0.1, p=0.5, border_mode=0),
            #A.Blur(self.transform_config["blur_prob"], self.transform_config["blur_limit"]),
            A.Normalize(),
        ])

    def val_transforms(self): 
        logger.info("Validation transforms returned")
        return A.Compose([
            A.Resize(self.transform_config["image_size"], self.transform_config["image_size"]),
            A.Normalize(),
        ])

    def test_transforms(self): 
        logger.info("Testing transforms returned")
        return A.Compose([
            A.Resize(self.transform_config["image_size"], self.transform_config["image_size"]),
            A.Normalize(),
        ])  
    
