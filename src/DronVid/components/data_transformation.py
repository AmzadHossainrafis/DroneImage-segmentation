import albumentations as A 
from utils.common import read_yaml 
from components import logger 

config = read_yaml("../../../config/config.yaml") 



'''
TODO: 
    transform class for training and validation data 
    all the transforms must be user defined 

'''

class Transform(object):
    """
    Transform class for training, validation, and testing data 

    return:
        image: transformed image 
        mask: transformed mask 
    """
    def __init__(self, transform_type) -> None:
        self.transform_config = config["Transform_config"]
        self.transform_type = transform_type

    def get_transforms(self):
        if self.transform_type == 'train':
            return self.train_transforms()
        elif self.transform_type == 'val':
            return self.val_transforms()
        elif self.transform_type == 'test':
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
        return A.Compose([
            A.Resize(self.transform_config["image_size"], self.transform_config["image_size"]),
            A.Normalize(),
        ])

    def test_transforms(self): 
        return A.Compose([
            A.Resize(self.transform_config["image_size"], self.transform_config["image_size"]),
            A.Normalize(),
        ])  
    

if __name__ == "__main__":
    transform = Transform("train").get_transforms()
    print(transform)