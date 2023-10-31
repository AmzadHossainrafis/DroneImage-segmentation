import numpy as np
from PIL import Image 
from torch.utils.data import Dataset
import albumentations as A
from utils.common import read_yaml

# from data_transformation import Transform

config = read_yaml("../../../config/config.yaml") 


''' 
Bug report: 

    1. The dataset is not loading properly. - Fixed 
    2. Image and mask are not in the same order. - Fixed 
    3. The mask is not in one hot encoding. - Fixed
    

'''




class DroneDataset(Dataset):
    '''
    DroneDataset class for loading the dataset 

    args: 
        img_path: path of the images
        mask_path: path of the masks
        X: list of image names
        transform: data augmentation 

    
    return : 
        image: image tensor
        mask: mask tensor 
    
    '''
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.model_config = config["Model_config"]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_path + self.X[idx] + '.jpg'))
        mask = np.array(Image.open(self.mask_path + self.X[idx] + '.png')) # relabel classes from 1,2 --> 0,1 where 0 is background
        
        # augment images
        # to know more about albumentation visit https://albumentations.ai/docs/ 
        if self.transform!=None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        norm = A.Normalize()(image = image, mask = np.expand_dims(mask, 0))
        #covert mask to one hot encoding 

        
        mask_one_hot = np.zeros(( mask.shape[0], mask.shape[1],self.model_config['num_classes']),)  # 22 classe    , 
        for i, unique_value in enumerate(np.unique(mask)):
            mask_one_hot[:, :, i][mask == unique_value] = 1


        return norm['image'].transpose(2, 0, 1), mask_one_hot.transpose(2, 0, 1).astype('float32')
        


# if __name__ == "__main__":
#     data_dir = config["Dir"]["original_data_path"]
#     df = create_df(data_dir)
#     train_csv = config["Data_injection"]["train_csv"]
#     X = df["id"].values
#     transform = Transform("train").get_transforms()
#     dataset = DroneDataset(config["Dir"]["original_data_path"], config["Dir"]["label_path"], X, transform)
#     image, mask = dataset[0]
#     print(image.shape)
#     print(mask.shape)
   