from DronVid.components.data_injection import DataIjection
from DronVid.components.data_transformation import Transform
from DronVid.components.model_evaluation import ModelEvaluation
from DronVid.components.data_loader import DroneDataset
from DronVid.components.trainer import Trainer
from DronVid.components.utils.logger import logger
from DronVid.components.utils.common import read_yaml , create_df
from DronVid.components.models import UNet1
import torch
from torch.utils.data import DataLoader 
import segmentation_models_pytorch as smp





config = read_yaml("config/config.yaml")

#train pipeline
    #data injection 
    #data transformation 
    #data loader

class TrainPipeline:
    """
    TrainPipeline class is responsible for training the model and saving the model in the model directory.
    includes the data injection, data transformation, data loader, model evaluation and trainer class.

    args: none 

    return:
        None




    """



    def __init__(self) -> None:
        
        self.data_injection = DataIjection()
        self.transform = Transform()
       
        self.model = smp.Unet('resnet34', encoder_weights='imagenet', classes=22, activation=None).to(config["Train_config"]["device"])
        self.DroneDataset= DroneDataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = torch.nn.CategoricalCrossEntropy()


    def runner(self):  
        logger.info("Training pipeline started")
        logger.info('-------------------------------------------------')
        logger.info('Data injection is started ')
        
        data_dir = config["Dir"]["original_data_path"]
        df = create_df(data_dir)
        X = df["id"].values
        data_injection = DataIjection()
        train, test, val = data_injection.inject(X)

        logger.info('Data injection finished')
        logger.info('-------------------------------------------------')
        logger.info('Data trasnformation started')


        train_augmentation=self.transform.get_transforms("train")
        logger.info(f'Transform is collected for train ')
        val_augmentatinon = self.transform.get_transforms('val')
        logger.info('Transform is collected for val')
        test_augmentation= self.transform.get_transforms('test')
        logger.info('Transform is collected for test')
        logger.info('Data trasnformation finished')
        logger.info('-------------------------------------------------')

        train_dataset = self.DroneDataset(config['Dir']["original_data_path"], config['Dir']["label_path"], train,train_augmentation)
        valid_dataset = self.DroneDataset(config['Dir']["original_data_path"], config['Dir']["label_path"], val,val_augmentatinon)
        test_dataset = self.DroneDataset(config['Dir']["original_data_path"], config['Dir']["label_path"], test,test_augmentation)


        train_dl = DataLoader(train_dataset, batch_size=config["Train_config"]["batch_size"], shuffle=True, num_workers=0)
        val_dl = DataLoader(valid_dataset, batch_size=config["Train_config"]["batch_size"], shuffle=False, num_workers=0)
        test_dl = DataLoader(test_dataset, batch_size=config["Train_config"]["batch_size"], shuffle=False, num_workers=0)

        logger.info('Data loader is created')
        logger.info('-------------------------------------------------')
        logger.info('Model is created')
        model = self.model
        logger.info('-------------------------------------------------')

        loss= self.loss
        #train the model 
        optimizer = self.optimizer
        logger.info('Training is started')
        logger.info('-------------------------------------------------')
        trainer = Trainer(model, loss, optimizer, config["Train_config"]["device"])
        train_loss,val_loss=trainer.train(train_dl, val_dl, config["Train_config"]["epochs"], config["Dir"]["model_dir"])
        logger.info('Training is finished')
        logger.info('-------------------------------------------------')
        logger.info(train_loss)
        logger.info(val_loss)
        logger.info('-------------------------------------------------')
        logger.info('Model is saved')
        # torch.save(model.state_dict(), config["Dir"]["model_dir"])
    
        logger.info('-------------------------------------------------')
        logger.info('Model evaluation started') 
        logger.info('-------------------------------------------------')

        model_evaluation = ModelEvaluation(config["Dir"]["model_dir"], config["Train_config"]["device"], loss)
        model_evaluation.evaluate(test_dl, model, loss)
        logger.info('-------------------------------------------------')
        logger.info('Model is evaluated')


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.runner()
