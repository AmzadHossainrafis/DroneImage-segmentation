from tqdm import tqdm 
import torch
from DronVid.components.utils.logger import logger
from DronVid.components.utils.common import read_yaml
from torch.cuda.amp import autocast, GradScaler
import mlflow 



config = read_yaml("config/config.yaml")   


'''
Bug report: 


    1. The model is not training. - Fixed
    2. The model is not saving. - Fixed 
    3. eval mode is not working. - Fixed 
    4. model is not evaluating - not fixed
    error:
           train_pipeline.runner()
            File "G:\CamVid\src\DronVid\pipeline\train_pipeline.py", line 98, in runner
            train_loss,val_loss=trainer.train(train_dl, val_dl, config["Train_config"]["epochs"], config["Dir"]["model_dir"])
            TypeError: cannot unpack non-iterable NoneType object


'''





class Trainer:

    """
    Trainer class is responsible for training the model and saving the model in the model directory.
    train_step: train the model for one epoch
    validate_step: validate the model for one epoch
    save_model: save the model with lowest validation loss
    train: train the model for given number of epochs and save the best model
    

    args:
        model: model to train
        loss: loss function
        optimizer: optimizer
        device: device to train on

    return:
        None

    """

    def __init__(self, model, loss, optimizer, device):
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')


    def train_step(self, dataloader):
        scaler = GradScaler()
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
        for i, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (i + 1)})
        pbar.close()

    def validate_step(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
        with torch.no_grad():
            for i, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                with autocast():
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (i + 1)})
        pbar.close()
        return total_loss / len(dataloader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, train_dl, val_dl, epochs, model_path):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("batch_size", len(train_dl.dataset))
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("model", str(self.model))
            mlflow.log_param("optimizer", str(self.optimizer))
            mlflow.log_param("device", str(self.device))
            mlflow.log_param("loss", str(self.criterion))

            for epoch in range(epochs-1):
                self.train_step(train_dl)
                val_loss = self.validate_step(val_dl)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(model_path)
                    logger.info("Saving model with lowest loss: {}".format(val_loss))
                logger.info("Epoch: {} | val_loss: {}".format(epoch, val_loss))
                logger.info("Best val loss: {}".format(self.best_val_loss))
                logger.info("--------------------------------------------------")

                # Log metrics
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("best_val_loss", self.best_val_loss, step=epoch)    
            return self.train_loss, self.val_loss

  
       
        
        


        

