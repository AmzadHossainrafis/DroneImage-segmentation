from tqdm import tqdm 
import torch
from DronVid.components.utils.logger import logger
from DronVid.components.utils.common import read_yaml
from torch.cuda.amp import autocast, GradScaler



config = read_yaml("config/config.yaml")   


'''
Bug report: 


    1. The model is not training. - Fixed
    2. The model is not saving. - Fixed 
    3. eval mode is not working. - Fixed 


'''





class Trainer:
    def __init__(self, model, loss, optimizer, device):
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')
        scaler= GradScaler()

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
        for epoch in range(epochs):
            self.train_step(train_dl)
            val_loss = self.validate_step(val_dl)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(model_path)
                logger.info("Saving model with lowest loss: {}".format(val_loss))
            logger.info("Epoch: {} | val_loss: {}".format(epoch, val_loss))
            logger.info("Best val loss: {}".format(self.best_val_loss))
            logger.info("--------------------------------------------------")

  
       
        
        


        

