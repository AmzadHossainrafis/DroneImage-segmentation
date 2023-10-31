from utils.common import * 
import torch
import tqdm
import logging
from tqdm import tqdm 



config = read_yaml("../../../config/config.yaml")   


'''
Bug report: 


    1. The model is not training. - Fixed
    2. The model is not saving. - Fixed 
    3. eval mode is not working. - Fixed 


'''
config = read_yaml("../../../config/config.yaml")
logger = logging.getLogger(__name__)


class Trainer(object): 
    """
    Trainer class for training the model, saving the model, logging the training and validation loss

    args:
        batch_size: batch size for training 
        model: model to train 
        optimizer: optimizer for training 
        scheduler: scheduler for training 
        train_loss: list of training loss 
        val_loss: list of validation loss 
        device: cpu or gpu

    return:
        train_loss: list of training loss 
        val_loss: list of validation loss
    
    
    """


    def __init__(self, batch_size , model, auto_save=False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.auto_save = auto_save
        self.train_config = config["Train_config"]
        

    def train(train_dl, val_dl,model,loss,optimizer):
        try : 
            for epoch in range(self.train_config["epochs"]):
                train_loss = []
                val_loss = []
                print("Training Started...")
                logger.info(f"Training Started for epoch {epoch+1}")
                for epoch in tqdm(range(self.train_config["epochs"])):
                    print(f"Epoch {epoch+1} of {self.train_config['epochs']}")
                    model.train()
                    running_loss = 0.0
                    for x, y in tqdm(train_dl):
                        x = x.to("cuda")
                        y = y.to("cuda")
                        optimizer.zero_grad()
                        y_hat = model(x)
                        loss = loss(y_hat, y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    epoch_loss = running_loss / len(train_dl)
                    train_loss.append(epoch_loss)
                    print(f"Train Loss: {epoch_loss:.4f}")
                    
                    model.eval()
                    running_loss = 0.0
                    for x, y in val_dl:
                        x = x.to("cuda")
                        y = y.to("cuda")
                        with torch.no_grad():
                            y_hat = model(x)
                            loss = loss(y_hat, y)
                            running_loss += loss.item()

                    logger.info(f"Epoch {epoch+1} - Validation Loss: {epoch_loss:.4f}")
                    epoch_loss = running_loss / len(val_dl)
                    val_loss.append(epoch_loss)
                    print(f"Val Loss: {epoch_loss:.4f}")
            

            if self.auto_save:
                torch.save(model.state_dict(), config["artifacts"]["model_dir"] + "model.pt")
                logger.info("Model saved successfully")
                print("Model saved successfully")

                return train_loss, val_loss 
            else:
                return train_loss, val_loss
        

        
        except Exception as e:
            logger.error("Error in training the model")
            print(e)
      



        
        


        

