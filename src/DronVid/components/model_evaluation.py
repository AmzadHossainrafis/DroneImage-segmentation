import torch 
import torch.nn as nn 



class ModelEvaluation: 
    '''
    Class for evaluating the model

    '''
    def __init__(self, model_dir, device, criterion):
        '''
        Constructor for ModelEvaluation class

        Args:
            model (torch.nn.Module): the model to be evaluated
            device (torch.device): the device to run the model on
            criterion (torch.nn.Module): the loss function to be used
        '''
        self.model = torch.load(model_dir)
        self.device = device 
        self.criterion = criterion  


    def evaluate(self, data_loader):
        '''
        Evaluate the model

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader to be used for evaluation

        Returns:
            loss (float): the loss of the model on the given data
            accuracy (float): the accuracy of the model on the given data
        '''
        loss = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
              
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                accuracy += self.calculate_accuracy(outputs, labels)
        return loss / len(data_loader), accuracy / len(data_loader)
    

    
