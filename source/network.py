import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm

class NNClassifier():
    def __init__(self, model, lr=1e-3, device='cpu', optimizer=None):
        """
        If optimizer is passed, then lr will be ignored
        """
        self.model = model
        self.device = device
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
        self.train_history = []
        self.valid_history = []

    
    def predict_proba(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        self.model.eval()
        self.model.to(self.device)
        X = torch.FloatTensor(X).to(self.device)
        N = len(X)
        
        proba = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                X_batch = X[i : min(i + batch_size, N)]

                proba.append(self.model(X_batch))
            
        proba = torch.cat(proba).to('cpu').numpy()
        return proba
    
    
    def predict(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        proba = self.predict_proba(X, batch_size)
        predict = proba.argmax(1)
        return predict
    
    
    def evaluate_score(self, X, y, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        predict = self.predict(X, batch_size)
        return (predict == y).mean()
    
    
    def loss(self, X, y, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        proba = self.predict_proba(X, batch_size)
        proba = torch.FloatTensor(proba).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        loss = torch.nn.functional.cross_entropy(proba, y).item()
        return loss


    def fit(self, X, y, epochs, batch_size, valid_data=None, log_every_epoch=None):
        """
        Parameters:
        -----------
        - X: numpy.array
        
        - y: numpy.array
        
        - batch_size: int
        
        - valid_data: tuple (numpy.array, numpy.array) (default: None)
            (X_valid, y_valid)
            
        - log_every_epoch: int (default: None)
        """
        self.model.train()
        self.model.to(self.device)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)

        N = len(X)
        
        bar = tqdm(range(1, epochs+1)) # progress bar
        for epoch in bar:
            cum_loss_train = 0
            part = 0
            for i in range(0, N, batch_size):
                part += 1
                X_batch = X[i : min(i + batch_size, N)]
                y_batch = y[i : min(i + batch_size, N)]

                proba_batch = self.model(X_batch)

                loss = torch.nn.functional.cross_entropy(proba_batch, y_batch)
                cum_loss_train += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.train_history.append(cum_loss_train / part)
                
            if valid_data is not None:
                valid_loss = self.loss(valid_data[0], valid_data[1], batch_size)
                self.valid_history.append(valid_loss)
                    
            if log_every_epoch is not None and epoch % log_every_epoch == 0:
                descr = None
                t_loss = self.train_history[-1]
                descr = ('t_loss: %5.3f' % t_loss)
                
                if valid_data is not None:
                    v_loss = self.valid_history[-1]
                    descr += ('v_loss: %5.3f' % v_loss)
                    
                bar.set_description(descr)

    def fit_loader(self, train_loader, valid_loader, epochs, log_every_epoch=None):
        """
        Fit neural network with torch.utils.data.DataLoader

        Parameters:
        -----------
        - train_loader: torch.utils.data.DataLoader

        - valid_loader: torch.utils.data.DataLoader

        - epochs: int
            
        - log_every_epoch: int (default: None)
        """
        self.model.train()
        self.model.to(self.device)
        
        bar = tqdm(range(1, epochs+1)) # progress bar
        for epoch in bar:
            cum_loss_train = 0
            part = 0
            for X_batch, y_batch in train_loader:
                part += 1
                X_batch =X_batch.to(self.device)
                y_batch =y_batch.to(self.device)

                proba_batch = self.model(X_batch)

                loss = torch.nn.functional.cross_entropy(proba_batch, y_batch)
                cum_loss_train += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.train_history.append(cum_loss_train / part)
                
            if valid_loader is not None:
                valid_loss = self.validate_loader(valid_loader)
                self.valid_history.append(valid_loss)
                    
            if log_every_epoch is not None and epoch % log_every_epoch == 0:
                descr = None
                t_loss = self.train_history[-1]
                descr = ('t_loss: %5.3f' % t_loss)
                
                if valid_loader is not None:
                    v_loss = self.valid_history[-1]
                    descr += ('v_loss: %5.3f' % v_loss)
                    
                bar.set_description(descr)

    def validate_loader(self, loader):
        """
        Validate loader.

        Parameters:
        -----------
        - loader: torch.utils.data.DataLoader
        """
        self.model.eval()
        self.model.to(self.device)
        
        proba = []
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                proba = self.model(X_batch)
                loss += torch.nn.functional.cross_entropy(proba, y_batch).item()

        return loss / len(loader)



                    


        