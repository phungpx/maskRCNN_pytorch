import torch

from ...module import Module
from ignite import engine as e
from abc import abstractmethod


class Engine(Module):
    '''
        Base class for all engines. Your engine should subclass this class.
        Class Engine contains an Ignite Engine that controls running process over a dataset.
        Method _update is a function receiving the running Ignite Engine and the current batch in each iteration and returns data to be stored in the Ignite Engine's state.
        Parameters:
            dataset_name (str): dataset which engine run over.
            device (str): device on which model and tensor is allocated.
            max_epochs (int): number of epochs training process runs.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Engine, self).__init__()
        self.dataset = dataset
        self.device = device
        self.max_epochs = max_epochs
        self.engine = e.Engine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    @abstractmethod
    def _update(self, engine, batch):
        pass


class Trainer(Engine):
    '''
        Engine controls training process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        print(f'[Info] parameters of model: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)} params.')

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
        samples = list(sample.to(self.device) for sample in params[0])
        targets = [{k: v.to(self.device) for k, v in sample.items()} for sample in params[1]]
        losses = self.model(samples, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            samples = list(sample.to(self.device) for sample in params[0])
            targets = [{k: v.to(self.device) for k, v in sample.items()} for sample in params[1]]
            params[0], params[1] = self.model(samples), targets
            return params
