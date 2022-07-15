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
        assert 'loss' in self.frame, 'The frame does not have loss.'
        assert 'logger' in self.frame, 'The frame does not have logger.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']
        self.logger = self.frame['logger']
        self.scaler = self.frame.get('scaler', None)
        self.writer = self.frame.get('writer', None)
        if self.scaler is not None:
            self.logger.info('using FP16 mode.')

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
        samples = list(sample.to(self.device) for sample in params[0])
        targets = [{k: v.to(self.device) for k, v in sample.items()} for sample in params[1]]

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            losses = self.model(samples, targets)
            loss = self.loss(losses)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.writer is not None:
            step = engine.state.iteration
            # log learning_rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(tag='learning_rate', scalar_value=current_lr, global_step=step)
            # log loss
            self.writer.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=step)
            # log all training losses
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(tag=f'train_{loss_name}', scalar_value=loss_value.item(), global_step=step)

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
