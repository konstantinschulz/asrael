from typing import List, Tuple, Dict
import torch.nn.functional as F

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy, Loss
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import SubsetRandomSampler, DataLoader

from datasets import CorpusCorporum


class ConcatPoolingGRUAdaptive(torch.nn.Module):
    def __init__(self, embedding_dim: int, n_hidden: int, n_out: int, device: torch.device,
                 dropout: float, dataset_path: str, dataset_max_size: int, batch_size: int, validation_split: float,
                 shuffle_dataset: bool, random_seed: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.h: torch.Tensor
        self.device: torch.device = device
        self.dataset_path: str = dataset_path
        self.dataset_max_size: int = dataset_max_size
        self.batch_size: int = batch_size
        self.validation_split: float = validation_split
        self.shuffle_dataset = shuffle_dataset
        self.random_seed: int = random_seed
        self.train_dl, self.val_dl, self.dataset = self.build_data_loaders()
        self.model_vocab_size: int = self.dataset.vocab_size + 2
        self.emb = torch.nn.Embedding(self.model_vocab_size, self.embedding_dim)
        self.emb_drop = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(self.embedding_dim, self.n_hidden, dropout=dropout)
        self.out = torch.nn.Linear(self.n_hidden * 3, self.n_out)
        self.trainer: Engine = self.build_trainer()

    @staticmethod
    def track_progress(train_evaluator: Engine, validation_evaluator: Engine, loss_fn: callable, trainer: Engine):
        def max_output_transform(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            """It converts the predicted output probabilties to indices for accuracy calculation"""
            y_pred, y = output
            return torch.max(y_pred, dim=1)[1], y

        # attach running loss (will be displayed in progess bar)
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        # attach running accuracy (will be displayed in progess bar)
        RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]])).attach(trainer, 'acc')
        # attach accuracy and loss to train_evaluator
        Accuracy(output_transform=max_output_transform).attach(train_evaluator, 'accuracy')
        Loss(loss_fn).attach(train_evaluator, 'bce')
        # attach accuracy and loss to validation_evaluator
        Accuracy(output_transform=max_output_transform).attach(validation_evaluator, 'accuracy')
        Loss(loss_fn).attach(validation_evaluator, 'bce')

    def build_data_loaders(self) -> Tuple[DataLoader, DataLoader, CorpusCorporum]:
        print("Building dataset...")
        dataset: CorpusCorporum = CorpusCorporum(self.dataset_path, self.dataset_max_size)
        print(f"Vocabulary size: {dataset.vocab_size}")
        # Creating data indices for training and validation splits:
        indices: List[int] = list(range(dataset.sent_count))
        split: int = int(np.floor(self.validation_split * dataset.sent_count))
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_indices)
        valid_sampler: SubsetRandomSampler = SubsetRandomSampler(val_indices)
        train_dl = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                               collate_fn=CorpusCorporum.collate_fn)
        val_dl = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                             collate_fn=CorpusCorporum.collate_fn)
        return train_dl, val_dl, dataset

    def build_trainer(self) -> Engine:
        loss_fn: callable = F.nll_loss
        optimizer: torch.optim.Adam = torch.optim.Adam(self.parameters(), 1e-3)
        model = self

        def process_function(engine: Engine, batch: Tuple[torch.Tensor, torch.Tensor, List[int]]) -> \
                Tuple[float, torch.Tensor, torch.Tensor]:
            """Single training loop to be attached to trainer Engine"""
            model.train()
            optimizer.zero_grad()
            x, y, lengths = batch
            x, y = x.to(model.device), y.to(model.device)
            y_pred: torch.Tensor = model(x, lengths)
            loss: torch.Tensor = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            return loss.item(), torch.max(y_pred, dim=1)[1], y

        def eval_function(engine: Engine, batch: Tuple[torch.Tensor, torch.Tensor, List[int]]) -> \
                Tuple[torch.Tensor, torch.Tensor]:
            """Single evaluator loop to be attached to trainer and evaluator Engine"""
            model.eval()
            with torch.no_grad():
                x, y, lengths = batch
                x, y = x.to(model.device), y.to(model.device)
                y_pred: torch.Tensor = model(x, lengths)
                return y_pred, y

        trainer: Engine = Engine(process_function)
        train_evaluator: Engine = Engine(eval_function)
        validation_evaluator: Engine = Engine(eval_function)
        ConcatPoolingGRUAdaptive.track_progress(train_evaluator, validation_evaluator, loss_fn, trainer)
        pbar = ProgressBar(persist=True, bar_format="")
        pbar.attach(trainer, ['loss', 'acc'])
        self.log_results(train_evaluator, validation_evaluator, pbar, trainer)
        return trainer

    def forward(self, seq: torch.Tensor, lengths: List[int]):
        self.h = self.init_hidden(seq.size(1))
        embs_tensor: torch.Tensor = self.emb_drop(self.emb(seq))
        embs: PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(embs_tensor, lengths)
        gru_out, self.h = self.gru(embs, self.h)
        gru_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(gru_out)
        avg_pool: torch.Tensor = F.adaptive_avg_pool1d(gru_out.permute(1, 2, 0), 1).view(seq.size(1), -1)
        max_pool: torch.Tensor = F.adaptive_max_pool1d(gru_out.permute(1, 2, 0), 1).view(seq.size(1), -1)
        outp: torch.Tensor = self.out(torch.cat([self.h[-1], avg_pool, max_pool], dim=1))
        return F.log_softmax(outp, dim=-1)  # it will return log of softmax

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros((1, batch_size, self.n_hidden), requires_grad=True).to(self.device)

    def log_results(self, train_evaluator: Engine, validation_evaluator: Engine, pbar: ProgressBar, trainer: Engine):
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine: Engine):
            """This function will run after each epoch and report the training loss and accuracy (defined above)"""
            train_evaluator.run(self.train_dl)
            metrics: Dict[str, float] = train_evaluator.state.metrics
            avg_accuracy: float = metrics['accuracy']
            avg_bce: float = metrics['bce']
            pbar.log_message(
                f'Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_bce:.4f}')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            """This function will run after each epoch and report the validation loss and accuracy (defined above)"""
            validation_evaluator.run(self.val_dl)
            metrics: Dict[str, float] = validation_evaluator.state.metrics
            avg_accuracy: float = metrics['accuracy']
            avg_bce: float = metrics['bce']
            pbar.log_message(
                f'Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_bce:.4f}')
            pbar.n = pbar.last_print_n = 0
