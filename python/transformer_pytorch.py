import math
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torchtext
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchtext.data.utils import get_tokenizer
import time

from torchtext.datasets import LanguageModelingDataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        pe: Tensor = torch.zeros(max_len, d_model)
        position: Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term: Tensor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5) -> None:
        super(TransformerModel, self).__init__()
        self.model_type: str = 'Transformer'
        self.src_mask: Tensor = None
        self.pos_encoder: PositionalEncoding = PositionalEncoding(ninp, dropout)
        encoder_layers: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder: nn.Embedding = nn.Embedding(ntoken, ninp)
        self.ninp: int = ninp
        self.decoder: nn.Linear = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, size: int) -> Tensor:
        mask: Tensor = torch.triu(torch.ones(size, size)).transpose(0, 1)
        mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(mask, float(0.0))
        return mask

    def init_weights(self):
        initrange: float = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output: Tensor = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


def batchify(dataset: LanguageModelingDataset, batch_size: int) -> Tensor:
    data: Tensor = TEXT.numericalize([dataset.examples[0].text])
    # Divide the dataset into batch_size parts.
    nbatch: int = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, idx: int) -> Tuple[Tensor, Tensor]:
    seq_len: int = min(bptt, len(source) - 1 - idx)
    data: Tensor = source[idx:idx + seq_len]
    target: Tensor = source[idx + 1:idx + 1 + seq_len].view(-1)
    return data, target


def train(epoch: int, train_data: Tensor):
    model.train()  # Turn on the train mode
    total_loss: float = 0.
    start_time: float = time.time()
    ntokens: int = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output: Tensor = model(data)
        loss: Tensor = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval: int = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss: float = total_loss / log_interval
            elapsed: float = time.time() - start_time
            print(f'| epoch {epoch} | {batch}/{len(train_data) // bptt} batches | ' +
                  f'lr {scheduler.get_lr()[0]} | ms/batch {elapsed * 1000 / log_interval} | ' +
                  f'loss {cur_loss} | ppl {math.exp(cur_loss)}')
            total_loss: float = 0
            start_time = time.time()


def evaluate(eval_model: TransformerModel, eval_data: Tensor) -> float:
    eval_model.eval()  # Turn on the evaluation mode
    total_loss: float = 0.
    ntokens: int = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            output: Tensor = eval_model(data)
            output_flat: Tensor = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


TEXT: torchtext.data.Field = torchtext.data.Field(
    tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>', lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size: int = 10
eval_batch_size: int = 5
train_data: Tensor = batchify(train_txt, batch_size)[:1000]
val_data: Tensor = batchify(val_txt, eval_batch_size)[:200]
test_data: Tensor = batchify(test_txt, eval_batch_size)[:200]
bptt: int = 35
ntokens: int = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize: int = 20  # embedding dimension 200
nhid: int = 20  # the dimension of the feedforward network model in nn.TransformerEncoder 200
nlayers: int = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder 2
nhead: int = 2  # the number of heads in the multiheadattention models 2
dropout: float = 0.1  # the dropout value 0.2
model: TransformerModel = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
lr: float = 0.5  # learning rate 5.0
optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler: _LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs: int = 10  # The number of epochs 3
best_model: TransformerModel = model
best_val_loss: float = sys.maxsize
print("Training...")

for epoch in range(1, epochs + 1):
    epoch_start_time: float = time.time()
    train(epoch, train_data)
    val_loss: float = evaluate(model, val_data)
    print('-' * 89)
    print(f'| end of epoch {epoch} | time: {time.time() - epoch_start_time}s | valid loss {val_loss} | ' +
          f'valid ppl {math.exp(val_loss)}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
test_loss: float = evaluate(best_model, test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss} | test ppl {math.exp(test_loss)}')
print('=' * 89)
