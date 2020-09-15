from collections import Counter
from itertools import islice
from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CorpusCorporum(Dataset):
    def __init__(self, dataset_path: str, dataset_max_size: int = 0) -> None:
        self.dataset_path: str = dataset_path
        self.sent_count: int = 0
        words_counter = Counter()
        with open(self.dataset_path) as f:
            line: str = f.readline()
            while line:
                self.sent_count += 1
                if self.sent_count == dataset_max_size:
                    # print(self.sent_count)
                    break
                words_counter.update(tok for tok in line.strip()[:-1].lower().split())
                line = f.readline()
        self.PAD: int = 0
        self.UNK: int = 1
        self.word2idx: Dict[str, int] = {'<PAD>': self.PAD, '<UNK>': self.UNK}
        self.word2idx.update({word: i + 2 for i, (word, count) in tqdm(enumerate(words_counter.most_common()))})
        self.vocab_size: int = len(words_counter)

    def __len__(self) -> int:
        return self.sent_count

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        line: str
        with open(self.dataset_path) as f:
            line = next(islice(f, index, index + 1))
        line_parts: List[str] = line.split("\t")
        tokens: List[str] = [x for x in line_parts[0].lower().split()]
        vec: torch.Tensor = torch.tensor([self.word2idx.get(token, self.UNK) for token in tokens])
        return vec, int(line_parts[1])

    @staticmethod
    def collate_fn(data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """This function will be used to pad the sentences to max length in the batch and transpose the batch from
           batch_size x max_seq_len to max_seq_len x batch_size.
           It will return padded vectors, labels and lengths of each sentence (before padding)
           It will be used in the Dataloader.
        """
        data.sort(key=lambda x: len(x[0]), reverse=True)
        lengths: List[int] = [max(list(sent.shape)) for sent, label in data]
        labels: List[int] = []
        padded_sents: torch.Tensor = torch.zeros(len(data), max(lengths)).long()
        for i, (sent, label) in enumerate(data):
            padded_sents[i, :lengths[i]] = sent
            labels.append(label)
        padded_sents: torch.Tensor = padded_sents.transpose(0, 1)
        return padded_sents, torch.tensor(labels).long(), lengths
