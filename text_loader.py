# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    # Initialize your data, download, etc.

    def __init__(self, filename="./data/shakespeare.txt.gz"):
        self.len = 0
        with gzip.open(filename, 'rt') as f:
            self.targetLines = [x.strip() for x in f if x.strip()]
            self.srcLines = [torch.tensor([ord(c) for c in x.lower().replace(' ', '')])
                             for x in self.targetLines]
            self.len = len(self.srcLines)

    def __getitem__(self, index):
        return self.srcLines[index][:-1], self.srcLines[index][1:]

    def __len__(self):
        return self.len


def collate_fn(batch):
    src_list, target_list = [], []
    for src, target in batch:
        src_list.append(src)
        target_list.append(target)
    return pad_sequence(src_list, batch_first=True, padding_value=0), pad_sequence(target_list, batch_first=True, padding_value=0)


# Test the loader
if __name__ == "__main__":
    dataset = TextDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collate_fn)

    for i, (src, target) in enumerate(train_loader):
        print(i, "data", src)
