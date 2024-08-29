import time

import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(512, 1)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            print("loading idx {} from worker {}".format(idx, worker_info.id))
        x = self.data[idx]
        return x

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = MyDataset()
    loader = DataLoader(dataset, prefetch_factor=2, num_workers=8, batch_size=2)

    loader_iter = iter(loader)  # preloading starts here
    # with the default prefetch_factor of 2, 2*num_workers=16 batches will be preloaded
    # the max index printed by __getitem__ is thus 31 (16*batch_size=32 samples loaded)

    time.sleep(3)
    print("Begin iteration")

    time.sleep(1)
    data = next(
        loader_iter
    )  # this will consume a batch and preload the next one from a single worker to fill the queue
    # batch_size=2 new samples should be loaded
    print(data.shape)

    time.sleep(1)
    data = next(loader_iter)
    print(data.shape)

    time.sleep(1)
    data = next(loader_iter)
    print(data.shape)

    time.sleep(1)
    data = next(loader_iter)
    print(data.shape)
