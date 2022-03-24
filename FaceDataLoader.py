from torch.utils import data
from torchvision import transforms

from FaceDataset import FaceDataset


class FaceDataLoader():
    def __init__(self, data_dir, set, component, batch_size, n_workers, shuffle):
        self.data_dir = data_dir
        self.set = set
        self.component = component
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.shuffle = shuffle
        self.trans = transforms.Compose([
            transforms.Pad((10, 0, 10, 0)),
            transforms.ToTensor()
        ])

    def get_data_loader(self):
        dataset = FaceDataset(self.data_dir, self.set, self.component, self.trans)
        data_loader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
            drop_last=True,
            pin_memory=True
        )

        print('\ndata len:', dataset.__len__(), '\n')
        return data_loader
