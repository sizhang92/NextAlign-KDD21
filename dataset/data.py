import torch
from torch.utils.data import Dataset


class Train_Data(Dataset):
    def __init__(self, context_pairs1, context_pairs2):
        # data loading
        self.context_pairs1 = torch.from_numpy(context_pairs1).long()
        self.context_pairs2 = torch.from_numpy(context_pairs2).long()
        self.n_samples = context_pairs1.shape[0]

    def __getitem__(self, index):
        return self.context_pairs1[index], self.context_pairs2[index]

    def __len__(self):
        return self.n_samples


class Test_Data(Dataset):
    def __init__(self, test_pairs):
        # data loading
        self.nodes1 = torch.from_numpy(test_pairs[:, 0]).long()
        self.nodes2 = torch.from_numpy(test_pairs[:, 1]).long()
        self.n_samples = self.nodes1.shape[0]

    def __getitem__(self, index):
        return self.nodes1[index], self.nodes2[index]

    def __len__(self):
        return self.n_samples