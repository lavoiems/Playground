from torchvision import datasets, transforms
import torch


def load_svhn(root, train_batch_size, test_batch_size=None):
    train = datasets.SVHN(root, split='train', download=True, transform=transforms.ToTensor())
    test = datasets.SVHN(root, split='test', download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=2)

    return train_loader, test_loader
