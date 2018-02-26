from torchvision import datasets, transforms
import torch


def load_svhn(root, train_batch_size, test_batch_size=None):
    train = datasets.SVHN(root, split='train', download=True, transform=transforms.ToTensor())
    test = datasets.SVHN(root, split='test', download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=2)

    return train_loader, test_loader


def load_mnist(root, train_batch_size, test_batch_size=None):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
        ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=2)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape


def load_cifar(root, train_batch_size, test_batch_size=None):
    train = datasets.CIFAR10(root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=2)

    shape = train_loader.dataset.train_data.transpose(0,3,1,2).shape[1:]

    return train_loader, test_loader, shape
