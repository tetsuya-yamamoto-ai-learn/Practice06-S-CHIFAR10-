import urllib

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms


def set_proxy():
    proxy_support = urllib.request.ProxyHandler({'http': 'xxx.xxx.xxx.xx:xxxx'})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)


def set_CIFAR10_dataset():
    # proxyの設定
    set_proxy()
    # 画像変換の設定
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 訓練用のデータセットの準備
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    # 訓練用のデータローダーの準備
    trainloader = DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    # テスト用のデータセットの準備
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    # テスト用のデータローダーの準備
    testloader = DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes
