import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import save, load
from torchvision.utils import make_grid

from dataset import set_CIFAR10_dataset
from my_imshow import imshow


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_nn_model(net, train_loader, criterion, optimizer, PATH):
    # ニューラルネットの学習
    for epoch in range(2):

        runnning_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # データを画像データろラベルに分割
            inputs, labels = data

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝搬 + 逆伝搬 + 最適化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 損失の表示
            runnning_loss += loss.item()
            if i % 2000 == 1999:  # 2000のミニバッチごとに表示を行う
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {runnning_loss / 2000:.3f}')
                runnning_loss = 0.0
    print('Finished Training')

    # モデルの書き出し
    save(net.state_dict(), PATH)


def main():
    # dataloader(dataset)の準備
    train_loader, test_loader, classes = set_CIFAR10_dataset()

    # # 画像の取得
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # # 画像の表示
    # imshow(make_grid(images))
    #
    # # ラベルの表示
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # # ニューラルネットの準備
    # net = Net()
    #
    # # 損失関数と最適化アルゴリズムの設定
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # モデルの学習と書き出し
    PATH = './cifar_net.pth'
    # make_nn_model(net, train_loader, criterion, optimizer, PATH)

    # テストデータの取り出し
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # テスト画像を表示
    imshow(make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:.5}s' for j in range(4)))

    # モデルの読み込み
    net = Net()
    net.load_state_dict(load(PATH))

    # モデルによる予測
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted', ' '.join(f'{classes[predicted[j]]}' for j in range(4)))

    # 全体のテストデータに対しての精度を表示


if __name__ == '__main__':
    main()
