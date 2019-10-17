import torch
from torch import save, nn as nn
from torch.nn import functional as F


def make_nn_model(net, train_loader, criterion, optimizer, PATH):
    # GPUが使用できるときは使用する
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    # ニューラルネットの学習
    for epoch in range(2):

        runnning_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # データを画像データろラベルに分割
            inputs, labels = data

            # GPUに転送
            inputs, labels = inputs.to(device), labels.to(device)

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
