import torch
import torch.nn as nn
from torch import load, optim
from torchvision.utils import make_grid

from MyNetwork import make_nn_model, Net
from calc_accuracy import class_test_data_accuracy, all_test_data_accuracy
from dataset import set_CIFAR10_dataset
from my_imshow import imshow


def main():
    # dataloader(dataset)の準備
    train_loader, test_loader, classes = set_CIFAR10_dataset()

    # 画像の取得
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # 画像の表示
    imshow(make_grid(images))

    # ラベルの表示
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # ニューラルネットの準備
    net = Net()

    # 損失関数と最適化アルゴリズムの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # モデルの学習と書き出し
    PATH = './cifar_net.pth'
    make_nn_model(net, train_loader, criterion, optimizer, PATH)

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

    # 全データにおける正答率の計算と表示
    correct, total = all_test_data_accuracy(net, test_loader)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}')

    # クラスごとにおける正答率の計算と表示
    class_correct, class_total = class_test_data_accuracy(net, test_loader)
    for i in range(10):
        print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:2.0f} %')


if __name__ == '__main__':
    main()
