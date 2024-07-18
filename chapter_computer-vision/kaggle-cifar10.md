# 實戰 Kaggle 比賽：影象分類 (CIFAR-10)
:label:`sec_kaggle_cifar10`

之前幾節中，我們一直在使用深度學習框架的高階API直接獲取張量格式的影象資料集。
但是在實踐中，影象資料集通常以影象檔案的形式出現。
本節將從原始影象檔案開始，然後逐步組織、讀取並將它們轉換為張量格式。

我們在 :numref:`sec_image_augmentation`中對CIFAR-10資料集做了一個實驗。CIFAR-10是電腦視覺領域中的一個重要的資料集。
本節將運用我們在前幾節中學到的知識來參加CIFAR-10影象分類問題的Kaggle競賽，(**比賽的網址是https://www.kaggle.com/c/cifar-10**)。

 :numref:`fig_kaggle_cifar10`顯示了競賽網站頁面上的訊息。
為了能送出結果，首先需要註冊一個Kaggle賬戶。

![CIFAR-10 影象分類競賽頁面上的訊息。競賽用的資料集可透過點選『Data』頁籤獲取。](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`

首先，匯入競賽所需的套件和模組。

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import collections
import math
import os
import pandas as pd
import shutil
import paddle
from paddle import nn
import paddle.vision as paddlevision
```

## 獲取並組織資料集

比賽資料集分為訓練集和測試集，其中訓練集包含50000張、測試集包含300000張影象。
在測試集中，10000張影象將被用於評估，而剩下的290000張影象將不會被進行評估，包含它們只是為了防止手動標記測試集並送出標記結果。
兩個資料集中的影象都是png格式，高度和寬度均為32畫素並有三個顏色通道（RGB）。
這些圖片共涵蓋10個類別：飛機、汽車、鳥類、貓、鹿、狗、青蛙、馬、船和卡車。
 :numref:`fig_kaggle_cifar10`的左上角顯示了資料集中飛機、汽車和鳥類的一些影象。

### 下載資料集

登入Kaggle後，我們可以點選 :numref:`fig_kaggle_cifar10`中顯示的CIFAR-10影象分類競賽網頁上的『Data』頁籤，然後單擊『Download All』按鈕下載資料集。
在`../data`中解壓下載的檔案並在其中解壓縮`train.7z`和`test.7z`後，在以下路徑中可以找到整個資料集：

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

`train`和`test`資料夾分別包含訓練和測試影象，`trainLabels.csv`含有訓練影象的標籤，
`sample_submission.csv`是送出檔案的範例。

為了便於入門，[**我們提供包含前1000個訓練影象和5個隨機測試影象的資料集的小規模樣本**]。
要使用Kaggle競賽的完整資料集，需要將以下`demo`變數設定為`False`。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# 如果使用完整的Kaggle競賽的資料集，設定demo為False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**整理資料集**]

我們需要整理資料集來訓練和測試模型。
首先，我們用以下函式讀取CSV檔案中的標籤，它回傳一個字典，該字典將檔案名中不帶副檔名的部分對映到其標籤。

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """讀取fname來給標籤字典回傳一個檔案名"""
    with open(fname, 'r') as f:
        # 跳過檔案頭行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 訓練樣本 :', len(labels))
print('# 類別 :', len(set(labels.values())))
```

接下來，我們定義`reorg_train_valid`函式來[**將驗證集從原始的訓練集中拆分出來**]。
此函式中的引數`valid_ratio`是驗證集中的樣本數與原始訓練集中的樣本數之比。
更具體地說，令$n$等於樣本最少的類別中的影象數量，而$r$是比率。
驗證集將為每個類別拆分出$\max(\lfloor nr\rfloor,1)$張影象。
讓我們以`valid_ratio=0.1`為例，由於原始的訓練集有50000張影象，因此`train_valid_test/train`路徑中將有45000張影象用於訓練，而剩下5000張影象將作為路徑`train_valid_test/valid`中的驗證集。
組織資料集後，同類別的影象將被放置在同一資料夾下。

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """將檔案複製到目標目錄"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """將驗證集從原始的訓練集中拆分出來"""
    # 訓練資料集中樣本最少的類別中的樣本數
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 驗證集中每個類別的樣本數
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

下面的`reorg_test`函式用來[**在預測期間整理測試集，以方便讀取**]。

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """在預測期間整理測試集，以方便讀取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

最後，我們使用一個函式來[**呼叫前面定義的函式**]`read_csv_labels`、`reorg_train_valid`和`reorg_test`。

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

在這裡，我們只將樣本數據集的批次大小設定為32。
在實際訓練和測試中，應該使用Kaggle競賽的完整資料集，並將`batch_size`設定為更大的整數，例如128。
我們將10％的訓練樣本作為調整超引數的驗證集。

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**影象增廣**]

我們使用影象增廣來解決過擬合的問題。例如在訓練中，我們可以隨機水平翻轉影象。
我們還可以對彩色影象的三個RGB通道執行標準化。
下面，我們列出了其中一些可以調整的操作。

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # 在高度和寬度上將影象放大到40畫素的正方形
    gluon.data.vision.transforms.Resize(40),
    # 隨機裁剪出一個高度和寬度均為40畫素的正方形影象，
    # 生成一個面積為原始影象面積0.64～1倍的小正方形，
    # 然後將其縮放為高度和寬度均為32畫素的正方形
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # 標準化影象的每個通道
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # 在高度和寬度上將影象放大到40畫素的正方形
    torchvision.transforms.Resize(40),
    # 隨機裁剪出一個高度和寬度均為40畫素的正方形影象，
    # 生成一個面積為原始影象面積0.64～1倍的小正方形，
    # 然後將其縮放為高度和寬度均為32畫素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 標準化影象的每個通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab paddle
transform_train = paddlevision.transforms.Compose([
    # 在高度和寬度上將影象放大到40畫素的正方形
    paddlevision.transforms.Resize(40),
    # 隨機裁剪出一個高度和寬度均為40畫素的正方形影象，
    # 生成一個面積為原始影象面積0.64到1倍的小正方形，
    # 然後將其縮放為高度和寬度均為32畫素的正方形
    paddlevision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                              ratio=(1.0, 1.0)),
    paddlevision.transforms.RandomHorizontalFlip(),
    paddlevision.transforms.ToTensor(),
    # 標準化影象的每個通道
    paddlevision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

在測試期間，我們只對影象執行標準化，以消除評估結果中的隨機性。

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab paddle
transform_test = paddlevision.transforms.Compose([
    paddlevision.transforms.ToTensor(),
    paddlevision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## 讀取資料集

接下來，我們[**讀取由原始影象組成的資料集**]，每個樣本都包括一張圖片和一個標籤。

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

```{.python .input}
#@tab paddle
train_ds, train_valid_ds = [paddlevision.datasets.DatasetFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]
    
valid_ds, test_ds = [paddlevision.datasets.DatasetFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

在訓練期間，我們需要[**指定上面定義的所有影象增廣操作**]。
當驗證集在超引數調整過程中用於模型評估時，不應引入影象增廣的隨機性。
在最終預測之前，我們根據訓練集和驗證集組合而成的訓練模型進行訓練，以充分利用所有標記的資料。

```{.python .input}
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

```{.python .input}
#@tab paddle
train_iter, train_valid_iter = [paddle.io.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = paddle.io.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                                  drop_last=True)

test_iter = paddle.io.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 drop_last=False)
```

## 定義[**模型**]

:begin_tab:`mxnet`
在這裡，我們基於`HybridBlock`類建立剩餘區塊，這與 :numref:`sec_resnet`中描述的實現方法略有不同，是為了提高計算效率。
:end_tab:

```{.python .input}
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
接下來，我們定義Resnet-18模型。
:end_tab:

```{.python .input}
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
在訓練開始之前，我們使用 :numref:`subsec_xavier`中描述的Xavier初始化。
:end_tab:

:begin_tab:`pytorch`
我們定義了 :numref:`sec_resnet`中描述的Resnet-18模型。
:end_tab:

:begin_tab:`paddle`
我們定義了 :numref:`sec_resnet`中描述的Resnet-18模型。
:end_tab:

```{.python .input}
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

```{.python .input}
#@tab paddle
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## 定義[**訓練函式**]

我們將根據模型在驗證集上的表現來選擇模型並調整超引數。
下面我們定義了模型訓練函式`train`。

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab paddle
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    scheduler = paddle.optimizer.lr.StepDecay(lr, lr_period, lr_decay)
    trainer = paddle.optimizer.Momentum(learning_rate=scheduler, momentum=0.9, parameters=net.parameters(),
                              weight_decay=wd)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = paddle.DataParallel(net)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**訓練和驗證模型**]

現在，我們可以訓練和驗證模型了，而以下所有超引數都可以調整。
例如，我們可以增加週期的數量。當`lr_period`和`lr_decay`分別設定為4和0.9時，最佳化演算法的學習速率將在每4個週期乘以0.9。
為便於示範，我們在這裡只訓練20個週期。

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab paddle
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## 在 Kaggle 上[**對測試集進行分類並送出結果**]

在獲得具有超引數的滿意的模型後，我們使用所有標記的資料（包括驗證集）來重新訓練模型並對測試集進行分類。

```{.python .input}
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab paddle
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X)
    preds.extend(y_hat.argmax(axis=1).astype(paddle.int32).numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

向Kaggle送出結果的方法與 :numref:`sec_kaggle_house`中的方法類似，上面的程式碼將生成一個
`submission.csv`檔案，其格式符合Kaggle競賽的要求。

## 小結

* 將包含原始影象檔案的資料集組織為所需格式後，我們可以讀取它們。

:begin_tab:`mxnet`
* 我們可以在影象分類競賽中使用卷積神經網路、影象增廣和混合程式設計。
:end_tab:

:begin_tab:`pytorch`
* 我們可以在影象分類競賽中使用卷積神經網路和影象增廣。
:end_tab:

:begin_tab:`paddle`
* 我們可以在影象分類競賽中使用卷積神經網路和影象增廣。
:end_tab:

## 練習

1. 在這場Kaggle競賽中使用完整的CIFAR-10資料集。將超引數設為`batch_size = 128`，`num_epochs = 100`，`lr = 0.1`，`lr_period = 50`，`lr_decay = 0.1`。看看在這場比賽中能達到什麼準確度和排名。能進一步改進嗎？
1. 不使用影象增廣時，能獲得怎樣的準確度？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2830)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2831)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11814)
:end_tab:
