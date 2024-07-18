# 實戰Kaggle比賽：狗的品種識別（ImageNet Dogs）

本節我們將在Kaggle上實戰狗品種識別問題。
本次(**比賽網址是https://www.kaggle.com/c/dog-breed-identification**)。
 :numref:`fig_kaggle_dog`顯示了鑑定比賽網頁上的訊息。
需要一個Kaggle賬戶才能送出結果。

在這場比賽中，我們將識別120類不同品種的狗。
這個資料集實際上是著名的ImageNet的資料集子集。與 :numref:`sec_kaggle_cifar10`中CIFAR-10資料集中的影象不同，
ImageNet資料集中的影象更高更寬，且尺寸不一。

![狗的品種鑑定比賽網站，可以透過單擊『資料』頁籤來獲得比賽資料集。](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.vision as paddlevision
from paddle import nn
import os
```

## 獲取和整理資料集

比賽資料集分為訓練集和測試集，分別包含RGB（彩色）通道的10222張、10357張JPEG影象。
在訓練資料集中，有120種犬類，如拉布拉多、貴賓、臘腸、薩摩耶、哈士奇、吉娃娃和約克夏等。

### 下載資料集

登入Kaggle後，可以點選 :numref:`fig_kaggle_dog`中顯示的競爭網頁上的『資料』頁籤，然後點選『全部下載』按鈕下載資料集。在`../data`中解壓下載的檔案後，將在以下路徑中找到整個資料集：

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test


上述結構與 :numref:`sec_kaggle_cifar10`的CIFAR-10類似，其中資料夾`train/`和`test/`分別包含訓練和測試狗影象，`labels.csv`包含訓練影象的標籤。

同樣，為了便於入門，[**我們提供完整資料集的小規模樣本**]：`train_valid_test_tiny.zip`。
如果要在Kaggle比賽中使用完整的資料集，則需要將下面的`demo`變數更改為`False`。

```{.python .input}
#@tab all
#@save 
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# 如果使用Kaggle比賽的完整資料集，請將下面的變數更改為False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**整理資料集**]

我們可以像 :numref:`sec_kaggle_cifar10`中所做的那樣整理資料集，即從原始訓練集中拆分驗證集，然後將影象移動到按標籤分組的子資料夾中。

下面的`reorg_dog_data`函式讀取訓練資料標籤、拆分驗證集並整理訓練集。

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**影象增廣**]

回想一下，這個狗品種資料集是ImageNet資料集的子集，其影象大於 :numref:`sec_kaggle_cifar10`中CIFAR-10資料集的影象。
下面我們看一下如何在相對較大的影象上使用影象增廣。

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # 隨機裁剪影象，所得影象為原始面積的0.08～1之間，高寬比在3/4和4/3之間。
    # 然後，縮放影象以建立224x224的新影象
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # 隨機更改亮度，對比度和飽和度
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # 新增隨機噪聲
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # 標準化影象的每個通道
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # 隨機裁剪影象，所得影象為原始面積的0.08～1之間，高寬比在3/4和4/3之間。
    # 然後，縮放影象以建立224x224的新影象
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 隨機更改亮度，對比度和飽和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 新增隨機噪聲
    torchvision.transforms.ToTensor(),
    # 標準化影象的每個通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab paddle
transform_train = paddlevision.transforms.Compose([
    # 隨機裁剪影象，所得影象為原始面積的0.08到1之間，高寬比在3/4和4/3之間。
    # 然後，縮放影象以建立224x224的新影象
    paddlevision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    paddlevision.transforms.RandomHorizontalFlip(),
    # 隨機更改亮度，對比度和飽和度
    paddlevision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 新增隨機噪聲
    paddlevision.transforms.ToTensor(),
    # 標準化影象的每個通道
    paddlevision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

測試時，我們只使用確定性的影象預處理操作。

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # 從影象中心裁切224x224大小的圖片
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 從影象中心裁切224x224大小的圖片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab paddle
transform_test = paddlevision.transforms.Compose([
    paddlevision.transforms.Resize(256),
    # 從影象中心裁切224x224大小的圖片
    paddlevision.transforms.CenterCrop(224),
    paddlevision.transforms.ToTensor(),
    paddlevision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**讀取資料集**]

與 :numref:`sec_kaggle_cifar10`一樣，我們可以讀取整理後的含原始影象檔案的資料集。

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

下面我們建立資料載入器實例的方式與 :numref:`sec_kaggle_cifar10`相同。

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

## [**微調預訓練模型**]

同樣，本次比賽的資料集是ImageNet資料集的子集。
因此，我們可以使用 :numref:`sec_fine_tuning`中討論的方法在完整ImageNet資料集上選擇預訓練的模型，然後使用該模型提取影象特徵，以便將其輸入到定製的小規模輸出網路中。
深度學習框架的高階API提供了在ImageNet資料集上預訓練的各種模型。
在這裡，我們選擇預訓練的ResNet-34模型，我們只需重複使用此模型的輸出層（即提取的特徵）的輸入。
然後，我們可以用一個可以訓練的小型自定義輸出網路替換原始輸出層，例如堆疊兩個完全連線的圖層。
與 :numref:`sec_fine_tuning`中的實驗不同，以下內容不重新訓練用於特徵提取的預訓練模型，這節省了梯度下降的時間和記憶體空間。

回想一下，我們使用三個RGB通道的均值和標準差來對完整的ImageNet資料集進行影象標準化。
事實上，這也符合ImageNet上預訓練模型的標準化操作。

```{.python .input}
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # 定義一個新的輸出網路
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 共有120個輸出類別
    finetune_net.output_new.add(nn.Dense(120))
    # 初始化輸出網路
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # 將模型引數分配給用於計算的CPU或GPU
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定義一個新的輸出網路，共有120個輸出類別
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 將模型引數分配給用於計算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 凍結引數
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

```{.python .input}
#@tab paddle
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = paddlevision.models.resnet34(pretrained=True)
    # 定義一個新的輸出網路，共有120個輸出類別
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 凍結引數
    for param in finetune_net.features.parameters():
        param.stop_gradient = True
    return finetune_net
```

在[**計算損失**]之前，我們首先獲取預訓練模型的輸出層的輸入，即提取的特徵。
然後我們使用此特徵作為我們小型自定義輸出網路的輸入來計算損失。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')
```

```{.python .input}
#@tab paddle
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## 定義[**訓練函式**]

我們將根據模型在驗證集上的表現選擇模型並調整超引數。
模型訓練函式`train`只迭代小型自定義輸出網路的引數。

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 只訓練小型自定義輸出網路
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 只訓練小型自定義輸出網路
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab paddle
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 只訓練小型自定義輸出網路
    net = paddle.DataParallel(net)
    scheduler = paddle.optimizer.lr.StepDecay(lr, lr_period, lr_decay)
    trainer = paddle.optimizer.Momentum(learning_rate=scheduler, 
                                        parameters=(param for param in net.parameters() if not param.stop_gradient), 
                                        momentum=0.9, 
                                        weight_decay=wd)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            trainer.clear_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {float(valid_loss):.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**訓練和驗證模型**]

現在我們可以訓練和驗證模型了，以下超引數都是可調的。
例如，我們可以增加迭代輪數。
另外，由於`lr_period`和`lr_decay`分別設定為2和0.9，
因此最佳化演算法的學習速率將在每2個迭代後乘以0.9。

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab paddle
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**對測試集分類**]並在Kaggle送出結果

與 :numref:`sec_kaggle_cifar10`中的最後一步類似，最終所有標記的資料（包括驗證集）都用於訓練模型和對測試集進行分類。
我們將使用訓練好的自定義輸出網路進行分類。

```{.python .input}
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab paddle
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = paddle.nn.functional.softmax(net(data), axis=0)
    preds.extend(output.detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

上面的程式碼將生成一個`submission.csv`檔案，以 :numref:`sec_kaggle_house`中描述的方式提在Kaggle上送出。

## 小結

* ImageNet資料集中的影象比CIFAR-10影象尺寸大，我們可能會修改不同資料集上任務的影象增廣操作。
* 要對ImageNet資料集的子集進行分類，我們可以利用完整ImageNet資料集上的預訓練模型來提取特徵並僅訓練小型自定義輸出網路，這將減少計算時間和節省記憶體空間。

## 練習

1. 試試使用完整Kaggle比賽資料集，增加`batch_size`（批次大小）和`num_epochs`（迭代輪數），或者設計其它超引數為`lr = 0.01`，`lr_period = 10`，和`lr_decay = 0.1`時，能取得什麼結果？
1. 如果使用更深的預訓練模型，會得到更好的結果嗎？如何調整超引數？能進一步改善結果嗎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2832)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2833)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11815)
:end_tab:
