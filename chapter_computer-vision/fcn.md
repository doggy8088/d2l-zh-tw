# 全卷積網路
:label:`sec_fcn`

如 :numref:`sec_semantic_segmentation`中所介紹的那樣，語義分割是對影象中的每個畫素分類。
*全卷積網路*（fully convolutional network，FCN）採用卷積神經網路實現了從影象畫素到畫素類別的變換 :cite:`Long.Shelhamer.Darrell.2015`。
與我們之前在影象分類或目標檢測部分介紹的卷積神經網路不同，全卷積網路將中間層特徵圖的高和寬變換回輸入影象的尺寸：這是透過在 :numref:`sec_transposed_conv`中引入的*轉置卷積*（transposed convolution）實現的。
因此，輸出的類別預測與輸入影象在畫素級別上具有一一對應關係：通道維的輸出即該位置對應畫素的類別預測。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.nn import functional as F
import paddle.vision as paddlevision
```

## 構造模型

下面我們瞭解一下全卷積網路模型最基本的設計。
如 :numref:`fig_fcn`所示，全卷積網路先使用卷積神經網路抽取影象特徵，然後透過$1\times 1$卷積層將通道數變換為類別個數，最後在 :numref:`sec_transposed_conv`中透過轉置卷積層將特徵圖的高和寬變換為輸入影象的尺寸。
因此，模型輸出與輸入影象的高和寬相同，且最終輸出通道包含了該空間位置畫素的類別預測。

![全卷積網路](../img/fcn.svg)
:label:`fig_fcn`

下面，我們[**使用在ImageNet資料集上預訓練的ResNet-18模型來提取影象特徵**]，並將該網路記為`pretrained_net`。
ResNet-18模型的最後幾層包括全域平均彙聚層和全連線層，然而全卷積網路中不需要它們。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

```{.python .input}
#@tab paddle
pretrained_net = paddlevision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

接下來，我們[**建立一個全卷積網路`net`**]。
它複製了ResNet-18中大部分的預訓練層，除了最後的全域平均彙聚層和最接近輸出的全連線層。

```{.python .input}
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch, paddle
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

給定高度為320和寬度為480的輸入，`net`的前向傳播將輸入的高和寬減小至原來的$1/32$，即10和15。

```{.python .input}
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab paddle
X = paddle.rand(shape=(1, 3, 320, 480))
net(X).shape
```

接下來[**使用$1\times1$卷積層將輸出通道數轉換為Pascal VOC2012資料集的類數（21類）。**]
最後需要(**將特徵圖的高度和寬度增加32倍**)，從而將其變回輸入影象的高和寬。
回想一下 :numref:`sec_padding`中卷積層輸出形狀的計算方法：
由於$(320-64+16\times2+32)/32=10$且$(480-64+16\times2+32)/32=15$，我們構造一個步幅為$32$的轉置卷積層，並將卷積核的高和寬設為$64$，填充為$16$。
我們可以看到如果步幅為$s$，填充為$s/2$（假設$s/2$是整數）且卷積核的高和寬為$2s$，轉置卷積核會將輸入的高和寬分別放大$s$倍。

```{.python .input}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

```{.python .input}
#@tab paddle
num_classes = 21
net.add_sublayer('final_conv', nn.Conv2D(512, num_classes, kernel_size=1))
net.add_sublayer('transpose_conv', nn.Conv2DTranspose(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**初始化轉置卷積層**]

在影象處理中，我們有時需要將影象放大，即*上取樣*（upsampling）。
*雙線性內嵌*（bilinear interpolation）
是常用的上取樣方法之一，它也經常用於初始化轉置卷積層。

為了解釋雙線性內嵌，假設給定輸入影象，我們想要計算上取樣輸出影象上的每個畫素。

1. 將輸出影象的坐標$(x,y)$對映到輸入影象的坐標$(x',y')$上。
例如，根據輸入與輸出的尺寸之比來對映。
請注意，對映後的$x′$和$y′$是實數。
2. 在輸入影象上找到離坐標$(x',y')$最近的4個畫素。
3. 輸出影象在坐標$(x,y)$上的畫素依據輸入影象上這4個畫素及其與$(x',y')$的相對距離來計算。

雙線性內嵌的上取樣可以透過轉置卷積層實現，核心由以下`bilinear_kernel`函式構造。
限於篇幅，我們只給出`bilinear_kernel`函式的實現，不討論演算法的原理。

```{.python .input}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

```{.python .input}
#@tab paddle
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (paddle.arange(kernel_size).reshape([-1, 1]),
          paddle.arange(kernel_size).reshape([1, -1]))
    filt = (1 - paddle.abs(og[0] - center) / factor) * \
           (1 - paddle.abs(og[1] - center) / factor)
    weight = paddle.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

讓我們用[**雙線性內嵌的上取樣實驗**]它由轉置卷積層實現。
我們構造一個將輸入的高和寬放大2倍的轉置卷積層，並將其卷積核用`bilinear_kernel`函式初始化。

```{.python .input}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

```{.python .input}
#@tab paddle
conv_trans = nn.Conv2DTranspose(3, 3, kernel_size=4, padding=1, stride=2,
                                bias_attr=False)
conv_trans.weight.set_value(bilinear_kernel(3, 3, 4));
```

讀取影象`X`，將上取樣的結果記作`Y`。為了列印影象，我們需要調整通道維的位置。

```{.python .input}
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

```{.python .input}
#@tab paddle
img = paddlevision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].transpose([1, 2, 0]).detach()
```

可以看到，轉置卷積層將影象的高和寬分別放大了2倍。
除了坐標刻度不同，雙線性內嵌放大的影象和在 :numref:`sec_bbox`中列印出的原圖看上去沒什麼兩樣。

```{.python .input}
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

```{.python .input}
#@tab paddle
d2l.set_figsize()
print('input image shape:', img.transpose([1, 2, 0]).shape)
d2l.plt.imshow(img.transpose([1, 2, 0]));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

全卷積網路[**用雙線性內嵌的上取樣初始化轉置卷積層。對於$1\times 1$卷積層，我們使用Xavier初始化引數。**]

```{.python .input}
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

```{.python .input}
#@tab paddle
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.set_value(W);
```

## [**讀取資料集**]

我們用 :numref:`sec_semantic_segmentation`中介紹的語義分割讀取資料集。
指定隨機裁剪的輸出影象的形狀為$320\times 480$：高和寬都可以被$32$整除。

```{.python .input}
#@tab mxnet, pytorch
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

```{.python .input}
#@tab paddle
import os    
def load_data_voc(batch_size, crop_size):
    """載入VOC語義分割資料集
    Defined in :numref:`sec_semantic_segmentation`"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    train_iter = paddle.io.DataLoader(
        d2l.VOCSegDataset(True, crop_size, voc_dir), batch_size=batch_size,
        shuffle=True, return_list=True, drop_last=True, num_workers=0)
    test_iter = paddle.io.DataLoader(
        d2l.VOCSegDataset(False, crop_size, voc_dir), batch_size=batch_size,
        drop_last=True, return_list=True, num_workers=0)
    return train_iter, test_iter

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = load_data_voc(batch_size, crop_size)
```

## [**訓練**]

現在我們可以訓練全卷積網路了。
這裡的損失函式和準確率計算與影象分類中的並沒有本質上的不同，因為我們使用轉置卷積層的通道來預測畫素的類別，所以需要在損失計算中指定通道維。
此外，模型基於每個畫素的預測類別是否正確來計算準確率。

```{.python .input}
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab paddle
def loss(inputs, targets):
    return F.cross_entropy(inputs.transpose([0, 2, 3, 1]), targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = paddle.optimizer.SGD(learning_rate=lr, parameters=net.parameters(), weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices[:1])
```

## [**預測**]

在預測時，我們需要將輸入影象在各個通道做標準化，並轉成卷積神經網路所需要的四維輸入格式。

```{.python .input}
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab paddle
def predict(img):
    X = paddle.to_tensor(test_iter.dataset.normalize_image(img),dtype='float32').unsqueeze(0)
    pred = net(X).argmax(axis=1)
    return pred.reshape([pred.shape[1], pred.shape[2]])
```

為了[**視覺化預測的類別**]給每個畫素，我們將預測類別對映回它們在資料集中的標註顏色。

```{.python .input}
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

```{.python .input}
#@tab paddle
def label2image(pred):
    colormap = paddle.to_tensor(d2l.VOC_COLORMAP)
    X = pred.astype(paddle.int32)
    return colormap[X]
```

測試資料集中的影象大小和形狀各異。
由於模型使用了步幅為32的轉置卷積層，因此當輸入影象的高或寬無法被32整除時，轉置卷積層輸出的高或寬會與輸入影象的尺寸有偏差。
為了解決這個問題，我們可以在影象中擷取多區塊高和寬為32的整數倍的矩形區域，並分別對這些區域中的畫素做前向傳播。
請注意，這些區域的並集需要完整覆蓋輸入影象。
當一個畫素被多個區域所覆蓋時，它在不同區域前向傳播中轉置卷積層輸出的平均值可以作為`softmax`運算的輸入，從而預測類別。

為簡單起見，我們只讀取幾張較大的測試影象，並從影象的左上角開始擷取形狀為$320\times480$的區域用於預測。
對於這些測試影象，我們逐一列印它們擷取的區域，再列印預測結果，最後列印標註的類別。

```{.python .input}
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab paddle
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = paddlevision.transforms.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.transpose([1,2,0]).astype('uint8'), pred,
             paddlevision.transforms.crop(
                 test_labels[i], *crop_rect).transpose([1, 2, 0]).astype("uint8")]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## 小結

* 全卷積網路先使用卷積神經網路抽取影象特徵，然後透過$1\times 1$卷積層將通道數變換為類別個數，最後透過轉置卷積層將特徵圖的高和寬變換為輸入影象的尺寸。
* 在全卷積網路中，我們可以將轉置卷積層初始化為雙線性內嵌的上取樣。

## 練習

1. 如果將轉置卷積層改用Xavier隨機初始化，結果有什麼變化？
1. 調節超引數，能進一步提升模型的精度嗎？
1. 預測測試影象中所有畫素的類別。
1. 最初的全卷積網路的論文中 :cite:`Long.Shelhamer.Darrell.2015`還使用了某些卷積神經網路中間層的輸出。試著實現這個想法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3298)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3297)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11811)
:end_tab:
