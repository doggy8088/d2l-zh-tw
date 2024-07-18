# 使用區塊的網路（VGG）
:label:`sec_vgg`

雖然AlexNet證明深層神經網路卓有成效，但它沒有提供一個通用的樣板來指導後續的研究人員設計新的網路。
在下面的幾個章節中，我們將介紹一些常用於設計深層神經網路的啟發式概念。

與晶片設計中工程師從放置晶體管到邏輯元件再到邏輯區塊的過程類似，神經網路架構的設計也逐漸變得更加抽象。研究人員開始從單個神經元的角度思考問題，發展到整個層，現在又轉向區塊，重複層的模式。

使用區塊的想法首先出現在牛津大學的[視覺幾何組（visual geometry group）](http://www.robots.ox.ac.uk/~vgg/)的*VGG網路*中。透過使用迴圈和子程式，可以很容易地在任何現代深度學習框架的程式碼中實現這些重複的架構。

## (**VGG區塊**)

經典卷積神經網路的基本組成部分是下面的這個序列：

1. 帶填充以保持解析度的卷積層；
1. 非線性啟用函式，如ReLU；
1. 彙聚層，如最大彙聚層。

而一個VGG區塊與之類似，由一系列卷積層組成，後面再加上用於空間下采樣的最大彙聚層。在最初的VGG論文中 :cite:`Simonyan.Zisserman.2014`，作者使用了帶有$3\times3$卷積核、填充為1（保持高度和寬度）的卷積層，和帶有$2 \times 2$彙聚視窗、步幅為2（每個區塊後的解析度減半）的最大彙聚層。在下面的程式碼中，我們定義了一個名為`vgg_block`的函式來實現一個VGG區塊。

:begin_tab:`mxnet,tensorflow`
該函式有兩個引數，分別對應於卷積層的數量`num_convs`和輸出通道的數量`num_channels`.
:end_tab:

:begin_tab:`pytorch`
該函式有三個引數，分別對應於卷積層的數量`num_convs`、輸入通道的數量`in_channels`
和輸出通道的數量`out_channels`.
:end_tab:

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2D(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

## [**VGG網路**]

與AlexNet、LeNet一樣，VGG網路可以分為兩部分：第一部分主要由卷積層和彙聚層組成，第二部分由全連線層組成。如 :numref:`fig_vgg`中所示。

![從AlexNet到VGG，它們本質上都是區塊設計。](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

VGG神經網路連線 :numref:`fig_vgg`的幾個VGG區塊（在`vgg_block`函式中定義）。其中有超引數變數`conv_arch`。該變數指定了每個VGG區塊裡卷積層個數和輸出通道數。全連線模組則與AlexNet中的相同。

原始VGG網路有5個卷積區塊，其中前兩個區塊各有一個卷積層，後三個區塊各包含兩個卷積層。
第一個模組有64個輸出通道，每個後續模組將輸出通道數量翻倍，直到該數字達到512。由於該網路使用8個卷積層和3個全連線層，因此它通常被稱為VGG-11。

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

下面的程式碼實現了VGG-11。可以透過在`conv_arch`上執行for迴圈來簡單實現。

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # 卷積層部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全連線層部分
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷積層部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全連線層部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # 卷積層部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全連線層部分
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab paddle
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷積層部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         # 全連線層部分
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 10))

net = vgg(conv_arch)
```

接下來，我們將建立一個高度和寬度為224的單通道資料樣本，以[**觀察每個層輸出的形狀**]。

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab paddle
X = paddle.randn(shape=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

正如從程式碼中所看到的，我們在每個區塊的高度和寬度減半，最終高度和寬度都為7。最後再展平表示，送入全連線層處理。

## 訓練模型

[**由於VGG-11比AlexNet計算量更大，因此我們建立了一個通道數較少的網路**]，足夠用於訓練Fashion-MNIST資料集。

```{.python .input}
#@tab mxnet, pytorch, paddle
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# 回想一下，這必須是一個將被放入『d2l.train_ch6()』的函式，為了利用我們現有的CPU/GPU裝置，這樣模型建立/編譯需要在strategy.scope()中
net = lambda: vgg(small_conv_arch)
```

除了使用略高的學習率外，[**模型訓練**]過程與 :numref:`sec_alexnet`中的AlexNet類似。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* VGG-11使用可複用的卷積區塊構造網路。不同的VGG模型可透過每個塊中卷積層數量和輸出通道數量的差異來定義。
* 區塊的使用導致網路定義的非常簡潔。使用區塊可以有效地設計複雜的網路。
* 在VGG論文中，Simonyan和Ziserman嘗試了各種架構。特別是他們發現深層且窄的卷積（即$3 \times 3$）比較淺層且寬的卷積更有效。

## 練習

1. 列印層的尺寸時，我們只看到8個結果，而不是11個結果。剩餘的3層訊息去哪了？
1. 與AlexNet相比，VGG的計算要慢得多，而且它還需要更多的顯存。分析出現這種情況的原因。
1. 嘗試將Fashion-MNIST資料集影象的高度和寬度從224改為96。這對實驗有什麼影響？
1. 請參考VGG論文 :cite:`Simonyan.Zisserman.2014`中的表1建立其他常見模型，如VGG-16或VGG-19。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1867)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1866)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1865)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11789)
:end_tab:
