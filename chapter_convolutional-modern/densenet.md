# 稠密連線網路（DenseNet）

ResNet極大地改變瞭如何引數化深層網路中函式的觀點。
*稠密連線網路*（DenseNet） :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`在某種程度上是ResNet的邏輯擴充套件。讓我們先從數學上了解一下。

## 從ResNet到DenseNet

回想一下任意函式的泰勒展開式（Taylor expansion），它把這個函式分解成越來越高階的項。在$x$接近0時，

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

同樣，ResNet將函式展開為

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

也就是說，ResNet將$f$分解為兩部分：一個簡單的線性項和一個複雜的非線性項。
那麼再向前拓展一步，如果我們想將$f$拓展成超過兩部分的訊息呢？
一種方案便是DenseNet。

![ResNet（左）與 DenseNet（右）在跨層連線上的主要區別：使用相加和使用連結。](../img/densenet-block.svg)
:label:`fig_densenet_block`

如 :numref:`fig_densenet_block`所示，ResNet和DenseNet的關鍵區別在於，DenseNet輸出是*連線*（用圖中的$[,]$表示）而不是如ResNet的簡單相加。
因此，在應用越來越複雜的函式序列後，我們執行從$\mathbf{x}$到其展開式的對映：

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

最後，將這些展開式結合到多層感知機中，再次減少特徵的數量。
實現起來非常簡單：我們不需要新增術語，而是將它們連線起來。
DenseNet這個名字由變數之間的『稠密連線』而得來，最後一層與之前的所有層緊密相連。
稠密連線如 :numref:`fig_densenet`所示。

![稠密連線。](../img/densenet.svg)
:label:`fig_densenet`

稠密網路主要由2部分構成：*稠密區塊*（dense block）和*轉場層*（transition layer）。
前者定義如何連線輸入和輸出，而後者則控制通道數量，使其不會太複雜。

## (**稠密區塊體**)

DenseNet使用了ResNet改良版的『批次規範化、啟用和卷積』架構（參見 :numref:`sec_resnet`中的練習）。
我們首先實現一下這個架構。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2D(input_channels), nn.ReLU(),
        nn.Conv2D(input_channels, num_channels, kernel_size=3, padding=1))
```

一個*稠密區塊*由多個卷積區塊組成，每個卷積區塊使用相同數量的輸出通道。
然而，在前向傳播中，我們將每個卷積區塊的輸入和輸出在通道維上連結。

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 連線通道維度上每個區塊的輸入和輸出
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 連線通道維度上每個區塊的輸入和輸出
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

```{.python .input}
#@tab paddle
class DenseBlock(nn.Layer):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 連線通道維度上每個區塊的輸入和輸出
            X = paddle.concat(x=[X, Y], axis=1)
        return X
```

在下面的例子中，我們[**定義一個**]有2個輸出通道數為10的(**`DenseBlock`**)。
使用通道數為3的輸入時，我們會得到通道數為$3+2\times 10=23$的輸出。
卷積區塊的通道數控制了輸出通道數相對於輸入通道數的增長，因此也被稱為*增長率*（growth rate）。

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab paddle
blk = DenseBlock(2, 3, 10)
X = paddle.randn([4, 3, 8, 8])
Y = blk(X)
Y.shape
```

## [**轉場層**]

由於每個稠密區塊都會帶來通道數的增加，使用過多則會過於複雜化模型。
而轉場層可以用來控制模型複雜度。
它透過$1\times 1$卷積層來減小通道數，並使用步幅為2的平均彙聚層減半高和寬，從而進一步降低模型複雜度。

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```{.python .input}
#@tab paddle
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2D(input_channels), nn.ReLU(),
        nn.Conv2D(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2D(kernel_size=2, stride=2))
```

對上一個例子中稠密區塊的輸出[**使用**]通道數為10的[**轉場層**]。
此時輸出的通道數減為10，高和寬均減半。

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch, paddle
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## [**DenseNet模型**]

我們來構造DenseNet模型。DenseNet首先使用同ResNet一樣的單卷積層和最大彙聚層。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

```{.python .input}
#@tab paddle
b1 = nn.Sequential(
    nn.Conv2D(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2D(64), nn.ReLU(),
    nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
```

接下來，類似於ResNet使用的4個殘差區塊，DenseNet使用的是4個稠密區塊。
與ResNet類似，我們可以設定每個稠密區塊使用多少個卷積層。
這裡我們設成4，從而與 :numref:`sec_resnet`的ResNet-18保持一致。
稠密區塊裡的卷積層通道數（即增長率）設為32，所以每個稠密區塊將增加128個通道。

在每個模組之間，ResNet透過步幅為2的殘差區塊減小高和寬，DenseNet則使用轉場層來減半高和寬，並減半通道數。

```{.python .input}
# num_channels為當前的通道數
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # 上一個稠密區塊的輸出通道數
    num_channels += num_convs * growth_rate
    # 在稠密區塊之間新增一個轉換層，使通道數量減半
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# num_channels為當前的通道數
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一個稠密區塊的輸出通道數
    num_channels += num_convs * growth_rate
    # 在稠密區塊之間新增一個轉換層，使通道數量減半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # num_channels為當前的通道數
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # 上一個稠密區塊的輸出通道數
        num_channels += num_convs * growth_rate
        # 在稠密區塊之間新增一個轉換層，使通道數量減半
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

```{.python .input}
#@tab paddle
# num_channels為當前的通道數
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一個稠密區塊的輸出通道數
    num_channels += num_convs * growth_rate
    # 在稠密區塊之間新增一個轉換層，使通道數量減半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

與ResNet類似，最後接上全域彙聚層和全連線層來輸出結果。

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

```{.python .input}
#@tab paddle
net = nn.Sequential(
    b1, *blks, 
    nn.BatchNorm2D(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2D((1, 1)), 
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

## [**訓練模型**]

由於這裡使用了比較深的網路，本節裡我們將輸入高和寬從224降到96來簡化計算。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* 在跨層連線上，不同於ResNet中將輸入與輸出相加，稠密連線網路（DenseNet）在通道維上連結輸入與輸出。
* DenseNet的主要建立模組是稠密區塊和轉場層。
* 在建立DenseNet時，我們需要透過新增轉場層來控制網路的維數，從而再次減少通道的數量。

## 練習

1. 為什麼我們在轉場層使用平均彙聚層而不是最大彙聚層？
1. DenseNet的優點之一是其模型引數比ResNet小。為什麼呢？
1. DenseNet一個詬病的問題是記憶體或顯存消耗過多。
    1. 真的是這樣嗎？可以把輸入形狀換成$224 \times 224$，來看看實際的顯存消耗。
    1. 有另一種方法來減少顯存消耗嗎？需要改變框架麼？
1. 實現DenseNet論文 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`表1所示的不同DenseNet版本。
1. 應用DenseNet的思想設計一個基於多層感知機的模型。將其應用於 :numref:`sec_kaggle_house`中的房價預測任務。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1882)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1880)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1881)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11794)
:end_tab:
