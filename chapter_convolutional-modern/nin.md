# 網路中的網路（NiN）
:label:`sec_nin`

LeNet、AlexNet和VGG都有一個共同的設計模式：透過一系列的卷積層與彙聚層來提取空間結構特徵；然後透過全連線層對特徵的表徵進行處理。
AlexNet和VGG對LeNet的改進主要在於如何擴大和加深這兩個模組。
或者，可以想象在這個過程的早期使用全連線層。然而，如果使用了全連線層，可能會完全放棄表徵的空間結構。
*網路中的網路*（*NiN*）提供了一個非常簡單的解決方案：在每個畫素的通道上分別使用多層感知機 :cite:`Lin.Chen.Yan.2013`

## (**NiN區塊**)

回想一下，卷積層的輸入和輸出由四維張量組成，張量的每個軸分別對應樣本、通道、高度和寬度。
另外，全連線層的輸入和輸出通常是分別對應於樣本和特徵的二維張量。
NiN的想法是在每個畫素位置（針對每個高度和寬度）應用一個全連線層。
如果我們將權重連線到每個空間位置，我們可以將其視為$1\times 1$卷積層（如 :numref:`sec_channels`中所述），或作為在每個畫素位置上獨立作用的全連線層。
從另一個角度看，即將空間維度中的每個畫素視為單個樣本，將通道維度視為不同特徵（feature）。

 :numref:`fig_nin`說明了VGG和NiN及它們的區塊之間主要架構差異。
NiN區塊以一個普通卷積層開始，後面是兩個$1 \times 1$的卷積層。這兩個$1 \times 1$卷積層充當帶有ReLU啟用函式的逐畫素全連線層。
第一層的卷積視窗形狀通常由使用者設定。
隨後的卷積視窗形狀固定為$1 \times 1$。

![對比 VGG 和 NiN 及它們的區塊之間主要架構差異。](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(), 
        nn.Conv2D(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), 
        nn.Conv2D(out_channels, out_channels, kernel_size=1),
        nn.ReLU())
```

## [**NiN模型**]

最初的NiN網路是在AlexNet後不久提出的，顯然從中得到了一些啟示。
NiN使用視窗形狀為$11\times 11$、$5\times 5$和$3\times 3$的卷積層，輸出通道數量與AlexNet中的相同。
每個NiN區塊後有一個最大彙聚層，彙聚視窗形狀為$3\times 3$，步幅為2。

NiN和AlexNet之間的一個顯著區別是NiN完全取消了全連線層。
相反，NiN使用一個NiN區塊，其輸出通道數等於標籤類別的數量。最後放一個*全域平均彙聚層*（global average pooling layer），生成一個對數機率	（logits）。NiN設計的一個優點是，它顯著減少了模型所需引數的數量。然而，在實踐中，這種設計有時會增加訓練模型的時間。

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # 標籤類別數是10
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # 全域平均彙聚層將視窗形狀自動設定成輸入的高和寬
        nn.GlobalAvgPool2D(),
        # 將四維的輸出轉成二維的輸出，其形狀為(批次大小,10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 標籤類別數是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 將四維的輸出轉成二維的輸出，其形狀為(批次大小,10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # 標籤類別數是10
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # 將四維的輸出轉成二維的輸出，其形狀為(批次大小,10)
        tf.keras.layers.Flatten(),
        ])
```

```{.python .input}
#@tab paddle
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2D(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2D(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2D(3, stride=2), nn.Dropout(0.5),
    # 標籤類別數是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2D((1, 1)),
    # 將四維的輸出轉成二維的輸出，其形狀為(批次大小,10)
    nn.Flatten())
```

我們建立一個資料樣本來[**檢視每個區塊的輸出形狀**]。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab paddle
X = paddle.rand(shape=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**訓練模型**]

和以前一樣，我們使用Fashion-MNIST來訓練模型。訓練NiN與訓練AlexNet、VGG時相似。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* NiN使用由一個卷積層和多個$1\times 1$卷積層組成的區塊。該區塊可以在卷積神經網路中使用，以允許更多的每畫素非線性。
* NiN去除了容易造成過擬合的全連線層，將它們替換為全域平均彙聚層（即在所有位置上進行求和）。該彙聚層通道數量為所需的輸出數量（例如，Fashion-MNIST的輸出為10）。
* 移除全連線層可減少過擬合，同時顯著減少NiN的引數。
* NiN的設計影響了許多後續卷積神經網路的設計。

## 練習

1. 調整NiN的超引數，以提高分類準確性。
1. 為什麼NiN塊中有兩個$1\times 1$卷積層？刪除其中一個，然後觀察和分析實驗現象。
1. 計算NiN的資源使用情況。
    1. 引數的數量是多少？
    1. 計算量是多少？
    1. 訓練期間需要多少顯存？
    1. 預測期間需要多少顯存？
1. 一次性直接將$384 \times 5 \times 5$的表示縮減為$10 \times 5 \times 5$的表示，會存在哪些問題？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1870)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1869)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1868)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11790)
:end_tab:
