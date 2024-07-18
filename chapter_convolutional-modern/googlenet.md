# 含並行連結的網路（GoogLeNet）
:label:`sec_googlenet`

在2014年的ImageNet影象識別挑戰賽中，一個名叫*GoogLeNet* :cite:`Szegedy.Liu.Jia.ea.2015`的網路架構大放異彩。
GoogLeNet吸收了NiN中串聯網路的思想，並在此基礎上做了改進。
這篇論文的一個重點是解決了什麼樣大小的卷積核最合適的問題。
畢竟，以前流行的網路使用小到$1 \times 1$，大到$11 \times 11$的卷積核。
本文的一個觀點是，有時使用不同大小的卷積核組合是有利的。
本節將介紹一個稍微簡化的GoogLeNet版本：我們省略了一些為穩定訓練而新增的特殊屬性，現在有了更好的訓練方法，這些屬性不是必要的。

## (**Inception區塊**)

在GoogLeNet中，基本的卷積區塊被稱為*Inception區塊*（Inception block）。這很可能得名於電影《盜夢空間》（Inception），因為電影中的一句話『我們需要走得更深』（『We need to go deeper』）。

![Inception區塊的架構。](../img/inception.svg)
:label:`fig_inception`

如 :numref:`fig_inception`所示，Inception區塊由四條並行路徑組成。
前三條路徑使用視窗大小為$1\times 1$、$3\times 3$和$5\times 5$的卷積層，從不同空間大小中提取訊息。
中間的兩條路徑在輸入上執行$1\times 1$卷積，以減少通道數，從而降低模型的複雜性。
第四條路徑使用$3\times 3$最大彙聚層，然後使用$1\times 1$卷積層來改變通道數。
這四條路徑都使用合適的填充來使輸入與輸出的高和寬一致，最後我們將每條線路的輸出在通道維度上連結，並構成Inception區塊的輸出。在Inception塊中，通常調整的超引數是每層輸出通道數。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # c1--c4是每條路徑的輸出通道數
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 線路1，單1x1卷積層
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 線路2，1x1卷積層後接3x3卷積層
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # 線路3，1x1卷積層後接5x5卷積層
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # 線路4，3x3最大彙聚層後接1x1卷積層
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道維度上連結輸出
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # c1--c4是每條路徑的輸出通道數
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 線路1，單1x1卷積層
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 線路2，1x1卷積層後接3x3卷積層
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 線路3，1x1卷積層後接5x5卷積層
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 線路4，3x3最大彙聚層後接1x1卷積層
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道維度上連結輸出
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # c1--c4是每條路徑的輸出通道數
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 線路1，單1x1卷積層
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # 線路2，1x1卷積層後接3x3卷積層
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # 線路3，1x1卷積層後接5x5卷積層
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # 線路4，3x3最大彙聚層後接1x1卷積層
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道維度上連結輸出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Inception(nn.Layer):
    # c1--c4是每條路徑的輸出通道數
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 線路1，單1x1卷積層
        self.p1_1 = nn.Conv2D(in_channels, c1, kernel_size=1)
        # 線路2，1x1卷積層後接3x3卷積層
        self.p2_1 = nn.Conv2D(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2D(c2[0], c2[1], kernel_size=3, padding=1)
        # 線路3，1x1卷積層後接5x5卷積層
        self.p3_1 = nn.Conv2D(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2D(c3[0], c3[1], kernel_size=5, padding=2)
        # 線路4，3x3最大池化層後接1x1卷積層
        self.p4_1 = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2D(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道維度上連結輸出
        return paddle.concat(x=[p1, p2, p3, p4], axis=1)
```

那麼為什麼GoogLeNet這個網路如此有效呢？
首先我們考慮一下濾波器（filter）的組合，它們可以用各種濾波器尺寸探索影象，這意味著不同大小的濾波器可以有效地識別不同範圍的影象細節。
同時，我們可以為不同的濾波器分配不同數量的引數。

## [**GoogLeNet模型**]

如 :numref:`fig_inception_full`所示，GoogLeNet一共使用9個Inception區塊和全域平均彙聚層的堆疊來生成其估計值。Inception區塊之間的最大彙聚層可降低維度。
第一個模組類似於AlexNet和LeNet，Inception區塊的組合從VGG繼承，全域平均彙聚層避免了在最後使用全連線層。

![GoogLeNet架構。](../img/inception-full.svg)
:label:`fig_inception_full`

現在，我們逐一實現GoogLeNet的每個模組。第一個模組使用64個通道、$7\times 7$卷積層。

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

```{.python .input}
#@tab paddle
b1 = nn.Sequential(nn.Conv2D(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(), 
                   nn.MaxPool2D(kernel_size=3, stride=2,padding=1))
```

第二個模組使用兩個卷積層：第一個卷積層是64個通道、$1\times 1$卷積層；第二個卷積層使用將通道數量增加三倍的$3\times 3$卷積層。
這對應於Inception塊中的第二條路徑。

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

```{.python .input}
#@tab paddle
b2 = nn.Sequential(nn.Conv2D(64, 64, kernel_size=1), 
                   nn.ReLU(),
                   nn.Conv2D(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
```

第三個模組串聯兩個完整的Inception區塊。
第一個Inception區塊的輸出通道數為$64+128+32+32=256$，四個路徑之間的輸出通道數量比為$64:128:32:32=2:4:1:1$。
第二個和第三個路徑首先將輸入通道的數量分別減少到$96/192=1/2$和$16/192=1/12$，然後連線第二個卷積層。第二個Inception區塊的輸出通道數增加到$128+192+96+64=480$，四個路徑之間的輸出通道數量比為$128:192:96:64 = 4:6:3:2$。
第二條和第三條路徑首先將輸入通道的數量分別減少到$128/256=1/2$和$32/256=1/8$。

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

```{.python .input}
#@tab paddle
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
```

第四模組更加複雜，
它串聯了5個Inception區塊，其輸出通道數分別是$192+208+48+64=512$、$160+224+64+64=512$、$128+256+64+64=512$、$112+288+64+64=528$和$256+320+128+128=832$。
這些路徑的通道數分配和第三模組中的類似，首先是含$3×3$卷積層的第二條路徑輸出最多通道，其次是僅含$1×1$卷積層的第一條路徑，之後是含$5×5$卷積層的第三條路徑和含$3×3$最大彙聚層的第四條路徑。
其中第二、第三條路徑都會先按比例減小通道數。
這些比例在各個Inception塊中都略有不同。

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

```{.python .input}
#@tab paddle
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
```

第五模組包含輸出通道數為$256+320+128+128=832$和$384+384+128+128=1024$的兩個Inception區塊。
其中每條路徑通道數的分配思路和第三、第四模組中的一致，只是在具體數值上有所不同。
需要注意的是，第五模組的後面緊跟輸出層，該模組同NiN一樣使用全域平均彙聚層，將每個通道的高和寬變成1。
最後我們將輸出變成二維陣列，再接上一個輸出個數為標籤類別數的全連線層。

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])

# 『net』必須是一個將被傳遞給『d2l.train_ch6（）』的函式。
# 為了利用我們現有的CPU/GPU裝置，這樣模型建立/編譯需要在『strategy.scope()』
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

```{.python .input}
#@tab paddle
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2D((1, 1)), 
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

GoogLeNet模型的計算複雜，而且不如VGG那樣便於修改通道數。
[**為了使Fashion-MNIST上的訓練短小精悍，我們將輸入的高和寬從224降到96**]，這簡化了計算。下面示範各個模組輸出的形狀變化。

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab paddle
X = paddle.rand(shape=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**訓練模型**]

和以前一樣，我們使用Fashion-MNIST資料集來訓練我們的模型。在訓練之前，我們將圖片轉換為$96 \times 96$解析度。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* Inception區塊相當於一個有4條路徑的子網路。它透過不同視窗形狀的卷積層和最大彙聚層來並行抽取訊息，並使用$1×1$卷積層減少每畫素級別上的通道維數從而降低模型複雜度。
*  GoogLeNet將多個設計精細的Inception區塊與其他層（卷積層、全連線層）串聯起來。其中Inception區塊的通道數分配之比是在ImageNet資料集上透過大量的實驗得來的。
* GoogLeNet和它的後繼者們一度是ImageNet上最有效的模型之一：它以較低的計算複雜度提供了類似的測試精度。

## 練習

1. GoogLeNet有一些後續版本。嘗試實現並執行它們，然後觀察實驗結果。這些後續版本包括：
    * 新增批次規範化層 :cite:`Ioffe.Szegedy.2015`（batch normalization），在 :numref:`sec_batch_norm`中將介紹；
    * 對Inception模組進行調整 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`；
    * 使用標籤平滑（label smoothing）進行模型正則化 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`；
    * 加入殘差連線 :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`。（ :numref:`sec_resnet`將介紹）。
1. 使用GoogLeNet的最小影象大小是多少？
1. 將AlexNet、VGG和NiN的模型引數大小與GoogLeNet進行比較。後兩個網路架構是如何顯著減少模型引數大小的？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1873)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1871)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1872)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11791)
:end_tab:
