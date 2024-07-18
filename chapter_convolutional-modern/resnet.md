# 殘差網路（ResNet）
:label:`sec_resnet`

隨著我們設計越來越深的網路，深刻理解『新新增的層如何提升神經網路的效能』變得至關重要。更重要的是設計網路的能力，在這種網路中，新增層會使網路更具表現力，
為了取得質的突破，我們需要一些數學基礎知識。

## 函式類

首先，假設有一類特定的神經網路架構$\mathcal{F}$，它包括學習速率和其他超引數設定。
對於所有$f \in \mathcal{F}$，存在一些引數集（例如權重和偏置），這些引數可以透過在合適的資料集上進行訓練而獲得。
現在假設$f^*$是我們真正想要找到的函式，如果是$f^* \in \mathcal{F}$，那我們可以輕而易舉的訓練得到它，但通常我們不會那麼幸運。
相反，我們將嘗試找到一個函式$f^*_\mathcal{F}$，這是我們在$\mathcal{F}$中的最佳選擇。
例如，給定一個具有$\mathbf{X}$屬性和$\mathbf{y}$標籤的資料集，我們可以嘗試透過解決以下最佳化問題來找到它：

$$f^*_\mathcal{F} := \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

那麼，怎樣得到更近似真正$f^*$的函式呢？
唯一合理的可能性是，我們需要設計一個更強大的架構$\mathcal{F}'$。
換句話說，我們預計$f^*_{\mathcal{F}'}$比$f^*_{\mathcal{F}}$『更近似』。
然而，如果$\mathcal{F} \not\subseteq \mathcal{F}'$，則無法保證新的體系『更近似』。
事實上，$f^*_{\mathcal{F}'}$可能更糟：
如 :numref:`fig_functionclasses`所示，對於非巢狀函式（non-nested function）類，較複雜的函式類並不總是向『真』函式$f^*$靠攏（複雜度由$\mathcal{F}_1$向$\mathcal{F}_6$遞增）。
在 :numref:`fig_functionclasses`的左邊，雖然$\mathcal{F}_3$比$\mathcal{F}_1$更接近$f^*$，但$\mathcal{F}_6$卻離的更遠了。
相反對於 :numref:`fig_functionclasses`右側的巢狀函式（nested function）類$\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$，我們可以避免上述問題。

![對於非巢狀函式類，較複雜（由較大區域表示）的函式類不能保證更接近『真』函式（ $f^*$ ）。這種現象在巢狀函式類別中不會發生。](../img/functionclasses.svg)
:label:`fig_functionclasses`

因此，只有當較複雜的函式類包含較小的函式類時，我們才能確保提高它們的效能。
對於深度神經網路，如果我們能將新新增的層訓練成*恆等對映*（identity function）$f(\mathbf{x}) = \mathbf{x}$，新模型和原模型將同樣有效。
同時，由於新模型可能得出更優的解來擬合訓練資料集，因此新增層似乎更容易降低訓練誤差。

針對這一問題，何愷明等人提出了*殘差網路*（ResNet） :cite:`He.Zhang.Ren.ea.2016`。
它在2015年的ImageNet影象識別挑戰賽奪魁，並深刻影響了後來的深度神經網路的設計。
殘差網路的核心思想是：每個附加層都應該更容易地包含原始函式作為其元素之一。
於是，*殘差區塊*（residual blocks）便誕生了，這個設計對如何建立深層神經網路產生了深遠的影響。
憑藉它，ResNet贏得了2015年ImageNet大規模視覺識別挑戰賽。

## (**殘差區塊**)

讓我們聚焦於神經網路區域性：如圖 :numref:`fig_residual_block`所示，假設我們的原始輸入為$x$，而希望學出的理想對映為$f(\mathbf{x})$（作為 :numref:`fig_residual_block`上方啟用函式的輸入）。
 :numref:`fig_residual_block`左圖虛線框中的部分需要直接擬合出該對映$f(\mathbf{x})$，而右圖虛線框中的部分則需要擬合出殘差對映$f(\mathbf{x}) - \mathbf{x}$。
殘差對映在現實中往往更容易最佳化。
以本節開頭提到的恆等對映作為我們希望學出的理想對映$f(\mathbf{x})$，我們只需將 :numref:`fig_residual_block`中右圖虛線框內上方的加權運算（如仿射）的權重和偏置引數設成0，那麼$f(\mathbf{x})$即為恆等對映。
實際中，當理想對映$f(\mathbf{x})$極接近於恆等對映時，殘差對映也易於捕捉恆等對映的細微波動。
 :numref:`fig_residual_block`右圖是ResNet的基礎架構--*殘差區塊*（residual block）。
在殘差塊中，輸入可透過跨層資料線路更快地向前傳播。

![一個正常區塊（左圖）和一個殘差區塊（右圖）。](../img/residual-block.svg)
:label:`fig_residual_block`

ResNet沿用了VGG完整的$3\times 3$卷積層設計。
殘差區塊裡首先有2個有相同輸出通道數的$3\times 3$卷積層。
每個卷積層後接一個批次規範化層和ReLU啟用函式。
然後我們透過跨層資料通路，跳過這2個卷積運算，將輸入直接加在最後的ReLU啟用函式前。
這樣的設計要求2個卷積層的輸出與輸入形狀一樣，從而使它們可以相加。
如果想改變通道數，就需要引入一個額外的$1\times 1$卷積層來將輸入變換成需要的形狀後再做相加運算。
殘差區塊的實現如下：

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Residual(nn.Block):  #@save
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
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

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Residual(tf.keras.Model):  #@save
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn
from paddle.nn import functional as F

class Residual(nn.Layer):  #@save
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2D(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2D(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2D(num_channels)
        self.bn2 = nn.BatchNorm2D(num_channels)
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

如 :numref:`fig_resnet_block`所示，此程式碼生成兩種型別的網路：
一種是當`use_1x1conv=False`時，應用ReLU非線性函式之前，將輸入新增到輸出。
另一種是當`use_1x1conv=True`時，新增透過$1 \times 1$卷積調整通道和解析度。

![包含以及不包含 $1 \times 1$ 卷積層的殘差區塊。](../img/resnet-block.svg)
:label:`fig_resnet_block`

下面我們來檢視[**輸入和輸出形狀一致**]的情況。

```{.python .input}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab paddle
blk = Residual(3, 3)
X = paddle.rand([4, 3, 6, 6])
Y = blk(X)
Y.shape
```

我們也可以在[**增加輸出通道數的同時，減半輸出的高和寬**]。

```{.python .input}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab paddle
blk = Residual(3, 6, use_1x1conv=True, strides=2)
blk(X).shape
```

## [**ResNet模型**]

ResNet的前兩層跟之前介紹的GoogLeNet中的一樣：
在輸出通道數為64、步幅為2的$7 \times 7$卷積層後，接步幅為2的$3 \times 3$的最大彙聚層。
不同之處在於ResNet每個卷積層後增加了批次規範化層。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

```{.python .input}
#@tab paddle
b1 = nn.Sequential(nn.Conv2D(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2D(64), nn.ReLU(),
                   nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
```

GoogLeNet在後面接了4個由Inception區塊組成的模組。
ResNet則使用4個由殘差區塊組成的模組，每個模組使用若干個同樣輸出通道數的殘差區塊。
第一個模組的通道數同輸入通道數一致。
由於之前已經使用了步幅為2的最大彙聚層，所以無須減小高和寬。
之後的每個模組在第一個殘差區塊裡將上一個模組的通道數翻倍，並將高和寬減半。

下面我們來實現這個模組。注意，我們對第一個模組做了特別處理。

```{.python .input}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
#@tab pytorch
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

```{.python .input}
#@tab tensorflow
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
```

```{.python .input}
#@tab paddle
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

接著在ResNet加入所有殘差區塊，這裡每個模組使用2個殘差區塊。

```{.python .input}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

```{.python .input}
#@tab pytorch, paddle
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

```{.python .input}
#@tab tensorflow
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)
```

最後，與GoogLeNet一樣，在ResNet中加入全域平均彙聚層，以及全連線層輸出。

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

```{.python .input}
#@tab tensorflow
# 回想之前我們定義一個函式，以便用它在tf.distribute.MirroredStrategy的範圍，
# 來利用各種計算資源，例如gpu。另外，儘管我們已經建立了b1、b2、b3、b4、b5，
# 但是我們將在這個函式的作用域內重新建立它們
def net():
    return tf.keras.Sequential([
        # Thefollowinglayersarethesameasb1thatwecreatedearlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # Thefollowinglayersarethesameasb2,b3,b4,andb5thatwe
        # createdearlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])
```

```{.python .input}
#@tab paddle
net = nn.Sequential(b1, b2, b3, b4, b5, 
                    nn.AdaptiveAvgPool2D((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

每個模組有4個卷積層（不包括恆等對映的$1\times 1$卷積層）。
加上第一個$7\times 7$卷積層和最後一個全連線層，共有18層。
因此，這種模型通常被稱為ResNet-18。
透過設定不同的通道數和模組裡的殘差區塊數可以得到不同的ResNet模型，例如更深的含152層的ResNet-152。
雖然ResNet的主體架構跟GoogLeNet類似，但ResNet架構更簡單，修改也更方便。這些因素都導致了ResNet迅速被廣泛使用。
 :numref:`fig_resnet18`描述了完整的ResNet-18。

![ResNet-18 架構](../img/resnet18.svg)
:label:`fig_resnet18`

在訓練ResNet之前，讓我們[**觀察一下ResNet中不同模組的輸入形狀是如何變化的**]。
在之前所有架構中，解析度降低，通道數量增加，直到全域平均彙聚層聚集所有特徵。

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
X = tf.random.uniform(shape=(1, 224, 224, 1))
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

同之前一樣，我們在Fashion-MNIST資料集上訓練ResNet。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* 學習巢狀函式（nested function）是訓練神經網路的理想情況。在深層神經網路中，學習另一層作為恆等對映（identity function）較容易（儘管這是一個極端情況）。
* 殘差對映可以更容易地學習同一函式，例如將權重層中的引數近似為零。
* 利用殘差區塊（residual blocks）可以訓練出一個有效的深層神經網路：輸入可以透過層間的殘餘連線更快地向前傳播。
* 殘差網路（ResNet）對隨後的深層神經網路設計產生了深遠影響。

## 練習

1.  :numref:`fig_inception`中的Inception區塊與殘差區塊之間的主要區別是什麼？在刪除了Inception塊中的一些路徑之後，它們是如何相互關聯的？
1. 參考ResNet論文 :cite:`He.Zhang.Ren.ea.2016`中的表1，以實現不同的變體。
1. 對於更深層次的網路，ResNet引入了『bottleneck』架構來降低模型複雜性。請試著去實現它。
1. 在ResNet的後續版本中，作者將『卷積層、批次規範化層和啟用層』架構更改為『批次規範化層、啟用層和卷積層』架構。請嘗試做這個改進。詳見 :cite:`He.Zhang.Ren.ea.2016*1`中的圖1。
1. 為什麼即使函式類是巢狀的，我們仍然要限制增加函式的複雜性呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1879)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1877)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1878)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11793)
:end_tab:
