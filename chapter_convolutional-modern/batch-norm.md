# 批次規範化
:label:`sec_batch_norm`

訓練深層神經網路是十分困難的，特別是在較短的時間內使他們收斂更加棘手。
本節將介紹*批次規範化*（batch normalization） :cite:`Ioffe.Szegedy.2015`，這是一種流行且有效的技術，可持續加速深層網路的收斂速度。
再結合在 :numref:`sec_resnet`中將介紹的殘差區塊，批次規範化使得研究人員能夠訓練100層以上的網路。

## 訓練深層網路

為什麼需要批次規範化層呢？讓我們來回顧一下訓練神經網路時出現的一些實際挑戰。

首先，資料預處理的方式通常會對最終結果產生巨大影響。
回想一下我們應用多層感知機來預測房價的例子（ :numref:`sec_kaggle_house`）。
使用真實資料時，我們的第一步是標準化輸入特徵，使其平均值為0，方差為1。
直觀地說，這種標準化可以很好地與我們的最佳化器配合使用，因為它可以將引數的量級進行統一。

第二，對於典型的多層感知機或卷積神經網路。當我們訓練時，中間層中的變數（例如，多層感知機中的仿射變換輸出）可能具有更廣的變化範圍：不論是沿著從輸入到輸出的層，跨同一層中的單元，或是隨著時間的推移，模型引數的隨著訓練更新變幻莫測。
批次規範化的發明者非正式地假設，這些變數分佈中的這種偏移可能會阻礙網路的收斂。
直觀地說，我們可能會猜想，如果一個層的可變值是另一層的100倍，這可能需要對學習率進行補償調整。

第三，更深層的網路很複雜，容易過擬合。
這意味著正則化變得更加重要。

批次規範化應用於單個可選層（也可以應用到所有層），其原理如下：在每次訓練迭代中，我們首先規範化輸入，即透過減去其均值併除以其標準差，其中兩者均基於當前小批次處理。
接下來，我們應用比例係數和比例偏移。
正是由於這個基於*批次*統計的*標準化*，才有了*批次規範化*的名稱。

請注意，如果我們嘗試使用大小為1的小批次應用批次規範化，我們將無法學到任何東西。
這是因為在減去均值之後，每個隱藏單元將為0。
所以，只有使用足夠大的小批次，批次規範化這種方法才是有效且穩定的。
請注意，在應用批次規範化時，批次大小的選擇可能比沒有批次規範化時更重要。

從形式上來說，用$\mathbf{x} \in \mathcal{B}$表示一個來自小批次$\mathcal{B}$的輸入，批次規範化$\mathrm{BN}$根據以下表示式轉換$\mathbf{x}$：

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

在 :eqref:`eq_batchnorm`中，$\hat{\boldsymbol{\mu}}_\mathcal{B}$是小批次$\mathcal{B}$的樣本均值，$\hat{\boldsymbol{\sigma}}_\mathcal{B}$是小批次$\mathcal{B}$的樣本標準差。
應用標準化後，生成的小批次的平均值為0和單位方差為1。
由於單位方差（與其他一些魔法數）是一個主觀的選擇，因此我們通常包含
*拉伸引數*（scale）$\boldsymbol{\gamma}$和*偏移引數*（shift）$\boldsymbol{\beta}$，它們的形狀與$\mathbf{x}$相同。
請注意，$\boldsymbol{\gamma}$和$\boldsymbol{\beta}$是需要與其他模型引數一起學習的引數。

由於在訓練過程中，中間層的變化幅度不能過於劇烈，而批次規範化將每一層主動居中，並將它們重新調整為給定的平均值和大小（透過$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$）。

從形式上來看，我們計算出 :eqref:`eq_batchnorm`中的$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$，如下所示：

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

請注意，我們在方差估計值中新增一個小的常數$\epsilon > 0$，以確保我們永遠不會嘗試除以零，即使在經驗方差估計值可能消失的情況下也是如此。估計值$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$透過使用平均值和方差的噪聲（noise）估計來抵消縮放問題。
乍看起來，這種噪聲是一個問題，而事實上它是有益的。

事實證明，這是深度學習中一個反覆出現的主題。
由於尚未在理論上明確的原因，最佳化中的各種噪聲源通常會導致更快的訓練和較少的過擬合：這種變化似乎是正則化的一種形式。
在一些初步研究中， :cite:`Teye.Azizpour.Smith.2018`和 :cite:`Luo.Wang.Shao.ea.2018`分別將批次規範化的性質與貝葉斯先驗相關聯。
這些理論揭示了為什麼批次規範化最適應$50 \sim 100$範圍中的中等批次大小的難題。

另外，批次規範化層在』訓練模式『（透過小批次統計資料規範化）和『預測模式』（透過資料集統計規範化）中的功能不同。
在訓練過程中，我們無法得知使用整個資料集來估計平均值和方差，所以只能根據每個小批次的平均值和方差不斷訓練模型。
而在預測模式下，可以根據整個資料集精確計算批次規範化所需的平均值和方差。

現在，我們瞭解一下批次規範化在實踐中是如何工作的。

## 批次規範化層

回想一下，批次規範化和其他層之間的一個關鍵區別是，由於批次規範化在完整的小批次上執行，因此我們不能像以前在引入其他層時那樣忽略批次大小。
我們在下面討論這兩種情況：全連線層和卷積層，他們的批次規範化實現略有不同。

### 全連線層

通常，我們將批次規範化層置於全連線層中的仿射變換和啟用函式之間。
設全連線層的輸入為x，權重引數和偏置引數分別為$\mathbf{W}$和$\mathbf{b}$，啟用函式為$\phi$，批次規範化的運算符為$\mathrm{BN}$。
那麼，使用批次規範化的全連線層的輸出的計算詳情如下：

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值和方差是在應用變換的"相同"小批次上計算的。

### 卷積層

同樣，對於卷積層，我們可以在卷積層之後和非線性啟用函式之前應用批次規範化。
當卷積有多個輸出通道時，我們需要對這些通道的『每個』輸出執行批次規範化，每個通道都有自己的拉伸（scale）和偏移（shift）引數，這兩個引數都是標量。
假設我們的小批次包含$m$個樣本，並且對於每個通道，卷積的輸出具有高度$p$和寬度$q$。
那麼對於卷積層，我們在每個輸出通道的$m \cdot p \cdot q$個元素上同時執行每個批次規範化。
因此，在計算平均值和方差時，我們會收集所有空間位置的值，然後在給定通道內應用相同的均值和方差，以便在每個空間位置對值進行規範化。

### 預測過程中的批次規範化

正如我們前面提到的，批次規範化在訓練模式和預測模式下的行為通常不同。
首先，將訓練好的模型用於預測時，我們不再需要樣本均值中的噪聲以及在微批次上估計每個小批次產生的樣本方差了。
其次，例如，我們可能需要使用我們的模型對逐個樣本進行預測。
一種常用的方法是透過移動平均估算整個訓練資料集的樣本均值和方差，並在預測時使用它們得到確定的輸出。
可見，和暫退法一樣，批次規範化層在訓練模式和預測模式下的計算結果也是不一樣的。

## (**從零實現**)

下面，我們從頭開始實現一個具有張量的批次規範化層。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 透過autograd來判斷當前模式是訓練模式還是預測模式
    if not autograd.is_training():
        # 如果是在預測模式下，直接使用傳入的移動平均所得的均值和方差
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全連線層的情況，計算特徵維上的均值和方差
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 使用二維卷積層的情況，計算通道維上（axis=1）的均值和方差。
            # 這裡我們需要保持X的形狀以便後面可以做廣播運算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 訓練模式下，用當前的均值和方差做標準化
        X_hat = (X - mean) / np.sqrt(var + eps)
        # 更新移動平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 縮放和移位
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 透過is_grad_enabled來判斷當前模式是訓練模式還是預測模式
    if not torch.is_grad_enabled():
        # 如果是在預測模式下，直接使用傳入的移動平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全連線層的情況，計算特徵維上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二維卷積層的情況，計算通道維上（axis=1）的均值和方差。
            # 這裡我們需要保持X的形狀以便後面可以做廣播運算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 訓練模式下，用當前的均值和方差做標準化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移動平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 縮放和移位
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # 計算移動方差元平方根的倒數
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # 縮放和移位
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum, is_training=True):
    # 訓練模式還與預測模式的BN處理不同
    if not is_training:
        # 如果是在預測模式下，直接使用傳入的移動平均所得的均值和方差
        X_hat = (X - moving_mean) / (moving_var + eps) ** 0.5
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全連線層的情況，計算特徵維上的均值和方差
            mean = paddle.mean(X)
            var = paddle.mean(((X - mean) ** 2))
        else:
            # 使用二維卷積層的情況，計算通道維上（axis=1）的均值和方差。這裡我們需要保持
            # X的形狀以便後面可以做廣播運算
            mean = paddle.mean(X, axis=(0, 2, 3), keepdim=True)
            var = paddle.mean(((X - mean) ** 2), axis=(0, 2, 3), keepdim=True)
        # 訓練模式下用當前的均值和方差做標準化
        X_hat = (X - mean) / (var + eps) ** 0.5
        # 更新移動平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 縮放和移位
    return Y, moving_mean, moving_var
```

我們現在可以[**建立一個正確的`BatchNorm`層**]。
這個層將保持適當的引數：拉伸`gamma`和偏移`beta`,這兩個引數將在訓練過程中更新。
此外，我們的層將儲存均值和方差的移動平均值，以便在模型預測期間隨後使用。

撇開演算法細節，注意我們實現層的基礎設計模式。
通常情況下，我們用一個單獨的函式定義其數學原理，比如說`batch_norm`。
然後，我們將此功能整合到一個自定義層中，其程式碼主要處理資料移動到訓練裝置（如GPU）、分配和初始化任何必需的變數、跟蹤移動平均線（此處為均值和方差）等問題。
為了方便起見，我們並不擔心在這裡自動推斷輸入形狀，因此我們需要指定整個特徵的數量。
不用擔心，深度學習框架中的批次規範化API將為我們解決上述問題，我們稍後將展示這一點。

```{.python .input}
class BatchNorm(nn.Block):
    # num_features：完全連線層的輸出數量或卷積層的輸出通道數。
    # num_dims：2表示完全連線層，4表示卷積層
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 參與求梯度和迭代的拉伸和偏移引數，分別初始化成1和0
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # 非模型引數的變數初始化為0和1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # 如果X不在記憶體上，將moving_mean和moving_var
        # 複製到X所在顯存上
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # 儲存更新過的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # num_features：完全連線層的輸出數量或卷積層的輸出通道數。
    # num_dims：2表示完全連線層，4表示卷積層
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 參與求梯度和迭代的拉伸和偏移引數，分別初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型引數的變數初始化為0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在記憶體上，將moving_mean和moving_var
        # 複製到X所在顯存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 儲存更新過的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # 參與求梯度和迭代的拉伸和偏移引數，分別初始化成1和0
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # 非模型引數的變數初始化為0和1
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
#@tab paddle
class BatchNorm(nn.Layer):
    def __init__(self, num_features, num_dims=4):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 參與求梯度和迭代的拉伸和偏移引數，分別初始化成1和0
        self.gamma = self.create_parameter(
            attr=None,
            shape=shape,
            dtype='float32',
            is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=shape, dtype='float32')))
        self.beta = self.create_parameter(
            attr=None,
            shape=shape,
            dtype='float32',
            is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=shape, dtype='float32')))
        self.moving_mean = paddle.zeros(shape=shape, dtype='float32')
        self.moving_var = paddle.zeros(shape=shape, dtype='float32')

    def forward(self, X):
        # 儲存更新過的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9, is_training=self.training)
        return Y
```

##  使用批次規範化層的 LeNet

為了更好理解如何[**應用`BatchNorm`**]，下面我們將其應用(**於LeNet模型**)（ :numref:`sec_lenet`）。
回想一下，批次規範化是在卷積層或全連線層之後、相應的啟用函式之前應用的。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# 回想一下，這個函式必須傳遞給d2l.train_ch6。
# 或者說為了利用我們現有的CPU/GPU裝置，需要在strategy.scope()建立模型
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

```{.python .input}
#@tab paddle
net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(), nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(), 
    nn.Linear(84, 10))
```

和以前一樣，我們將[**在Fashion-MNIST資料集上訓練網路**]。
這個程式碼與我們第一次訓練LeNet（ :numref:`sec_lenet`）時幾乎完全相同，主要區別在於學習率大得多。

```{.python .input}
#@tab mxnet, pytorch, paddle
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

讓我們來看看從第一個批次規範化層中學到的[**拉伸引數`gamma`和偏移引數`beta`**]。

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

```{.python .input}
#@tab paddle
param = net.parameters()
print('gamma:', param[2].numpy().reshape(-1))
print('beta:', param[3].numpy().reshape(-1))
```

## [**簡明實現**]

除了使用我們剛剛定義的`BatchNorm`，我們也可以直接使用深度學習框架中定義的`BatchNorm`。
該程式碼看起來幾乎與我們上面的程式碼相同。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

```{.python .input}
#@tab paddle
net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5), nn.BatchNorm2D(6, momentum=0.1), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), nn.BatchNorm2D(16, momentum=0.1), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(), 
    nn.Linear(256, 120), nn.BatchNorm1D(120, momentum=0.1), nn.Sigmoid(), 
    nn.Linear(120, 84), nn.BatchNorm1D(84, momentum=0.1), nn.Sigmoid(), 
    nn.Linear(84, 10))
```

下面，我們[**使用相同超引數來訓練模型**]。
請注意，通常高階API變體執行速度快得多，因為它的程式碼已編譯為C++或CUDA，而我們的自定義程式碼由Python實現。

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 爭議

直觀地說，批次規範化被認為可以使最佳化更加平滑。
然而，我們必須小心區分直覺和對我們觀察到的現象的真實解釋。
回想一下，我們甚至不知道簡單的神經網路（多層感知機和傳統的卷積神經網路）為什麼如此有效。
即使在暫退法和權重衰減的情況下，它們仍然非常靈活，因此無法透過常規的學習理論泛化保證來解釋它們是否能夠泛化到看不見的資料。

在提出批次規範化的論文中，作者除了介紹了其應用，還解釋了其原理：透過減少*內部協變數偏移*（internal covariate shift）。
據推測，作者所說的*內部協變數轉移*類似於上述的投機直覺，即變數值的分佈在訓練過程中會發生變化。
然而，這種解釋有兩個問題：
1、這種偏移與嚴格定義的*協變數偏移*（covariate shift）非常不同，所以這個名字用詞不當；
2、這種解釋只提供了一種不明確的直覺，但留下了一個有待後續挖掘的問題：為什麼這項技術如此有效？
本書旨在傳達實踐者用來發展深層神經網路的直覺。
然而，重要的是將這些指導性直覺與既定的科學事實區分開來。
最終，當你掌握了這些方法，並開始撰寫自己的研究論文時，你會希望清楚地區分技術和直覺。

隨著批次規範化的普及，*內部協變數偏移*的解釋反覆出現在技術文獻的辯論，特別是關於『如何展示機器學習研究』的更廣泛的討論中。
Ali Rahimi在接受2017年NeurIPS大會的『接受時間考驗獎』（Test of Time Award）時發表了一篇令人難忘的演講。他將『內部協變數轉移』作為焦點，將現代深度學習的實踐比作煉金術。
他對該範例進行了詳細回顧 :cite:`Lipton.Steinhardt.2018`，概述了機器學習中令人不安的趨勢。
此外，一些作者對批次規範化的成功提出了另一種解釋：在某些方面，批次規範化的表現出與原始論文 :cite:`Santurkar.Tsipras.Ilyas.ea.2018`中聲稱的行為是相反的。

然而，與機器學習文獻中成千上萬類似模糊的說法相比，內部協變數偏移沒有更值得批評。
很可能，它作為這些辯論的焦點而產生共鳴，要歸功於目標受眾對它的廣泛認可。
批次規範化已經被證明是一種不可或缺的方法。它適用於幾乎所有影象分類器，並在學術界獲得了數萬引用。

## 小結

* 在模型訓練過程中，批次規範化利用小批次的均值和標準差，不斷調整神經網路的中間輸出，使整個神經網路各層的中間輸出值更加穩定。
* 批次規範化在全連線層和卷積層的使用略有不同。
* 批次規範化層和暫退層一樣，在訓練模式和預測模式下計算不同。
* 批次規範化有許多有益的副作用，主要是正則化。另一方面，』減少內部協變數偏移『的原始動機似乎不是一個有效的解釋。

## 練習

1. 在使用批次規範化之前，我們是否可以從全連線層或卷積層中刪除偏置引數？為什麼？
1. 比較LeNet在使用和不使用批次規範化情況下的學習率。
    1. 繪製訓練和測試準確度的提高。
    1. 學習率有多高？
1. 我們是否需要在每個層中進行批次規範化？嘗試一下？
1. 可以透過批次規範化來替換暫退法嗎？行為會如何改變？
1. 確定引數`beta`和`gamma`，並觀察和分析結果。
1. 檢視高階API中有關`BatchNorm`的線上文件，以檢視其他批次規範化的應用。
1. 研究思路：可以應用的其他『規範化』轉換？可以應用機率積分變換嗎？全秩協方差估計可以麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1876)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1874)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1875)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11792)
:end_tab:
