# 自定義層

深度學習成功背後的一個因素是神經網路的靈活性：
我們可以用創造性的方式組合不同的層，從而設計出適用於各種任務的架構。
例如，研究人員發明了專門用於處理影象、文字、序列資料和執行動態規劃的層。
有時我們會遇到或要自己發明一個現在在深度學習框架中還不存在的層。
在這些情況下，必須建立自定義層。本節將展示如何建立自定義層。

## 不帶引數的層

首先，我們(**構造一個沒有任何引數的自定義層**)。
回憶一下在 :numref:`sec_model_construction`對區塊的介紹，
這應該看起來很眼熟。
下面的`CenteredLayer`類要從其輸入中減去均值。
要建立它，我們只需繼承基礎層類並實現前向傳播功能。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn
import paddle.nn.functional as F

class CenteredLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

讓我們向該層提供一些資料，驗證它是否能按預期工作。

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab paddle
layer = CenteredLayer()
layer(paddle.to_tensor([1, 2, 3, 4, 5], dtype='float32'))
```

現在，我們可以[**將層作為元件合併到更復雜的模型中**]。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch, paddle
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

作為額外的健全性檢查，我們可以在向該網路傳送隨機數據後，檢查均值是否為0。
由於我們處理的是浮點數，因為儲存精度的原因，我們仍然可能會看到一個非常小的非零數。

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
#@tab paddle
Y = net(paddle.rand([4, 8]))
Y.mean()
```

## [**帶引數的層**]

以上我們知道了如何定義簡單的層，下面我們繼續定義具有引數的層，
這些引數可以透過訓練進行調整。
我們可以使用內建函式來建立引數，這些函式提供一些基本的管理功能。
比如管理訪問、初始化、共享、儲存和載入模型引數。
這樣做的好處之一是：我們不需要為每個自定義層編寫自定義的序列化程式。

現在，讓我們實現自定義版本的全連線層。
回想一下，該層需要兩個引數，一個用於表示權重，另一個用於表示偏置項。
在此實現中，我們使用修正線性單元作為啟用函式。
該層需要輸入引數：`in_units`和`units`，分別表示輸入數和輸出數。

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
#@tab paddle
class MyLinear(nn.Layer):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = paddle.create_parameter(shape=(in_units, units), dtype='float32')
        self.bias = paddle.create_parameter(shape=(units,), dtype='float32')
        
    def forward(self, X):
        linear = paddle.matmul(X, self.weight) + self.bias
        return F.relu(linear)
```

:begin_tab:`mxnet, tensorflow`
接下來，我們實例化`MyDense`類並訪問其模型引數。
:end_tab:

:begin_tab:`pytorch`
接下來，我們實例化`MyLinear`類並訪問其模型引數。
:end_tab:

:begin_tab:`paddle`
接下來，我們實例化`MyLinear`類並訪問其模型引數。
:end_tab:

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch, paddle
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

我們可以[**使用自定義層直接執行前向傳播計算**]。

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
#@tab paddle
linear(paddle.randn([2, 5]))
```

我們還可以(**使用自定義層建立模型**)，就像使用內建的全連線層一樣使用自定義層。

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
#@tab paddle
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(paddle.rand([2, 64]))
```

## 小結

* 我們可以透過基本層類設計自定義層。這允許我們定義靈活的新層，其行為與深度學習框架中的任何現有層不同。
* 在自定義層定義完成後，我們就可以在任意環境和網路架構中呼叫該自定義層。
* 層可以有區域性引數，這些引數可以透過內建函式建立。

## 練習

1. 設計一個接受輸入並計算張量降維的層，它回傳$y_k = \sum_{i, j} W_{ijk} x_i x_j$。
1. 設計一個回傳輸入資料的傅立葉係數前半部分的層。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1837)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1835)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1836)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11780)
:end_tab:
