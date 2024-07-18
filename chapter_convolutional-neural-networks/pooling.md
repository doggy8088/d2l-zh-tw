# 彙聚層
:label:`sec_pooling`

通常當我們處理影象時，我們希望逐漸降低隱藏表示的空間解析度、聚集訊息，這樣隨著我們在神經網路中層疊的上升，每個神經元對其敏感的感受野（輸入）就越大。

而我們的機器學習任務通常會跟全域影象的問題有關（例如，『影象是否包含一隻貓呢？』），所以我們最後一層的神經元應該對整個輸入的全域敏感。透過逐漸聚合訊息，生成越來越粗糙的對映，最終實現學習全域表示的目標，同時將卷積圖層的所有優勢保留在中間層。

此外，當檢測較底層的特徵時（例如 :numref:`sec_conv_layer`中所討論的邊緣），我們通常希望這些特徵保持某種程度上的平移不變性。例如，如果我們拍攝黑白之間輪廓清晰的影象`X`，並將整個影象向右移動一個畫素，即`Z[i, j] = X[i, j + 1]`，則新影象`Z`的輸出可能大不相同。而在現實中，隨著拍攝角度的移動，任何物體幾乎不可能發生在同一畫素上。即使用三腳架拍攝一個靜止的物體，由於快門的移動而引起的相機振動，可能會使所有物體左右移動一個畫素（除了高端相機配備了特殊功能來解決這個問題）。

本節將介紹*彙聚*（pooling）層，它具有雙重目的：降低卷積層對位置的敏感性，同時降低對空間降取樣表示的敏感性。

## 最大彙聚層和平均彙聚層

與卷積層類似，彙聚層運算符由一個固定形狀的視窗組成，該視窗根據其步幅大小在輸入的所有區域上滑動，為固定形狀視窗（有時稱為*彙聚視窗*）遍歷的每個位置計算一個輸出。
然而，不同於卷積層中的輸入與卷積核之間的互相關計算，彙聚層不包含引數。
相反，池運算是確定性的，我們通常計算彙聚視窗中所有元素的最大值或平均值。這些操作分別稱為*最大彙聚層*（maximum pooling）和*平均彙聚層*（average pooling）。

在這兩種情況下，與互相關運算符一樣，彙聚視窗從輸入張量的左上角開始，從左往右、從上往下的在輸入張量內滑動。在彙聚視窗到達的每個位置，它計算該視窗中輸入子張量的最大值或平均值。計算最大值或平均值是取決於使用了最大彙聚層還是平均彙聚層。

![彙聚視窗形狀為 $2\times 2$ 的最大彙聚層。著色部分是第一個輸出元素，以及用於計算這個輸出的輸入元素: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

 :numref:`fig_pooling`中的輸出張量的高度為$2$，寬度為$2$。這四個元素為每個彙聚視窗中的最大值：

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

彙聚視窗形狀為$p \times q$的彙聚層稱為$p \times q$彙聚層，彙聚操作稱為$p \times q$彙聚。

回到本節開頭提到的物件邊緣檢測範例，現在我們將使用卷積層的輸出作為$2\times 2$最大彙聚的輸入。
設定卷積層輸入為`X`，彙聚層輸出為`Y`。
無論`X[i, j]`和`X[i, j + 1]`的值相同與否，或`X[i, j + 1]`和`X[i, j + 2]`的值相同與否，彙聚層始終輸出`Y[i, j] = 1`。
也就是說，使用$2\times 2$最大彙聚層，即使在高度或寬度上移動一個元素，卷積層仍然可以識別到模式。

在下面的程式碼中的`pool2d`函式，我們(**實現彙聚層的前向傳播**)。
這類似於 :numref:`sec_conv_layer`中的`corr2d`函式。
然而，這裡我們沒有卷積核，輸出為輸入中每個區域的最大值或平均值。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

```{.python .input}
#@tab mxnet, pytorch, paddle
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

我們可以建立 :numref:`fig_pooling`中的輸入張量`X`，[**驗證二維最大彙聚層的輸出**]。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

此外，我們還可以(**驗證平均彙聚層**)。

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## [**填充和步幅**]

與卷積層一樣，彙聚層也可以改變輸出形狀。和以前一樣，我們可以透過填充和步幅以獲得所需的輸出形狀。
下面，我們用深度學習框架中內建的二維最大彙聚層，來示範彙聚層中填充和步幅的使用。
我們首先構造了一個輸入張量`X`，它有四個維度，其中樣本數和通道數都是1。

:begin_tab:`tensorflow`
請注意，Tensorflow採用『通道最後』（channels-last）語法，對其進行最佳化，
（即Tensorflow中輸入的最後維度是通道）。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

```{.python .input}
#@tab paddle
X = paddle.arange(16, dtype="float32").reshape((1, 1, 4, 4))
X
```

預設情況下，(**深度學習框架中的步幅與彙聚視窗的大小相同**)。
因此，如果我們使用形狀為`(3, 3)`的彙聚視窗，那麼預設情況下，我們得到的步幅形狀為`(3, 3)`。

```{.python .input}
pool2d = nn.MaxPool2D(3)
# 由於彙聚層中沒有引數，所以不需要呼叫初始化函式
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

```{.python .input}
#@tab paddle
pool2d = nn.MaxPool2D(3, stride=3)
pool2d(X)
```

[**填充和步幅可以手動設定**]。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
#@tab paddle
pool2d = nn.MaxPool2D(3, padding=1, stride=2)
pool2d(X)
```

:begin_tab:`mxnet`
當然，我們可以設定一個任意大小的矩形彙聚視窗，並分別設定填充和步幅的高度和寬度。
:end_tab:

:begin_tab:`pytorch`
當然，我們可以(**設定一個任意大小的矩形彙聚視窗，並分別設定填充和步幅的高度和寬度**)。
:end_tab:

:begin_tab:`tensorflow`
當然，我們可以設定一個任意大小的矩形彙聚視窗，並分別設定填充和步幅的高度和寬度。
:end_tab:

:begin_tab:`paddle`
當然，我們可以設定一個任意大小的矩形彙聚視窗，並分別設定填充和步幅的高度和寬度。
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
#@tab paddle
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), stride=(2, 3))
pool2d(X)
```

## 多個通道

在處理多通道輸入資料時，[**彙聚層在每個輸入通道上單獨運算**]，而不是像卷積層一樣在通道上對輸入進行彙總。
這意味著彙聚層的輸出通道數與輸入通道數相同。
下面，我們將在通道維度上連結張量`X`和`X + 1`，以建立具有2個通道的輸入。

:begin_tab:`tensorflow`
請注意，由於TensorFlow採用『通道最後』（channels-last）的語法，
我們需要沿輸入的最後一個維度進行串聯。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch, paddle
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)
```

如下所示，彙聚後輸出通道的數量仍然是2。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
#@tab paddle
pool2d = paddle.nn.MaxPool2D(3, padding=1, stride=2)
pool2d(X)
```

:begin_tab:`tensorflow`
請注意，上面的輸出乍一看似乎有所不同，但MXNet和PyTorch的結果從數值上是相同的。
不同之處在於維度，垂直讀取輸出會產生與其他實現相同的輸出。
:end_tab:

## 小結

* 對於給定輸入元素，最大彙聚層會輸出該視窗內的最大值，平均彙聚層會輸出該視窗內的平均值。
* 彙聚層的主要優點之一是減輕卷積層對位置的過度敏感。
* 我們可以指定彙聚層的填充和步幅。
* 使用最大彙聚層以及大於1的步幅，可減少空間維度（如高度和寬度）。
* 彙聚層的輸出通道數與輸入通道數相同。

## 練習

1. 嘗試將平均彙聚層作為卷積層的特殊情況實現。
1. 嘗試將最大彙聚層作為卷積層的特殊情況實現。
1. 假設彙聚層的輸入大小為$c\times h\times w$，則彙聚視窗的形狀為$p_h\times p_w$，填充為$(p_h, p_w)$，步幅為$(s_h, s_w)$。這個彙聚層的計算成本是多少？
1. 為什麼最大彙聚層和平均彙聚層的工作方式不同？
1. 我們是否需要最小彙聚層？可以用已知函式替換它嗎？
1. 除了平均彙聚層和最大彙聚層，是否有其它函式可以考慮（提示：回想一下`softmax`）？為什麼它不流行？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1858)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1857)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1856)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11786)
:end_tab:
