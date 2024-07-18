# 多層感知機的從零開始實現
:label:`sec_mlp_scratch`

我們已經在 :numref:`sec_mlp`中描述了多層感知機（MLP），
現在讓我們嘗試自己實現一個多層感知機。
為了與之前softmax迴歸（ :numref:`sec_softmax_scratch` ）
獲得的結果進行比較，
我們將繼續使用Fashion-MNIST影象分類資料集
（ :numref:`sec_fashion_mnist`）。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
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
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型引數

回想一下，Fashion-MNIST中的每個影象由
$28 \times 28 = 784$個灰度畫素值組成。
所有影象共分為10個類別。
忽略畫素之間的空間結構，
我們可以將每個影象視為具有784個輸入特徵
和10個類別的簡單分類資料集。
首先，我們將[**實現一個具有單隱藏層的多層感知機，
它包含256個隱藏單元**]。
注意，我們可以將這兩個變數都視為超引數。
通常，我們選擇2的若干次冪作為層的寬度。
因為記憶體在硬體中的分配和定址方式，這麼做往往可以在計算上更高效。

我們用幾個張量來表示我們的引數。
注意，對於每一層我們都要記錄一個權重矩陣和一個偏置向量。
跟以前一樣，我們要為損失關於這些引數的梯度分配記憶體。

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.zeros(num_outputs))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab paddle
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = paddle.randn([num_inputs, num_hiddens]) * 0.01
W1.stop_gradient = False
b1 = paddle.zeros([num_hiddens])
b1.stop_gradient = False
W2 = paddle.randn([num_hiddens, num_outputs]) * 0.01
W2.stop_gradient = False
b2 = paddle.zeros([num_outputs])
b2.stop_gradient = False

params = [W1, b1, W2, b2]
```

## 啟用函式

為了確保我們對模型的細節瞭如指掌，
我們將[**實現ReLU啟用函式**]，
而不是直接呼叫內建的`relu`函式。

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
#@tab paddle
def relu(X):
    a = paddle.zeros_like(X)
    return paddle.maximum(X, a)
```

## 模型

因為我們忽略了空間結構，
所以我們使用`reshape`將每個二維影象轉換為一個長度為`num_inputs`的向量。
只需幾行程式碼就可以(**實現我們的模型**)。

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # 這裡『@』代表矩陣乘法
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

```{.python .input}
#@tab paddle
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 這裡『@』代表矩陣乘法
    return (H@W2 + b2)
```

## 損失函式

由於我們已經從零實現過softmax函式（ :numref:`sec_softmax_scratch`），
因此在這裡我們直接使用高階API中的內建函式來計算softmax和交叉熵損失。
回想一下我們之前在 :numref:`subsec_softmax-implementation-revisited`中
對這些複雜問題的討論。
我們鼓勵感興趣的讀者檢視損失函式的原始碼，以加深對實現細節的瞭解。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch, paddle
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## 訓練

幸運的是，[**多層感知機的訓練過程與softmax迴歸的訓練過程完全相同**]。
可以直接呼叫`d2l`套件的`train_ch3`函式（參見 :numref:`sec_softmax_scratch` ），
將迭代週期數設定為10，並將學習率設定為0.1.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab paddle
num_epochs, lr = 10, 0.1
updater = paddle.optimizer.SGD(learning_rate=lr, parameters=params)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

為了對學習到的模型進行評估，我們將[**在一些測試資料上應用這個模型**]。

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## 小結

* 手動實現一個簡單的多層感知機是很容易的。然而如果有大量的層，從零開始實現多層感知機會變得很麻煩（例如，要命名和記錄模型的引數）。

## 練習

1. 在所有其他引數保持不變的情況下，更改超引數`num_hiddens`的值，並檢視此超引數的變化對結果有何影響。確定此超引數的最佳值。
1. 嘗試新增更多的隱藏層，並檢視它對結果有何影響。
1. 改變學習速率會如何影響結果？保持模型架構和其他超引數（包括輪數）不變，學習率設定為多少會帶來最好的結果？
1. 透過對所有超引數（學習率、輪數、隱藏層數、每層的隱藏單元數）進行聯合最佳化，可以得到的最佳結果是什麼？
1. 描述為什麼涉及多個超引數更具挑戰性。
1. 如果想要建立多個超引數的搜尋方法，請想出一個聰明的策略。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1800)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1804)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1798)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11769)
:end_tab:
