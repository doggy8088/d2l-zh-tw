# softmax迴歸的簡潔實現
:label:`sec_softmax_concise`

在 :numref:`sec_linear_concise`中，
我們發現(**透過深度學習框架的高階API能夠使實現**)
(~~softmax~~)
線性(**迴歸變得更加容易**)。
同樣，透過深度學習框架的高階API也能更方便地實現softmax迴歸模型。
本節如在 :numref:`sec_softmax_scratch`中一樣，
繼續使用Fashion-MNIST資料集，並保持批次大小為256。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
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

如我們在 :numref:`sec_softmax`所述，
[**softmax迴歸的輸出層是一個全連線層**]。
因此，為了實現我們的模型，
我們只需在`Sequential`中新增一個帶有10個輸出的全連線層。
同樣，在這裡`Sequential`並不是必要的，
但它是實現深度模型的基礎。
我們仍然以均值0和標準差0.01隨機初始化權重。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch不會隱式地調整輸入的形狀。因此，
# 我們線上性層前定義了展平層（flatten），來調整網路輸入的形狀
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

```{.python .input}
#@tab paddle
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.initializer.Normal(m.weight, std=0.01)

net.apply(init_weights);
```

## 重新審視Softmax的實現
:label:`subsec_softmax-implementation-revisited`

在前面 :numref:`sec_softmax_scratch`的例子中，
我們計算了模型的輸出，然後將此輸出送入交叉熵損失。
從數學上講，這是一件完全合理的事情。
然而，從計算角度來看，指數可能會造成數值穩定性問題。

回想一下，softmax函式$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，
其中$\hat y_j$是預測的機率分佈。
$o_j$是未規範化的預測$\mathbf{o}$的第$j$個元素。
如果$o_k$中的一些數值非常大，
那麼$\exp(o_k)$可能大於資料型別容許的最大數字，即*上溢*（overflow）。
這將使分母或分子變為`inf`（無窮大），
最後得到的是0、`inf`或`nan`（不是數字）的$\hat y_j$。
在這些情況下，我們無法得到一個明確定義的交叉熵值。

解決這個問題的一個技巧是：
在繼續softmax計算之前，先從所有$o_k$中減去$\max(o_k)$。
這裡可以看到每個$o_k$按常數進行的移動不會改變softmax的回傳值：

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$


在減法和規範化步驟之後，可能有些$o_j - \max(o_k)$具有較大的負值。
由於精度受限，$\exp(o_j - \max(o_k))$將有接近零的值，即*下溢*（underflow）。
這些值可能會四捨五入為零，使$\hat y_j$為零，
並且使得$\log(\hat y_j)$的值為`-inf`。
反向傳播幾步後，我們可能會發現自己面對一螢幕可怕的`nan`結果。

儘管我們要計算指數函式，但我們最終在計算交叉熵損失時會取它們的對數。
透過將softmax和交叉熵結合在一起，可以避免反向傳播過程中可能會困擾我們的數值穩定性問題。
如下麵的等式所示，我們避免計算$\exp(o_j - \max(o_k))$，
而可以直接使用$o_j - \max(o_k)$，因為$\log(\exp(\cdot))$被抵消了。

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

我們也希望保留傳統的softmax函式，以備我們需要評估透過模型輸出的機率。
但是，我們沒有將softmax機率傳遞到損失函式中，
而是[**在交叉熵損失函式中傳遞未規範化的預測，並同時計算softmax及其對數**]，
這是一種類似["LogSumExp技巧"](https://en.wikipedia.org/wiki/LogSumExp)的聰明方式。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch, paddle
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## 最佳化演算法

在這裡，我們(**使用學習率為0.1的小批次隨機梯度下降作為最佳化演算法**)。
這與我們線上性迴歸例子中的相同，這說明了最佳化器的普適性。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
```

## 訓練

接下來我們[**呼叫**] :numref:`sec_softmax_scratch`中(~~之前~~)
(**定義的訓練函式來訓練模型**)。

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

和以前一樣，這個演算法使結果收斂到一個相當高的精度，而且這次的程式碼比之前更精簡了。

## 小結

* 使用深度學習框架的高階API，我們可以更簡潔地實現softmax迴歸。
* 從計算的角度來看，實現softmax迴歸比較複雜。在許多情況下，深度學習框架在這些著名的技巧之外採取了額外的預防措施，來確保數值的穩定性。這使我們避免了在實踐中從零開始編寫模型時可能遇到的陷阱。

## 練習

1. 嘗試調整超引數，例如批次大小、迭代週期數和學習率，並檢視結果。
1. 增加迭代週期的數量。為什麼測試精度會在一段時間後降低？我們怎麼解決這個問題？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1794)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1793)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1792)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11761)
:end_tab:
