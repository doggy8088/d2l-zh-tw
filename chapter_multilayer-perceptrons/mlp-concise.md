# 多層感知機的簡潔實現
:label:`sec_mlp_concise`

本節將介紹(**透過高階API更簡潔地實現多層感知機**)。

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

## 模型

與softmax迴歸的簡潔實現（ :numref:`sec_softmax_concise`）相比，
唯一的區別是我們新增了2個全連線層（之前我們只新增了1個全連線層）。
第一層是[**隱藏層**]，它(**包含256個隱藏單元，並使用了ReLU啟用函式**)。
第二層是輸出層。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

```{.python .input}
#@tab paddle
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


for layer in net:
    if type(layer) == nn.Linear:
        weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))
        layer.weight_attr = weight_attr
```

[**訓練過程**]的實現與我們實現softmax迴歸時完全相同，
這種模組化設計使我們能夠將與模型架構有關的內容獨立出來。

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab paddle
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 小結

* 我們可以使用高階API更簡潔地實現多層感知機。
* 對於相同的分類問題，多層感知機的實現與softmax迴歸的實現相同，只是多層感知機的實現裡增加了帶有啟用函式的隱藏層。

## 練習

1. 嘗試新增不同數量的隱藏層（也可以修改學習率），怎麼樣設定效果最好？
1. 嘗試不同的啟用函式，哪個效果最好？
1. 嘗試不同的方案來初始化權重，什麼方法效果最好？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1803)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1802)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1801)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11770)
:end_tab:
