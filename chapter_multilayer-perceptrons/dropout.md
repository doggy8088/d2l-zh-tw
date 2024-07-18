# 暫退法（Dropout）
:label:`sec_dropout`

在 :numref:`sec_weight_decay` 中，
我們介紹了透過懲罰權重的$L_2$範數來正則化統計模型的經典方法。
在機率角度看，我們可以透過以下論證來證明這一技術的合理性：
我們已經假設了一個先驗，即權重的值取自均值為0的高斯分佈。
更直觀的是，我們希望模型深度挖掘特徵，即將其權重分散到許多特徵中，
而不是過於依賴少數潛在的虛假關聯。

## 重新審視過擬合

當面對更多的特徵而樣本不足時，線性模型往往會過擬合。
相反，當給出更多樣本而不是特徵，通常線性模型不會過擬合。
不幸的是，線性模型泛化的可靠性是有代價的。
簡單地說，線性模型沒有考慮到特徵之間的互動作用。
對於每個特徵，線性模型必須指定正的或負的權重，而忽略其他特徵。

泛化性和靈活性之間的這種基本權衡被描述為*偏差-方差權衡*（bias-variance tradeoff）。
線性模型有很高的偏差：它們只能表示一小類函式。
然而，這些模型的方差很低：它們在不同的隨機數據樣本上可以得出相似的結果。

深度神經網路位於偏差-方差譜的另一端。
與線性模型不同，神經網路並不侷限於單獨檢視每個特徵，而是學習特徵之間的互動。
例如，神經網路可能推斷『奈及利亞』和『西聯匯款』一起出現在電子郵件中表示垃圾郵件，
但單獨出現則不表示垃圾郵件。

即使我們有比特徵多得多的樣本，深度神經網路也有可能過擬合。
2017年，一組研究人員透過在隨機標記的影象上訓練深度網路。
這展示了神經網路的極大靈活性，因為人類很難將輸入和隨機標記的輸出聯絡起來，
但透過隨機梯度下降最佳化的神經網路可以完美地標記訓練集中的每一幅影象。
想一想這意味著什麼？
假設標籤是隨機均勻分配的，並且有10個類別，那麼分類器在測試資料上很難取得高於10%的精度，
那麼這裡的泛化差距就高達90%，如此嚴重的過擬合。

深度網路的泛化性質令人費解，而這種泛化性質的數學基礎仍然是懸而未決的研究問題。
我們鼓勵喜好研究理論的讀者更深入地研究這個主題。
本節，我們將著重對實際工具的探究，這些工具傾向於改進深層網路的泛化性。

## 擾動的穩健性

在探究泛化性之前，我們先來定義一下什麼是一個『好』的預測模型？
我們期待『好』的預測模型能在未知的資料上有很好的表現：
經典泛化理論認為，為了縮小訓練和測試效能之間的差距，應該以簡單的模型為目標。
簡單性以較小維度的形式展現，
我們在 :numref:`sec_model_selection` 討論線性模型的單項式函式時探討了這一點。
此外，正如我們在 :numref:`sec_weight_decay` 中討論權重衰減（$L_2$正則化）時看到的那樣，
引數的範數也代表了一種有用的簡單性度量。

簡單性的另一個角度是平滑性，即函式不應該對其輸入的微小變化敏感。
例如，當我們對影象進行分類時，我們預計向畫素新增一些隨機噪聲應該是基本無影響的。
1995年，克里斯托弗·畢曉普證明了
具有輸入噪聲的訓練等價於Tikhonov正則化 :cite:`Bishop.1995`。
這項工作用數學證實了『要求函式光滑』和『要求函式對輸入的隨機噪聲具有適應性』之間的聯絡。

然後在2014年，斯里瓦斯塔瓦等人 :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`
就如何將畢曉普的想法應用於網路的內部層提出了一個想法：
在訓練過程中，他們建議在計算後續層之前向網路的每一層注入噪聲。
因為當訓練一個有多層的深層網路時，注入噪聲只會在輸入-輸出對映上增強平滑性。

這個想法被稱為*暫退法*（dropout）。
暫退法在前向傳播過程中，計算每一內部層的同時注入噪聲，這已經成為訓練神經網路的常用技術。
這種方法之所以被稱為*暫退法*，因為我們從表面上看是在訓練過程中丟棄（drop out）一些神經元。
在整個訓練過程的每一次迭代中，標準暫退法包括在計算下一層之前將當前層中的一些節點置零。

需要說明的是，暫退法的原始論文提到了一個關於有性繁殖的類比：
神經網路過擬合與每一層都依賴於前一層啟用值相關，稱這種情況為『共適應性』。
作者認為，暫退法會破壞共適應性，就像有性生殖會破壞共適應的基因一樣。

那麼關鍵的挑戰就是如何注入這種噪聲。
一種想法是以一種*無偏向*（unbiased）的方式注入噪聲。
這樣在固定住其他層時，每一層的期望值等於沒有噪音時的值。

在畢曉普的工作中，他將高斯噪聲新增到線性模型的輸入中。
在每次訓練迭代中，他將從均值為零的分佈$\epsilon \sim \mathcal{N}(0,\sigma^2)$
取樣噪聲新增到輸入$\mathbf{x}$，
從而產生擾動點$\mathbf{x}' = \mathbf{x} + \epsilon$，
預期是$E[\mathbf{x}'] = \mathbf{x}$。

在標準暫退法正則化中，透過按保留（未丟棄）的節點的分數進行規範化來消除每一層的偏差。
換言之，每個中間活性值$h$以*暫退機率*$p$由隨機變數$h'$替換，如下所示：

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 機率為 } p \\
    \frac{h}{1-p} & \text{ 其他情況}
\end{cases}
\end{aligned}
$$

根據此模型的設計，其期望值保持不變，即$E[h'] = h$。

## 實踐中的暫退法

回想一下 :numref:`fig_mlp`中帶有1個隱藏層和5個隱藏單元的多層感知機。
當我們將暫退法應用到隱藏層，以$p$的機率將隱藏單元置為零時，
結果可以看作一個只包含原始神經元子集的網路。
比如在 :numref:`fig_dropout2`中，刪除了$h_2$和$h_5$，
因此輸出的計算不再依賴於$h_2$或$h_5$，並且它們各自的梯度在執行反向傳播時也會消失。
這樣，輸出層的計算不能過度依賴於$h_1, \ldots, h_5$的任何一個元素。

![dropout前後的多層感知機](../img/dropout2.svg)
:label:`fig_dropout2`

通常，我們在測試時不用暫退法。
給定一個訓練好的模型和一個新的樣本，我們不會丟棄任何節點，因此不需要標準化。
然而也有一些例外：一些研究人員在測試時使用暫退法，
用於估計神經網路預測的『不確定性』：
如果透過許多不同的暫退法遮蓋後得到的預測結果都是一致的，那麼我們可以說網路發揮更穩定。

## 從零開始實現

要實現單層的暫退法函式，
我們從均勻分佈$U[0, 1]$中抽取樣本，樣本數與這層神經網路的維度一致。
然後我們保留那些對應樣本大於$p$的節點，把剩下的丟棄。

在下面的程式碼中，(**我們實現 `dropout_layer` 函式，
該函式以`dropout`的機率丟棄張量輸入`X`中的元素**)，
如上所述重新縮放剩餘部分：將剩餘部分除以`1.0-dropout`。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情況中，所有元素都被丟棄
    if dropout == 1:
        return np.zeros_like(X)
    # 在本情況中，所有元素都被保留
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情況中，所有元素都被丟棄
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情況中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情況中，所有元素都被丟棄
    if dropout == 1:
        return tf.zeros_like(X)
    # 在本情況中，所有元素都被保留
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
from d2l import paddle as d2l

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情況中，所有元素都被丟棄。
    if dropout == 1:
        return paddle.zeros_like(X)
    # 在本情況中，所有元素都被保留。
    if dropout == 0:
        return X
    
    mask = (paddle.to_tensor(paddle.uniform(X.shape)) > dropout).astype('float32')
    return mask * X / (1.0 - dropout)
```

我們可以透過下面幾個例子來[**測試`dropout_layer`函式**]。
我們將輸入`X`透過暫退法操作，暫退機率分別為0、0.5和1。

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab paddle
X= paddle.arange(16, dtype = paddle.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### 定義模型引數

同樣，我們使用 :numref:`sec_fashion_mnist`中引入的Fashion-MNIST資料集。
我們[**定義具有兩個隱藏層的多層感知機，每個隱藏層包含256個單元**]。

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch, paddle
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### 定義模型

我們可以將暫退法應用於每個隱藏層的輸出（在啟用函式之後），
並且可以為每一層分別設定暫退機率：
常見的技巧是在靠近輸入層的地方設定較低的暫退機率。
下面的模型將第一個和第二個隱藏層的暫退機率分別設定為0.2和0.5，
並且暫退法只在訓練期間有效。

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # 只有在訓練模型時才使用dropout
    if autograd.is_training():
        # 在第一個全連線層之後新增一個dropout層
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # 在第二個全連線層之後新增一個dropout層
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在訓練模型時才使用dropout
        if self.training == True:
            # 在第一個全連線層之後新增一個dropout層
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二個全連線層之後新增一個dropout層
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        # 只有在訓練模型時才使用dropout
        if training:
            # 在第一個全連線層之後新增一個dropout層
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            # 在第二個全連線層之後新增一個dropout層
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab paddle
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Layer):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在訓練模型時才使用dropout
        if self.training == True:
            # 在第一個全連線層之後新增一個dropout層
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二個全連線層之後新增一個dropout層
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

### [**訓練和測試**]

這類似於前面描述的多層感知機訓練和測試。

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab paddle
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = paddle.optimizer.SGD(learning_rate=lr, parameters=net.parameters())
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## [**簡潔實現**]

對於深度學習框架的高階API，我們只需在每個全連線層之後新增一個`Dropout`層，
將暫退機率作為唯一的引數傳遞給它的建構式函式。
在訓練時，`Dropout`層將根據指定的暫退機率隨機丟棄上一層的輸出（相當於下一層的輸入）。
在測試時，`Dropout`層僅傳遞資料。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # 在第一個全連線層之後新增一個dropout層
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # 在第二個全連線層之後新增一個dropout層
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一個全連線層之後新增一個dropout層
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二個全連線層之後新增一個dropout層
        nn.Dropout(dropout2),
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
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # 在第一個全連線層之後新增一個dropout層
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # 在第二個全連線層之後新增一個dropout層
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
#@tab paddle
weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(std=0.01)) 

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256, weight_attr=weight_attr),
        nn.ReLU(),
        # 在第一個全連線層之後新增一個dropout層
        nn.Dropout(dropout1),
        nn.Linear(256, 256, weight_attr=weight_attr),
        nn.ReLU(),
        # 在第二個全連線層之後新增一個dropout層
        nn.Dropout(dropout2),
        nn.Linear(256, 10, weight_attr=weight_attr))
```

接下來，我們[**對模型進行訓練和測試**]。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.SGD(learning_rate=0.5, parameters=net.parameters())
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 小結

* 暫退法在前向傳播過程中，計算每一內部層的同時丟棄一些神經元。
* 暫退法可以避免過擬合，它通常與控制權重向量的維數和大小結合使用的。
* 暫退法將活性值$h$替換為具有期望值$h$的隨機變數。
* 暫退法僅在訓練期間使用。

## 練習

1. 如果更改第一層和第二層的暫退法機率，會發生什麼情況？具體地說，如果交換這兩個層，會發生什麼情況？設計一個實驗來回答這些問題，定量描述該結果，並總結定性的結論。
1. 增加訓練輪數，並將使用暫退法和不使用暫退法時獲得的結果進行比較。
1. 當應用或不應用暫退法時，每個隱藏層中啟用值的方差是多少？繪製一個曲線圖，以顯示這兩個模型的每個隱藏層中啟用值的方差是如何隨時間變化的。
1. 為什麼在測試時通常不使用暫退法？
1. 以本節中的模型為例，比較使用暫退法和權重衰減的效果。如果同時使用暫退法和權重衰減，會發生什麼情況？結果是累加的嗎？收益是否減少（或者說更糟）？它們互相抵消了嗎？
1. 如果我們將暫退法應用到權重矩陣的各個權重，而不是啟用值，會發生什麼？
1. 發明另一種用於在每一層注入隨機噪聲的技術，該技術不同於標準的暫退法技術。嘗試開發一種在Fashion-MNIST資料集（對於固定架構）上效能優於暫退法的方法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1812)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1813)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1811)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11774)
:end_tab:
