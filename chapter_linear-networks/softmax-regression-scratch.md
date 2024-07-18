# softmax迴歸的從零開始實現
:label:`sec_softmax_scratch`

(**就像我們從零開始實現線性迴歸一樣，**)
我們認為softmax迴歸也是重要的基礎，因此(**應該知道實現softmax迴歸的細節**)。
本節我們將使用剛剛在 :numref:`sec_fashion_mnist`中引入的Fashion-MNIST資料集，
並設定資料迭代器的批次大小為256。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型引數

和之前線性迴歸的例子一樣，這裡的每個樣本都將用固定長度的向量表示。
原始資料集中的每個樣本都是$28 \times 28$的影象。
本節[**將展平每個影象，把它們看作長度為784的向量。**]
在後麵的章節中，我們將討論能夠利用影象空間結構的特徵，
但現在我們暫時只把每個畫素位置看作一個特徵。

回想一下，在softmax迴歸中，我們的輸出與類別一樣多。
(**因為我們的資料集有10個類別，所以網路輸出維度為10**)。
因此，權重將構成一個$784 \times 10$的矩陣，
偏置將構成一個$1 \times 10$的行向量。
與線性迴歸一樣，我們將使用常態分佈初始化我們的權重`W`，偏置初始化為0。

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
#@tab paddle
num_inputs = 784
num_outputs = 10

W = paddle.normal(0, 0.01, shape=(num_inputs, num_outputs))
b = paddle.zeros(shape=(num_outputs,))
W.stop_gradient=False
b.stop_gradient=False
```

## 定義softmax操作

在實現softmax迴歸模型之前，我們簡要回顧一下`sum`運算符如何沿著張量中的特定維度工作。
如 :numref:`subseq_lin-alg-reduction`和
 :numref:`subseq_lin-alg-non-reduction`所述，
 [**給定一個矩陣`X`，我們可以對所有元素求和**]（預設情況下）。
 也可以只求同一個軸上的元素，即同一列（軸0）或同一行（軸1）。
 如果`X`是一個形狀為`(2, 3)`的張量，我們對列進行求和，
 則結果將是一個具有形狀`(3,)`的向量。
 當呼叫`sum`運算符時，我們可以指定保持在原始張量的軸數，而不折疊求和的維度。
 這將產生一個具有形狀`(1, 3)`的二維張量。

```{.python .input}
#@tab pytorch, paddle
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

回想一下，[**實現softmax**]由三個步驟組成：

1. 對每個項求冪（使用`exp`）；
1. 對每一行求和（小批次中每個樣本是一行），得到每個樣本的規範化常數；
1. 將每一行除以其規範化常數，確保結果的和為1。

在檢視程式碼之前，我們回顧一下這個表示式：

(**
$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$
**)

分母或規範化常數，有時也稱為*配分函式*（其對數稱為對數-配分函式）。
該名稱來自[統計物理學](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))中一個模擬粒子群分佈的方程。

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # 這裡應用了廣播機制
```

```{.python .input}
#@tab pytorch, paddle
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # 這裡應用了廣播機制
```

正如上述程式碼，對於任何隨機輸入，[**我們將每個元素變成一個非負數。
此外，依據機率原理，每行總和為1**]。

```{.python .input}
#@tab mxnet, pytorch, paddle
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

注意，雖然這在數學上看起來是正確的，但我們在程式碼實現中有點草率。
矩陣中的非常大或非常小的元素可能造成數值上溢或下溢，但我們沒有採取措施來防止這點。

## 定義模型

定義softmax操作後，我們可以[**實現softmax迴歸模型**]。
下面的程式碼定義了輸入如何透過網路對映到輸出。
注意，將資料傳遞到模型之前，我們使用`reshape`函式將每張原始影象展平為向量。

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## 定義損失函式

接下來，我們實現 :numref:`sec_softmax`中引入的交叉熵損失函式。
這可能是深度學習中最常見的損失函式，因為目前分類問題的數量遠遠超過迴歸問題的數量。

回顧一下，交叉熵採用真實標籤的預測機率的負對數似然。
這裡我們不使用Python的for迴圈迭代預測（這往往是低效的），
而是透過一個運算符選擇所有元素。
下面，我們[**建立一個資料樣本`y_hat`，其中包含2個樣本在3個類別的預測機率，
以及它們對應的標籤`y`。**]
有了`y`，我們知道在第一個樣本中，第一類是正確的預測；
而在第二個樣本中，第三類是正確的預測。
然後(**使用`y`作為`y_hat`中機率的索引**)，
我們選擇第一個樣本中第一個類別的機率和第二個樣本中第三個類別的機率。

```{.python .input}
#@tab mxnet, pytorch, paddle
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

現在我們只需一行程式碼就可以[**實現交叉熵損失函式**]。

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab paddle
def cross_entropy(y_hat, y):
    return - paddle.log(y_hat[[i for i in range(len(y_hat))], y.squeeze()])

cross_entropy(y_hat, y)
```

## 分類精度

給定預測機率分佈`y_hat`，當我們必須輸出硬預測（hard prediction）時，
我們通常選擇預測機率最高的類。
許多應用都要求我們做出選擇。如Gmail必須將電子郵件分類為『Primary（主要郵件）』、
『Social（社交郵件）』『Updates（更新郵件）』或『Forums（論壇郵件）』。
Gmail做分類時可能在內部估計機率，但最終它必須在類別中選擇一個。

當預測與標籤分類`y`一致時，即是正確的。
分類精度即正確預測數量與總預測數量之比。
雖然直接最佳化精度可能很困難（因為精度的計算不可導），
但精度通常是我們最關心的效能衡量標準，我們在訓練分類器時幾乎總會關注它。

為了計算精度，我們執行以下操作。
首先，如果`y_hat`是矩陣，那麼假定第二個維度儲存每個類別的預測分數。
我們使用`argmax`獲得每行中最大元素的索引來獲得預測類別。
然後我們[**將預測類別與真實`y`元素進行比較**]。
由於等式運算符『`==`』對資料型別很敏感，
因此我們將`y_hat`的資料型別轉換為與`y`的資料型別一致。
結果是一個包含0（錯）和1（對）的張量。
最後，我們求和會得到正確預測的數量。

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """計算預測正確的數量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

```{.python .input}
#@tab paddle
#@save
def accuracy(y_hat, y):
    """計算預測正確的數量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    if len(y_hat.shape) < len(y.shape):
        cmp = y_hat.astype(y.dtype) == y.squeeze()
    else:
        cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum())
```

我們將繼續使用之前定義的變數`y_hat`和`y`分別作為預測的機率分佈和標籤。
可以看到，第一個樣本的預測類別是2（該行的最大元素為0.6，索引為2），這與實際標籤0不一致。
第二個樣本的預測類別是2（該行的最大元素為0.5，索引為2），這與實際標籤2一致。
因此，這兩個樣本的分類精度率為0.5。

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

同樣，對於任意資料迭代器`data_iter`可訪問的資料集，
[**我們可以評估在任意模型`net`的精度**]。

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """計算在指定資料集上模型的精度"""
    metric = Accumulator(2)  # 正確預測數、預測總數
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """計算在指定資料集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 將模型設定為評估模式
    metric = Accumulator(2)  # 正確預測數、預測總數
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab paddle
#@save
def evaluate_accuracy(net, data_iter):
    """計算在指定資料集上模型的精度"""
    if isinstance(net, paddle.nn.Layer):
        net.eval()  # 將模型設定為評估模式
    metric = Accumulator(2)  # 正確預測數、預測總數
    with paddle.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

這裡定義一個實用程式類`Accumulator`，用於對多個變數進行累加。
在上面的`evaluate_accuracy`函式中，
我們在(**`Accumulator`實例中建立了2個變數，
分別用於儲存正確預測的數量和預測的總數量**)。
當我們遍歷資料集時，兩者都將隨著時間的推移而累加。

```{.python .input}
#@tab all
class Accumulator:  #@save
    """在n個變數上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

由於我們使用隨機權重初始化`net`模型，
因此該模型的精度應接近於隨機猜測。
例如在有10個類別情況下的精度為0.1。

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## 訓練

在我們看過 :numref:`sec_linear_scratch`中的線性迴歸實現，
[**softmax迴歸的訓練**]過程程式碼應該看起來非常眼熟。
在這裡，我們重構訓練過程的實現以使其可重複使用。
首先，我們定義一個函式來訓練一個迭代週期。
請注意，`updater`是更新模型引數的常用函式，它接受批次大小作為引數。
它可以是`d2l.sgd`函式，也可以是框架的內建最佳化函式。

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """訓練模型一個迭代週期（定義見第3章）"""
    # 訓練損失總和、訓練準確度總和、樣本數
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # 計算梯度並更新引數
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # 回傳訓練損失和訓練精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """訓練模型一個迭代週期（定義見第3章）"""
    # 將模型設定為訓練模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 訓練損失總和、訓練準確度總和、樣本數
    metric = Accumulator(3)
    for X, y in train_iter:
        # 計算梯度並更新引數
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch內建的最佳化器和損失函式
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定製的最佳化器和損失函式
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 回傳訓練損失和訓練精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """訓練模型一個迭代週期（定義見第3章）"""
    # 訓練損失總和、訓練準確度總和、樣本數
    metric = Accumulator(3)
    for X, y in train_iter:
        # 計算梯度並更新引數
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras內建的損失接受的是（標籤，預測），這不同於使用者在本書中的實現。
            # 本書的實現接受（預測，標籤），例如我們上面實現的『交叉熵』
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras的loss預設回傳一個批次的平均損失
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # 回傳訓練損失和訓練精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab paddle
#@save
def train_epoch_ch3(net, train_iter, loss, updater):
    """訓練模型一個迭代週期（定義見第3章）"""
    # 將模型設定為訓練模式
    if isinstance(net, paddle.nn.Layer):
        net.train()
    # 訓練損失總和、訓練準確度總和、樣本數
    metric = Accumulator(3)

    for X, y in train_iter:
        # 計算梯度並更新引數
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, paddle.optimizer.Optimizer):
            # 使用PaddlePaddle內建的最佳化器和損失函式
            updater.clear_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定製的最佳化器和損失函式
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
```

在展示訓練函式的實現之前，我們[**定義一個在動畫中繪製資料的實用程式類**]`Animator`，
它能夠簡化本書其餘部分的程式碼。

```{.python .input}
#@tab all
class Animator:  #@save
    """在動畫中繪製資料"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地繪製多條線
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函式捕獲引數
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向圖表中新增多個資料點
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

接下來我們實現一個[**訓練函式**]，
它會在`train_iter`訪問到的訓練資料集上訓練一個模型`net`。
該訓練函式將會執行多個迭代週期（由`num_epochs`指定）。
在每個迭代週期結束時，利用`test_iter`訪問到的測試資料集對模型進行評估。
我們將利用`Animator`類來視覺化訓練進度。

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """訓練模型（定義見第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

作為一個從零開始的實現，我們使用 :numref:`sec_linear_scratch`中定義的
[**小批次隨機梯度下降來最佳化模型的損失函式**]，設定學習率為0.1。

```{.python .input}
#@tab mxnet, pytorch, paddle
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """用小批次隨機梯度下降法更新引數"""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

現在，我們[**訓練模型10個迭代週期**]。
請注意，迭代週期（`num_epochs`）和學習率（`lr`）都是可調節的超引數。
透過更改它們的值，我們可以提高模型的分類精度。

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 預測

現在訓練已經完成，我們的模型已經準備好[**對影象進行分類預測**]。
給定一系列影象，我們將比較它們的實際標籤（文字輸出的第一行）和模型預測（文字輸出的第二行）。

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """預測標籤（定義見第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## 小結

* 藉助softmax迴歸，我們可以訓練多分類的模型。
* 訓練softmax迴歸迴圈模型與訓練線性迴歸模型非常相似：先讀取資料，再定義模型和損失函式，然後使用最佳化演算法訓練模型。大多數常見的深度學習模型都有類似的訓練過程。

## 練習

1. 本節直接實現了基於數學定義softmax運算的`softmax`函式。這可能會導致什麼問題？提示：嘗試計算$\exp(50)$的大小。
1. 本節中的函式`cross_entropy`是根據交叉熵損失函式的定義實現的。它可能有什麼問題？提示：考慮對數的定義域。
1. 請想一個解決方案來解決上述兩個問題。
1. 回傳機率最大的分類標籤總是最優解嗎？例如，醫療診斷場景下可以這樣做嗎？
1. 假設我們使用softmax迴歸來預測下一個單詞，可選取的單詞數目過多可能會帶來哪些問題?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1791)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1789)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1790)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11760)
:end_tab:
