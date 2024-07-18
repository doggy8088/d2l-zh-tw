# 影象卷積
:label:`sec_conv_layer`

上節我們解析了卷積層的原理，現在我們看看它的實際應用。由於卷積神經網路的設計是用於探索影象資料，本節我們將以影象為例。

## 互相關運算

嚴格來說，卷積層是個錯誤的叫法，因為它所表達的運算其實是*互相關運算*（cross-correlation），而不是卷積運算。
根據 :numref:`sec_why-conv`中的描述，在卷積層中，輸入張量和核張量透過(**互相關運算**)產生輸出張量。

首先，我們暫時忽略通道（第三維）這一情況，看看如何處理二維影象資料和隱藏表示。在 :numref:`fig_correlation`中，輸入是高度為$3$、寬度為$3$的二維張量（即形狀為$3 \times 3$）。卷積核的高度和寬度都是$2$，而卷積核視窗（或卷積視窗）的形狀由核心的高度和寬度決定（即$2 \times 2$）。

![二維互相關運算。陰影部分是第一個輸出元素，以及用於計算輸出的輸入張量元素和核張量元素：$0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

在二維互相關運算中，卷積視窗從輸入張量的左上角開始，從左到右、從上到下滑動。
當卷積視窗滑動到新一個位置時，包含在該視窗中的部分張量與卷積核張量進行按元素相乘，得到的張量再求和得到一個單一的標量值，由此我們得出了這一位置的輸出張量值。
在如上例子中，輸出張量的四個元素由二維互相關運算得到，這個輸出高度為$2$、寬度為$2$，如下所示：

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

注意，輸出大小略小於輸入大小。這是因為卷積核的寬度和高度大於1，
而卷積核只與影象中每個大小完全適合的位置進行互相關運算。
所以，輸出大小等於輸入大小$n_h \times n_w$減去卷積核大小$k_h \times k_w$，即：

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

這是因為我們需要足夠的空間在影象上『移動』卷積核。稍後，我們將看到如何透過在影象邊界周圍填充零來保證有足夠的空間移動卷積核，從而保持輸出大小不變。
接下來，我們在`corr2d`函式中實現如上過程，該函式接受輸入張量`X`和卷積核張量`K`，並回傳輸出張量`Y`。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
def corr2d(X, K):  #@save
    """計算二維互相關運算"""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """計算二維互相關運算"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

透過 :numref:`fig_correlation`的輸入張量`X`和卷積核張量`K`，我們來[**驗證上述二維互相關運算的輸出**]。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 卷積層

卷積層對輸入和卷積核權重進行互相關運算，並在新增標量偏置之後產生輸出。
所以，卷積層中的兩個被訓練的引數是卷積核權重和標量偏置。
就像我們之前隨機初始化全連線層一樣，在訓練基於卷積層的模型時，我們也隨機初始化卷積核權重。

基於上面定義的`corr2d`函式[**實現二維卷積層**]。在`__init__`建構式函式中，將`weight`和`bias`宣告為兩個模型引數。前向傳播函式呼叫`corr2d`函式並新增偏置。

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

```{.python .input}
#@tab paddle
class Conv2D(nn.Layer):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = paddle.ParamAttr(paddle.rand(kernel_size))
        self.bias = paddle.ParamAttr(paddle.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

高度和寬度分別為$h$和$w$的卷積核可以被稱為$h \times w$卷積或$h \times w$卷積核。
我們也將帶有$h \times w$卷積核的卷積層稱為$h \times w$卷積層。

## 影象中目標的邊緣檢測

如下是[**卷積層的一個簡單應用：**]透過找到畫素變化的位置，來(**檢測影象中不同顏色的邊緣**)。
首先，我們構造一個$6\times 8$畫素的黑白影象。中間四列為黑色（$0$），其餘畫素為白色（$1$）。

```{.python .input}
#@tab mxnet, pytorch, paddle
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

接下來，我們構造一個高度為$1$、寬度為$2$的卷積核`K`。當進行互相關運算時，如果水平相鄰的兩元素相同，則輸出為零，否則輸出為非零。

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

現在，我們對引數`X`（輸入）和`K`（卷積核）執行互相關運算。
如下所示，[**輸出`Y`中的1代表從白色到黑色的邊緣，-1代表從黑色到白色的邊緣**]，其他情況的輸出為$0$。

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

現在我們將輸入的二維影象轉置，再進行如上的互相關運算。
其輸出如下，之前檢測到的垂直邊緣消失了。
不出所料，這個[**卷積核`K`只可以檢測垂直邊緣**]，無法檢測水平邊緣。

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## 學習卷積核

如果我們只需尋找黑白邊緣，那麼以上`[1, -1]`的邊緣檢測器足以。然而，當有了更復雜數值的卷積核，或者連續的卷積層時，我們不可能手動設計濾波器。那麼我們是否可以[**學習由`X`生成`Y`的卷積核**]呢？

現在讓我們看看是否可以透過僅檢視『輸入-輸出』對來學習由`X`生成`Y`的卷積核。
我們先構造一個卷積層，並將其卷積核初始化為隨機張量。接下來，在每次迭代中，我們比較`Y`與卷積層輸出的平方誤差，然後計算梯度來更新卷積核。為了簡單起見，我們在此使用內建的二維卷積層，並忽略偏置。

```{.python .input}
# 構造一個二維卷積層，它具有1個輸出通道和形狀為（1，2）的卷積核
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# 這個二維卷積層使用四維輸入和輸出格式（批次大小、通道、高度、寬度），
# 其中批次大小和通道數都為1


X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # 學習率

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # 迭代卷積核
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# 構造一個二維卷積層，它具有1個輸出通道和形狀為（1，2）的卷積核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 這個二維卷積層使用四維輸入和輸出格式（批次大小、通道、高度、寬度），
# 其中批次大小和通道數都為1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 學習率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷積核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# 構造一個二維卷積層，它具有1個輸出通道和形狀為（1，2）的卷積核
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# 這個二維卷積層使用四維輸入和輸出格式（批次大小、高度、寬度、通道），
# 其中批次大小和通道數都為1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # 學習率

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # 迭代卷積核
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {tf.reduce_sum(l):.3f}')
```

```{.python .input}
#@tab paddle
# 構造一個二維卷積層，它具有1個輸出通道和形狀為（1，2）的卷積核
conv2d = nn.Conv2D(1, 1, kernel_size=(1, 2))

# 這個二維卷積層使用四維輸入和輸出格式（批次大小、通道、高度、寬度），
# 其中批次大小和通道數都為1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 學習率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.clear_gradients()
    l.sum().backward()
    # 迭代卷積核
    with paddle.no_grad():
        conv2d.weight[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum().item():.3f}')
```

在$10$次迭代之後，誤差已經降到足夠低。現在我們來看看我們[**所學的卷積核的權重張量**]。

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

```{.python .input}
#@tab paddle
d2l.reshape(conv2d.weight, (1, 2))
```

細心的讀者一定會發現，我們學習到的卷積核權重非常接近我們之前定義的卷積核`K`。

## 互相關和卷積

回想一下我們在 :numref:`sec_why-conv`中觀察到的互相關和卷積運算之間的對應關係。
為了得到正式的*卷積*運算輸出，我們需要執行 :eqref:`eq_2d-conv-discrete`中定義的嚴格卷積運算，而不是互相關運算。
幸運的是，它們差別不大，我們只需水平和垂直翻轉二維卷積核張量，然後對輸入張量執行*互相關*運算。

值得注意的是，由於卷積核是從資料中學習到的，因此無論這些層執行嚴格的卷積運算還是互相關運算，卷積層的輸出都不會受到影響。
為了說明這一點，假設卷積層執行*互相關*運算並學習 :numref:`fig_correlation`中的卷積核，該卷積核在這裡由矩陣$\mathbf{K}$表示。
假設其他條件不變，當這個層執行嚴格的*卷積*時，學習的卷積核$\mathbf{K}'$在水平和垂直翻轉之後將與$\mathbf{K}$相同。
也就是說，當卷積層對 :numref:`fig_correlation`中的輸入和$\mathbf{K}'$執行嚴格*卷積*運算時，將得到與互相關運算 :numref:`fig_correlation`中相同的輸出。

為了與深度學習文獻中的標準術語保持一致，我們將繼續把『互相關運算』稱為卷積運算，儘管嚴格地說，它們略有不同。
此外，對於卷積核張量上的權重，我們稱其為*元素*。

## 特徵對映和感受野

如在 :numref:`subsec_why-conv-channels`中所述， :numref:`fig_correlation`中輸出的卷積層有時被稱為*特徵對映*（feature map），因為它可以被視為一個輸入對映到下一層的空間維度的轉換器。
在卷積神經網路中，對於某一層的任意元素$x$，其*感受野*（receptive field）是指在前向傳播期間可能影響$x$計算的所有元素（來自所有先前層）。

請注意，感受野可能大於輸入的實際大小。讓我們用 :numref:`fig_correlation`為例來解釋感受野：
給定$2 \times 2$卷積核，陰影輸出元素值$19$的感受野是輸入陰影部分的四個元素。
假設之前輸出為$\mathbf{Y}$，其大小為$2 \times 2$，現在我們在其後附加一個卷積層，該卷積層以$\mathbf{Y}$為輸入，輸出單個元素$z$。
在這種情況下，$\mathbf{Y}$上的$z$的感受野包括$\mathbf{Y}$的所有四個元素，而輸入的感受野包括最初所有九個輸入元素。
因此，當一個特徵圖中的任意元素需要檢測更廣區域的輸入特徵時，我們可以建立一個更深的網路。

## 小結

* 二維卷積層的核心計算是二維互相關運算。最簡單的形式是，對二維輸入資料和卷積核執行互相關操作，然後新增一個偏置。
* 我們可以設計一個卷積核來檢測影象的邊緣。
* 我們可以從資料中學習卷積核的引數。
* 學習卷積核時，無論用嚴格卷積運算或互相關運算，卷積層的輸出不會受太大影響。
* 當需要檢測輸入特徵中更廣區域時，我們可以建立一個更深的卷積網路。

## 練習

1. 建立一個具有對角線邊緣的影象`X`。
    1. 如果將本節中舉例的卷積核`K`應用於`X`，會發生什麼情況？
    1. 如果轉置`X`會發生什麼？
    1. 如果轉置`K`會發生什麼？
1. 在我們建立的`Conv2D`自動求導時，有什麼錯誤訊息？
1. 如何透過改變輸入張量和卷積核張量，將互相關運算表示為矩陣乘法？
1. 手工設計一些卷積核。
    1. 二階導數的核的形式是什麼？
    1. 積分的核的形式是什麼？
    1. 得到$d$次導數的最小核的大小是多少？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1849)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1848)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1847)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11783)
:end_tab:
