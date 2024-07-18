# 轉置卷積
:label:`sec_transposed_conv`

到目前為止，我們所見到的卷積神經網路層，例如卷積層（ :numref:`sec_conv_layer`）和彙聚層（ :numref:`sec_pooling`），通常會減少下采樣輸入影象的空間維度（高和寬）。
然而如果輸入和輸出影象的空間維度相同，在以畫素級分類的語義分割中將會很方便。
例如，輸出畫素所處的通道維可以保有輸入畫素在同一位置上的分類結果。

為了實現這一點，尤其是在空間維度被卷積神經網路層縮小後，我們可以使用另一種型別的卷積神經網路層，它可以增加上取樣中間層特徵圖的空間維度。
本節將介紹
*轉置卷積*（transposed convolution） :cite:`Dumoulin.Visin.2016`，
用於逆轉下采樣導致的空間尺寸減小。

```{.python .input}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import paddle
from paddle import nn
```

## 基本操作

讓我們暫時忽略通道，從基本的轉置卷積開始，設步幅為1且沒有填充。
假設我們有一個$n_h \times n_w$的輸入張量和一個$k_h \times k_w$的卷積核。
以步幅為1滑動卷積核視窗，每行$n_w$次，每列$n_h$次，共產生$n_h n_w$個中間結果。
每個中間結果都是一個$(n_h + k_h - 1) \times (n_w + k_w - 1)$的張量，初始化為0。
為了計算每個中間張量，輸入張量中的每個元素都要乘以卷積核，從而使所得的$k_h \times k_w$張量替換中間張量的一部分。
請注意，每個中間張量被替換部分的位置與輸入張量中元素的位置相對應。
最後，所有中間結果相加以獲得最終結果。

例如， :numref:`fig_trans_conv`解釋瞭如何為$2\times 2$的輸入張量計算卷積核為$2\times 2$的轉置卷積。

![卷積核為 $2\times 2$ 的轉置卷積。陰影部分是中間張量的一部分，也是用於計算的輸入和卷積核張量元素。 ](../img/trans_conv.svg)
:label:`fig_trans_conv`

我們可以對輸入矩陣`X`和卷積核矩陣`K`(**實現基本的轉置卷積運算**)`trans_conv`。

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

與透過卷積核『減少』輸入元素的常規卷積（在 :numref:`sec_conv_layer`中）相比，轉置卷積透過卷積核『廣播』輸入元素，從而產生大於輸入的輸出。
我們可以透過 :numref:`fig_trans_conv`來建立輸入張量`X`和卷積核張量`K`從而[**驗證上述實現輸出**]。
此實現是基本的二維轉置卷積運算。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

或者，當輸入`X`和卷積核`K`都是四維張量時，我們可以[**使用高階API獲得相同的結果**]。

```{.python .input}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

```{.python .input}
#@tab paddle
X, K = X.reshape([1, 1, 2, 2]), K.reshape([1, 1, 2, 2])
tconv = nn.Conv2DTranspose(1, 1, kernel_size=2, bias_attr=False)
K = paddle.create_parameter(shape=K.shape, dtype="float32", 
        default_initializer=paddle.nn.initializer.Assign(K))
tconv.weight = K
tconv(X)
```

## [**填充、步幅和多通道**]

與常規卷積不同，在轉置卷積中，填充被應用於的輸出（常規卷積將填充應用於輸入）。
例如，當將高和寬兩側的填充數指定為1時，轉置卷積的輸出中將刪除第一和最後的行與列。

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

```{.python .input}
#@tab paddle
tconv = nn.Conv2DTranspose(1, 1, kernel_size=2, padding=1, bias_attr=False)
tconv.weight = K
tconv(X)
```

在轉置卷積中，步幅被指定為中間結果（輸出），而不是輸入。
使用 :numref:`fig_trans_conv`中相同輸入和卷積核張量，將步幅從1更改為2會增加中間張量的高和權重，因此輸出張量在 :numref:`fig_trans_conv_stride2`中。

![卷積核為$2\times 2$，步幅為2的轉置卷積。陰影部分是中間張量的一部分，也是用於計算的輸入和卷積核張量元素。](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`

以下程式碼可以驗證 :numref:`fig_trans_conv_stride2`中步幅為2的轉置卷積的輸出。

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

```{.python .input}
#@tab paddle
tconv = nn.Conv2DTranspose(1, 1, kernel_size=2, stride=2, bias_attr=False)
tconv.weight = K
tconv(X)
```

對於多個輸入和輸出通道，轉置卷積與常規卷積以相同方式運作。
假設輸入有$c_i$個通道，且轉置卷積為每個輸入通道分配了一個$k_h\times k_w$的卷積核張量。
當指定多個輸出通道時，每個輸出通道將有一個$c_i\times k_h\times k_w$的卷積核。

同樣，如果我們將$\mathsf{X}$代入卷積層$f$來輸出$\mathsf{Y}=f(\mathsf{X})$，並建立一個與$f$具有相同的超引數、但輸出通道數量是$\mathsf{X}$中通道數的轉置卷積層$g$，那麼$g(Y)$的形狀將與$\mathsf{X}$相同。
下面的範例可以解釋這一點。

```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab paddle
X = paddle.rand(shape=(1, 10, 16, 16))
conv = nn.Conv2D(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.Conv2DTranspose(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**與矩陣變換的聯絡**]
:label:`subsec-connection-to-mat-transposition`

轉置卷積為何以矩陣變換命名呢？
讓我們首先看看如何使用矩陣乘法來實現卷積。
在下面的範例中，我們定義了一個$3\times 3$的輸入`X`和$2\times 2$卷積核`K`，然後使用`corr2d`函式計算卷積輸出`Y`。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

```{.python .input}
#@tab paddle
X = d2l.arange(9.0, dtype="float32").reshape((3, 3))
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

接下來，我們將卷積核`K`重寫為包含大量0的稀疏權重矩陣`W`。
權重矩陣的形狀是（$4$，$9$），其中非0元素來自卷積核`K`。

```{.python .input}
#@tab mxnet, pytorch
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

```{.python .input}
#@tab paddle
def kernel2matrix(K):
    k, W = d2l.zeros([5]), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

逐行連結輸入`X`，獲得了一個長度為9的向量。
然後，`W`的矩陣乘法和向量化的`X`給出了一個長度為4的向量。
重塑它之後，可以獲得與上面的原始卷積操作所得相同的結果`Y`：我們剛剛使用矩陣乘法實現了卷積。

```{.python .input}
#@tab mxnet, pytorch
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

```{.python .input}
#@tab paddle
Y == d2l.matmul(W, d2l.reshape(X, [-1])).reshape((2, 2))
```

同樣，我們可以使用矩陣乘法來實現轉置卷積。
在下面的範例中，我們將上面的常規卷積$2 \times 2$的輸出`Y`作為轉置卷積的輸入。
想要透過矩陣相乘來實現它，我們只需要將權重矩陣`W`的形狀轉置為$(9, 4)$。

```{.python .input}
#@tab mxnet, pytorch
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

```{.python .input}
#@tab paddle
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, [-1])).reshape((3, 3))
```

抽象來看，給定輸入向量$\mathbf{x}$和權重矩陣$\mathbf{W}$，卷積的前向傳播函式可以透過將其輸入與權重矩陣相乘並輸出向量$\mathbf{y}=\mathbf{W}\mathbf{x}$來實現。
由於反向傳播遵循鏈式法則和$\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$，卷積的反向傳播函式可以透過將其輸入與轉置的權重矩陣$\mathbf{W}^\top$相乘來實現。
因此，轉置卷積層能夠交換卷積層的正向傳播函式和反向傳播函式：它的正向傳播和反向傳播函式將輸入向量分別與$\mathbf{W}^\top$和$\mathbf{W}$相乘。

## 小結

* 與透過卷積核減少輸入元素的常規卷積相反，轉置卷積透過卷積核廣播輸入元素，從而產生形狀大於輸入的輸出。
* 如果我們將$\mathsf{X}$輸入卷積層$f$來獲得輸出$\mathsf{Y}=f(\mathsf{X})$並創造一個與$f$有相同的超引數、但輸出通道數是$\mathsf{X}$中通道數的轉置卷積層$g$，那麼$g(Y)$的形狀將與$\mathsf{X}$相同。
* 我們可以使用矩陣乘法來實現卷積。轉置卷積層能夠交換卷積層的正向傳播函式和反向傳播函式。

## 練習

1. 在 :numref:`subsec-connection-to-mat-transposition`中，卷積輸入`X`和轉置的卷積輸出`Z`具有相同的形狀。他們的數值也相同嗎？為什麼？
1. 使用矩陣乘法來實現卷積是否有效率？為什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3301)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3302)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11810)
:end_tab:
