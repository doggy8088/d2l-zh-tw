# 填充和步幅
:label:`sec_padding`

在前面的例子 :numref:`fig_correlation`中，輸入的高度和寬度都為$3$，卷積核的高度和寬度都為$2$，生成的輸出表徵的維數為$2\times2$。
正如我們在 :numref:`sec_conv_layer`中所概括的那樣，假設輸入形狀為$n_h\times n_w$，卷積核形狀為$k_h\times k_w$，那麼輸出形狀將是$(n_h-k_h+1) \times (n_w-k_w+1)$。
因此，卷積的輸出形狀取決於輸入形狀和卷積核的形狀。

還有什麼因素會影響輸出的大小呢？本節我們將介紹*填充*（padding）和*步幅*（stride）。假設以下情景：
有時，在應用了連續的卷積之後，我們最終得到的輸出遠小於輸入大小。這是由於卷積核的寬度和高度通常大於$1$所導致的。比如，一個$240 \times 240$畫素的影象，經過$10$層$5 \times 5$的卷積後，將減少到$200 \times 200$畫素。如此一來，原始影象的邊界丟失了許多有用訊息。而*填充*是解決此問題最有效的方法；
有時，我們可能希望大幅降低影象的寬度和高度。例如，如果我們發現原始的輸入解析度十分冗餘。*步幅*則可以在這類情況下提供幫助。

## 填充

如上所述，在應用多層卷積時，我們常常丟失邊緣畫素。
由於我們通常使用小卷積核，因此對於任何單個卷積，我們可能只會丟失幾個畫素。
但隨著我們應用許多連續卷積層，累積丟失的畫素數就多了。
解決這個問題的簡單方法即為*填充*（padding）：在輸入影象的邊界填充元素（通常填充元素是$0$）。
例如，在 :numref:`img_conv_pad`中，我們將$3 \times 3$輸入填充到$5 \times 5$，那麼它的輸出就增加為$4 \times 4$。陰影部分是第一個輸出元素以及用於輸出計算的輸入和核張量元素：
$0\times0+0\times1+0\times2+0\times3=0$。

![帶填充的二維互相關。](../img/conv-pad.svg)
:label:`img_conv_pad`

通常，如果我們新增$p_h$行填充（大約一半在頂部，一半在底部）和$p_w$列填充（左側大約一半，右側一半），則輸出形狀將為

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1)。$$

這意味著輸出的高度和寬度將分別增加$p_h$和$p_w$。

在許多情況下，我們需要設定$p_h=k_h-1$和$p_w=k_w-1$，使輸入和輸出具有相同的高度和寬度。
這樣可以在建立網路時更容易地預測每個圖層的輸出形狀。假設$k_h$是奇數，我們將在高度的兩側填充$p_h/2$行。
如果$k_h$是偶數，則一種可能性是在輸入頂部填充$\lceil p_h/2\rceil$行，在底部填充$\lfloor p_h/2\rfloor$行。同理，我們填充寬度的兩側。

卷積神經網路中卷積核的高度和寬度通常為奇數，例如1、3、5或7。
選擇奇數的好處是，保持空間維度的同時，我們可以在頂部和底部填充相同數量的行，在左側和右側填充相同數量的列。

此外，使用奇數的核大小和填充大小也提供了書寫上的便利。對於任何二維張量`X`，當滿足：
1. 卷積核的大小是奇數；
2. 所有邊的填充行數和列數相同；
3. 輸出與輸入具有相同高度和寬度
則可以得出：輸出`Y[i, j]`是透過以輸入`X[i, j]`為中心，與卷積核進行互相關計算得到的。

比如，在下面的例子中，我們建立一個高度和寬度為3的二維卷積層，並(**在所有側邊填充1個畫素**)。給定高度和寬度為8的輸入，則輸出的高度和寬度也是8。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# 為了方便起見，我們定義了一個計算卷積層的函式。
# 此函式初始化卷積層權重，並對輸入和輸出提高和縮減相應的維數
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # 這裡的（1，1）表示批次大小和通道數都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前兩個維度：批次大小和通道
    return Y.reshape(Y.shape[2:])

# 請注意，這裡每邊都填充了1行或1列，因此總共新增了2行或2列
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# 為了方便起見，我們定義了一個計算卷積層的函式。
# 此函式初始化卷積層權重，並對輸入和輸出提高和縮減相應的維數
def comp_conv2d(conv2d, X):
    # 這裡的（1，1）表示批次大小和通道數都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前兩個維度：批次大小和通道
    return Y.reshape(Y.shape[2:])

# 請注意，這裡每邊都填充了1行或1列，因此總共新增了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# 為了方便起見，我們定義了一個計算卷積層的函式。
# 此函式初始化卷積層權重，並對輸入和輸出提高和縮減相應的維數
def comp_conv2d(conv2d, X):
    # 這裡的（1，1）表示批次大小和通道數都是1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # 省略前兩個維度：批次大小和通道
    return tf.reshape(Y, Y.shape[1:3])

# 請注意，這裡每邊都填充了1行或1列，因此總共新增了2行或2列
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn

# 為了方便起見，我們定義了一個計算卷積層的函式。
# 此函式初始化卷積層權重，並對輸入和輸出提高和縮減相應的維數
def comp_conv2d(conv2d, X):
    # 這裡的（1，1）表示批次大小和通道數都是1
    X = paddle.reshape(X, [1, 1] + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不關心的前兩維：批次和通道

# 請注意，這裡每邊都填充了1行或1列，因此總共新增了2行或2列
conv2d = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
X = paddle.rand((8, 8))
comp_conv2d(conv2d, X).shape
```

當卷積核的高度和寬度不同時，我們可以[**填充不同的高度和寬度**]，使輸出和輸入具有相同的高度和寬度。在如下範例中，我們使用高度為5，寬度為3的卷積核，高度和寬度兩邊的填充分別為2和1。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab paddle
conv2d = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## 步幅

在計算互相關時，卷積視窗從輸入張量的左上角開始，向下、向右滑動。
在前面的例子中，我們預設每次滑動一個元素。
但是，有時候為了高效計算或是縮減取樣次數，卷積視窗可以跳過中間位置，每次滑動多個元素。

我們將每次滑動元素的數量稱為*步幅*（stride）。到目前為止，我們只使用過高度或寬度為$1$的步幅，那麼如何使用較大的步幅呢？
 :numref:`img_conv_stride`是垂直步幅為$3$，水平步幅為$2$的二維互相關運算。
著色部分是輸出元素以及用於輸出計算的輸入和核心張量元素：$0\times0+0\times1+1\times2+2\times3=8$、$0\times0+6\times1+0\times2+0\times3=6$。

可以看到，為了計算輸出中第一列的第二個元素和第一行的第二個元素，卷積視窗分別向下滑動三行和向右滑動兩列。但是，當卷積視窗繼續向右滑動兩列時，沒有輸出，因為輸入元素無法填充視窗（除非我們新增另一列填充）。

![垂直步幅為 $3$，水平步幅為 $2$ 的二維互相關運算。](../img/conv-stride.svg)
:label:`img_conv_stride`

通常，當垂直步幅為$s_h$、水平步幅為$s_w$時，輸出形狀為

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

如果我們設定了$p_h=k_h-1$和$p_w=k_w-1$，則輸出形狀將簡化為$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$。
更進一步，如果輸入的高度和寬度可以被垂直和水平步幅整除，則輸出形狀將為$(n_h/s_h) \times (n_w/s_w)$。

下面，我們[**將高度和寬度的步幅設定為2**]，從而將輸入的高度和寬度減半。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab paddle
conv2d = nn.Conv2D(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

接下來，看(**一個稍微複雜的例子**)。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab paddle
conv2d = nn.Conv2D(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```
為了簡潔起見，當輸入高度和寬度兩側的填充數量分別為$p_h$和$p_w$時，我們稱之為填充$(p_h, p_w)$。當$p_h = p_w = p$時，填充是$p$。同理，當高度和寬度上的步幅分別為$s_h$和$s_w$時，我們稱之為步幅$(s_h, s_w)$。特別地，當$s_h = s_w = s$時，我們稱步幅為$s$。預設情況下，填充為0，步幅為1。在實踐中，我們很少使用不一致的步幅或填充，也就是說，我們通常有$p_h = p_w$和$s_h = s_w$。

## 小結

* 填充可以增加輸出的高度和寬度。這常用來使輸出與輸入具有相同的高和寬。
* 步幅可以減小輸出的高和寬，例如輸出的高和寬僅為輸入的高和寬的$1/n$（$n$是一個大於$1$的整數）。
* 填充和步幅可用於有效地調整資料的維度。

## 練習

1. 對於本節中的最後一個範例，計算其輸出形狀，以檢視它是否與實驗結果一致。
1. 在本節中的實驗中，試一試其他填充和步幅組合。
1. 對於音訊訊號，步幅$2$說明什麼？
1. 步幅大於$1$的計算優勢是什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1852)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1851)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1850)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11784)
:end_tab:
