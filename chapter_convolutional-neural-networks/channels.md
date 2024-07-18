# 多輸入多輸出通道
:label:`sec_channels`

雖然我們在 :numref:`subsec_why-conv-channels`中描述了構成每個影象的多個通道和多層卷積層。例如彩色影象具有標準的RGB通道來代表紅、綠和藍。
但是到目前為止，我們僅展示了單個輸入和單個輸出通道的簡化例子。
這使得我們可以將輸入、卷積核和輸出看作二維張量。

當我們新增通道時，我們的輸入和隱藏的表示都變成了三維張量。例如，每個RGB輸入影象具有$3\times h\times w$的形狀。我們將這個大小為$3$的軸稱為*通道*（channel）維度。本節將更深入地研究具有多輸入和多輸出通道的卷積核。

## 多輸入通道

當輸入包含多個通道時，需要構造一個與輸入資料具有相同輸入通道數的卷積核，以便與輸入資料進行互相關運算。假設輸入的通道數為$c_i$，那麼卷積核的輸入通道數也需要為$c_i$。如果卷積核的視窗形狀是$k_h\times k_w$，那麼當$c_i=1$時，我們可以把卷積核看作形狀為$k_h\times k_w$的二維張量。

然而，當$c_i>1$時，我們卷積核的每個輸入通道將包含形狀為$k_h\times k_w$的張量。將這些張量$c_i$連結在一起可以得到形狀為$c_i\times k_h\times k_w$的卷積核。由於輸入和卷積核都有$c_i$個通道，我們可以對每個通道輸入的二維張量和卷積核的二維張量進行互相關運算，再對通道求和（將$c_i$的結果相加）得到二維張量。這是多通道輸入和多輸入通道卷積核之間進行二維互相關運算的結果。

在 :numref:`fig_conv_multi_in`中，我們示範了一個具有兩個輸入通道的二維互相關運算的範例。陰影部分是第一個輸出元素以及用於計算這個輸出的輸入和核張量元素：$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$。

![兩個輸入通道的互相關計算。](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

為了加深理解，我們(**實現一下多輸入通道互相關運算**)。
簡而言之，我們所做的就是對每個通道執行互相關操作，然後將結果相加。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
```

```{.python .input}
#@tab mxnet, pytorch, paddle
def corr2d_multi_in(X, K):
    # 先遍歷『X』和『K』的第0個維度（通道維度），再把它們加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # 先遍歷『X』和『K』的第0個維度（通道維度），再把它們加在一起
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

我們可以構造與 :numref:`fig_conv_multi_in`中的值相對應的輸入張量`X`和核張量`K`，以(**驗證互相關運算的輸出**)。

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## 多輸出通道

到目前為止，不論有多少輸入通道，我們還只有一個輸出通道。然而，正如我們在 :numref:`subsec_why-conv-channels`中所討論的，每一層有多個輸出通道是至關重要的。在最流行的神經網路架構中，隨著神經網路層數的加深，我們常會增加輸出通道的維數，透過減少空間解析度以獲得更大的通道深度。直觀地說，我們可以將每個通道看作對不同特徵的回應。而現實可能更為複雜一些，因為每個通道不是獨立學習的，而是為了共同使用而最佳化的。因此，多輸出通道並不僅是學習多個單通道的檢測器。

用$c_i$和$c_o$分別表示輸入和輸出通道的數目，並讓$k_h$和$k_w$為卷積核的高度和寬度。為了獲得多個通道的輸出，我們可以為每個輸出通道建立一個形狀為$c_i\times k_h\times k_w$的卷積核張量，這樣卷積核的形狀是$c_o\times c_i\times k_h\times k_w$。在互相關運算中，每個輸出通道先獲取所有輸入通道，再以對應該輸出通道的卷積核計算出結果。

如下所示，我們實現一個[**計算多個通道的輸出的互相關函式**]。

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # 迭代『K』的第0個維度，每次都對輸入『X』執行互相關運算。
    # 最後將所有結果都疊加在一起
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

透過將核張量`K`與`K+1`（`K`中每個元素加$1$）和`K+2`連線起來，構造了一個具有$3$個輸出通道的卷積核。

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

下面，我們對輸入張量`X`與卷積核張量`K`執行互相關運算。現在的輸出包含$3$個通道，第一個通道的結果與先前輸入張量`X`和多輸入單輸出通道的結果一致。

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ 卷積層

[~~1x1卷積~~]

$1 \times 1$卷積，即$k_h = k_w = 1$，看起來似乎沒有多大意義。
畢竟，卷積的本質是有效提取相鄰畫素間的相關特徵，而$1 \times 1$卷積顯然沒有此作用。
儘管如此，$1 \times 1$仍然十分流行，經常包含在複雜深層網路的設計中。下面，讓我們詳細地解讀一下它的實際作用。

因為使用了最小視窗，$1\times 1$卷積失去了卷積層的特有能力——在高度和寬度維度上，識別相鄰元素間相互作用的能力。
其實$1\times 1$卷積的唯一計算發生在通道上。

 :numref:`fig_conv_1x1`展示了使用$1\times 1$卷積核與$3$個輸入通道和$2$個輸出通道的互相關計算。
這裡輸入和輸出具有相同的高度和寬度，輸出中的每個元素都是從輸入影象中同一位置的元素的線性組合。
我們可以將$1\times 1$卷積層看作在每個畫素位置應用的全連線層，以$c_i$個輸入值轉換為$c_o$個輸出值。
因為這仍然是一個卷積層，所以跨畫素的權重是一致的。
同時，$1\times 1$卷積層需要的權重維度為$c_o\times c_i$，再額外加上一個偏置。

![互相關計算使用了具有3個輸入通道和2個輸出通道的 $1\times 1$ 卷積核。其中，輸入和輸出具有相同的高度和寬度。](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

下面，我們使用全連線層實現$1 \times 1$卷積。
請注意，我們需要對輸入和輸出的資料形狀進行調整。

```{.python .input}
#@tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # 全連線層中的矩陣乘法
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

當執行$1\times 1$卷積運算時，上述函式相當於先前實現的互相關函式`corr2d_multi_in_out`。讓我們用一些樣本數據來驗證這一點。

```{.python .input}
#@tab mxnet, pytorch, paddle
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
#@tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## 小結

* 多輸入多輸出通道可以用來擴充套件卷積層的模型。
* 當以每畫素為基礎應用時，$1\times 1$卷積層相當於全連線層。
* $1\times 1$卷積層通常用於調整網路層的通道數量和控制模型複雜性。

## 練習

1. 假設我們有兩個卷積核，大小分別為$k_1$和$k_2$（中間沒有非線性啟用函式）。
    1. 證明運算可以用單次卷積來表示。
    1. 這個等效的單個卷積核的維數是多少呢？
    1. 反之亦然嗎？
1. 假設輸入為$c_i\times h\times w$，卷積核大小為$c_o\times c_i\times k_h\times k_w$，填充為$(p_h, p_w)$，步幅為$(s_h, s_w)$。
    1. 前向傳播的計算成本（乘法和加法）是多少？
    1. 記憶體佔用是多少？
    1. 反向傳播的記憶體佔用是多少？
    1. 反向傳播的計算成本是多少？
1. 如果我們將輸入通道$c_i$和輸出通道$c_o$的數量加倍，計算數量會增加多少？如果我們把填充數量翻一番會怎麼樣？
1. 如果卷積核的高度和寬度是$k_h=k_w=1$，前向傳播的計算複雜度是多少？
1. 本節最後一個範例中的變數`Y1`和`Y2`是否完全相同？為什麼？
1. 當卷積視窗不是$1\times 1$時，如何使用矩陣乘法實現卷積？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1855)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1854)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1853)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11785)
:end_tab:
