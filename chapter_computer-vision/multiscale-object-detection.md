# 多尺度目標檢測
:label:`sec_multiscale-object-detection`

在 :numref:`sec_anchor`中，我們以輸入影象的每個畫素為中心，生成了多個錨框。
基本而言，這些錨框代表了影象不同區域的樣本。
然而，如果為每個畫素都生成的錨框，我們最終可能會得到太多需要計算的錨框。
想象一個$561 \times 728$的輸入影象，如果以每個畫素為中心生成五個形狀不同的錨框，就需要在影象上標記和預測超過200萬個錨框（$561 \times 728 \times 5$）。

## 多尺度錨框
:label:`subsec_multiscale-anchor-boxes`

減少影象上的錨框數量並不困難。
比如，我們可以在輸入影象中均勻取樣一小部分畫素，並以它們為中心生成錨框。
此外，在不同尺度下，我們可以生成不同數量和不同大小的錨框。
直觀地說，比起較大的目標，較小的目標在影象上出現的可能性更多樣。
例如，$1 \times 1$、$1 \times 2$和$2 \times 2$的目標可以分別以4、2和1種可能的方式出現在$2 \times 2$影象上。
因此，當使用較小的錨框檢測較小的物體時，我們可以取樣更多的區域，而對於較大的物體，我們可以取樣較少的區域。

為了示範如何在多個尺度下生成錨框，讓我們先讀取一張影象。
它的高度和寬度分別為561和728畫素。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

回想一下，在 :numref:`sec_conv_layer`中，我們將卷積圖層的二維陣列輸出稱為特徵圖。
透過定義特徵圖的形狀，我們可以確定任何影象上均勻取樣錨框的中心。

`display_anchors`函式定義如下。
我們[**在特徵圖（`fmap`）上生成錨框（`anchors`），每個單位（畫素）作為錨框的中心**]。
由於錨框中的$(x, y)$軸坐標值（`anchors`）已經被除以特徵圖（`fmap`）的寬度和高度，因此這些值介於0和1之間，表示特徵圖中錨框的相對位置。

由於錨框（`anchors`）的中心分佈於特徵圖（`fmap`）上的所有單位，因此這些中心必須根據其相對空間位置在任何輸入影象上*均勻*分佈。
更具體地說，給定特徵圖的寬度和高度`fmap_w`和`fmap_h`，以下函式將*均勻地*對任何輸入影象中`fmap_h`行和`fmap_w`列中的畫素進行取樣。
以這些均勻取樣的畫素為中心，將會生成大小為`s`（假設清單`s`的長度為1）且寬高比（`ratios`）不同的錨框。

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前兩個維度上的值不影響輸出
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前兩個維度上的值不影響輸出
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab paddle
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前兩個維度上的值不影響輸出
    fmap = paddle.zeros(shape=[1, 10, fmap_h, fmap_w])
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = paddle.to_tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

首先，讓我們考慮[**探測小目標**]。
為了在顯示時更容易分辨，在這裡具有不同中心的錨框不會重疊：
錨框的尺度設定為0.15，特徵圖的高度和寬度設定為4。
我們可以看到，影象上4行和4列的錨框的中心是均勻分佈的。

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

然後，我們[**將特徵圖的高度和寬度減小一半，然後使用較大的錨框來檢測較大的目標**]。
當尺度設定為0.4時，一些錨框將彼此重疊。

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

最後，我們進一步[**將特徵圖的高度和寬度減小一半，然後將錨框的尺度增加到0.8**]。
此時，錨框的中心即是影象的中心。

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 多尺度檢測

既然我們已經生成了多尺度的錨框，我們就將使用它們來檢測不同尺度下各種大小的目標。
下面，我們介紹一種基於CNN的多尺度目標檢測方法，將在 :numref:`sec_ssd`中實現。

在某種規模上，假設我們有$c$張形狀為$h \times w$的特徵圖。
使用 :numref:`subsec_multiscale-anchor-boxes`中的方法，我們生成了$hw$組錨框，其中每組都有$a$個中心相同的錨框。
例如，在 :numref:`subsec_multiscale-anchor-boxes`實驗的第一個尺度上，給定10個（通道數量）$4 \times 4$的特徵圖，我們生成了16組錨框，每組包含3個中心相同的錨框。
接下來，每個錨框都根據真實值邊界框來標記了類別和偏移量。
在當前尺度下，目標檢測模型需要預測輸入影象上$hw$組錨框類別和偏移量，其中不同組錨框具有不同的中心。


假設此處的$c$張特徵圖是CNN基於輸入影象的正向傳播演算法獲得的中間輸出。
既然每張特徵圖上都有$hw$個不同的空間位置，那麼相同空間位置可以看作含有$c$個單元。
根據 :numref:`sec_conv_layer`中對感受野的定義，特徵圖在相同空間位置的$c$個單元在輸入影象上的感受野相同：
它們表徵了同一感受野內的輸入影象訊息。
因此，我們可以將特徵圖在同一空間位置的$c$個單元變換為使用此空間位置生成的$a$個錨框類別和偏移量。
本質上，我們用輸入影象在某個感受野區域內的訊息，來預測輸入影象上與該區域位置相近的錨框類別和偏移量。

當不同層的特徵圖在輸入影象上分別擁有不同大小的感受野時，它們可以用於檢測不同大小的目標。
例如，我們可以設計一個神經網路，其中靠近輸出層的特徵圖單元具有更寬的感受野，這樣它們就可以從輸入影象中檢測到較大的目標。

簡言之，我們可以利用深層神經網路在多個層次上對影象進行分層表示，從而實現多尺度目標檢測。
在 :numref:`sec_ssd`，我們將透過一個具體的例子來說明它是如何工作的。

## 小結

* 在多個尺度下，我們可以生成不同尺寸的錨框來檢測不同尺寸的目標。
* 透過定義特徵圖的形狀，我們可以決定任何影象上均勻取樣的錨框的中心。
* 我們使用輸入影象在某個感受野區域內的訊息，來預測輸入影象上與該區域位置相近的錨框類別和偏移量。
* 我們可以透過深入學習，在多個層次上的影象分層表示進行多尺度目標檢測。

## 練習

1. 根據我們在 :numref:`sec_alexnet`中的討論，深度神經網路學習影象特徵級別抽象層次，隨網路深度的增加而升級。在多尺度目標檢測中，不同尺度的特徵對映是否對應於不同的抽象層次？為什麼？
1. 在 :numref:`subsec_multiscale-anchor-boxes`中的實驗裡的第一個尺度（`fmap_w=4, fmap_h=4`）下，生成可能重疊的均勻分佈的錨框。
1. 給定形狀為$1 \times c \times h \times w$的特徵圖變數，其中$c$、$h$和$w$分別是特徵圖的通道數、高度和寬度。怎樣才能將這個變數轉換為錨框類別和偏移量？輸出的形狀是什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2947)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2948)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11805)
:end_tab:
