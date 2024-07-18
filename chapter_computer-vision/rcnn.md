# 區域卷積神經網路（R-CNN）系列
:label:`sec_rcnn`

除了 :numref:`sec_ssd`中描述的單發多框檢測之外，
區域卷積神經網路（region-based CNN或regions with CNN features，R-CNN） :cite:`Girshick.Donahue.Darrell.ea.2014`也是將深度模型應用於目標檢測的開創性工作之一。
本節將介紹R-CNN及其一系列改進方法：快速的R-CNN（Fast R-CNN） :cite:`Girshick.2015`、更快的R-CNN（Faster R-CNN） :cite:`Ren.He.Girshick.ea.2015`和掩碼R-CNN（Mask R-CNN） :cite:`He.Gkioxari.Dollar.ea.2017`。
限於篇幅，我們只著重介紹這些模型的設計思路。

## R-CNN

*R-CNN*首先從輸入影象中選取若干（例如2000個）*提議區域*（如錨框也是一種選取方法），並標註它們的類別和邊界框（如偏移量）。 :cite:`Girshick.Donahue.Darrell.ea.2014`然後，用卷積神經網路對每個提議區域進行前向傳播以抽取其特徵。
接下來，我們用每個提議區域的特徵來預測類別和邊界框。

![R-CNN模型](../img/r-cnn.svg)
:label:`fig_r-cnn`

 :numref:`fig_r-cnn`展示了R-CNN模型。具體來說，R-CNN包括以下四個步驟：

1. 對輸入影象使用*選擇性搜尋*來選取多個高質量的提議區域 :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`。這些提議區域通常是在多個尺度下選取的，並具有不同的形狀和大小。每個提議區域都將被標註類別和真實邊界框；
1. 選擇一個預訓練的卷積神經網路，並將其在輸出層之前截斷。將每個提議區域變形為網路需要的輸入尺寸，並透過前向傳播輸出抽取的提議區域特徵；
1. 將每個提議區域的特徵連同其標註的類別作為一個樣本。訓練多個支援向量機對目標分類，其中每個支援向量機用來判斷樣本是否屬於某一個類別；
1. 將每個提議區域的特徵連同其標註的邊界框作為一個樣本，訓練線性迴歸模型來預測真實邊界框。

儘管R-CNN模型透過預訓練的卷積神經網路有效地抽取了影象特徵，但它的速度很慢。
想象一下，我們可能從一張影象中選出上千個提議區域，這需要上千次的卷積神經網路的前向傳播來執行目標檢測。
這種龐大的計算量使得R-CNN在現實世界中難以被廣泛應用。

## Fast R-CNN

R-CNN的主要效能瓶頸在於，對每個提議區域，卷積神經網路的前向傳播是獨立的，而沒有共享計算。
由於這些區域通常有重疊，獨立的特徵抽取會導致重複的計算。
*Fast R-CNN* :cite:`Girshick.2015`對R-CNN的主要改進之一，是僅在整張圖象上執行卷積神經網路的前向傳播。

![Fast R-CNN模型](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`

 :numref:`fig_fast_r-cnn`中描述了Fast R-CNN模型。它的主要計算如下：

1. 與R-CNN相比，Fast R-CNN用來提取特徵的卷積神經網路的輸入是整個影象，而不是各個提議區域。此外，這個網路通常會參與訓練。設輸入為一張影象，將卷積神經網路的輸出的形狀記為$1 \times c \times h_1  \times w_1$；
1. 假設選擇性搜尋生成了$n$個提議區域。這些形狀各異的提議區域在卷積神經網路的輸出上分別標出了形狀各異的興趣區域。然後，這些感興趣的區域需要進一步抽取出形狀相同的特徵（比如指定高度$h_2$和寬度$w_2$），以便於連結後輸出。為了實現這一目標，Fast R-CNN引入了*興趣區域彙聚層*（RoI pooling）：將卷積神經網路的輸出和提議區域作為輸入，輸出連結後的各個提議區域抽取的特徵，形狀為$n \times c \times h_2 \times w_2$；
1. 透過全連線層將輸出形狀變換為$n \times d$，其中超引數$d$取決於模型設計；
1. 預測$n$個提議區域中每個區域的類別和邊界框。更具體地說，在預測類別和邊界框時，將全連線層的輸出分別轉換為形狀為$n \times q$（$q$是類別的數量）的輸出和形狀為$n \times 4$的輸出。其中預測類別時使用softmax迴歸。

在Fast R-CNN中提出的興趣區域彙聚層與 :numref:`sec_pooling`中介紹的彙聚層有所不同。在彙聚層中，我們透過設定彙聚視窗、填充和步幅的大小來間接控制輸出形狀。而興趣區域彙聚層對每個區域的輸出形狀是可以直接指定的。

例如，指定每個區域輸出的高和寬分別為$h_2$和$w_2$。
對於任何形狀為$h \times w$的興趣區域視窗，該視窗將被劃分為$h_2 \times w_2$子視窗網格，其中每個子視窗的大小約為$(h/h_2) \times (w/w_2)$。
在實踐中，任何子視窗的高度和寬度都應向上取整，其中的最大元素作為該子視窗的輸出。
因此，興趣區域彙聚層可從形狀各異的興趣區域中均抽取出形狀相同的特徵。

作為說明性範例， :numref:`fig_roi`中提到，在$4 \times 4$的輸入中，我們選取了左上角$3\times 3$的興趣區域。
對於該興趣區域，我們透過$2\times 2$的興趣區域彙聚層得到一個$2\times 2$的輸出。
請注意，四個劃分後的子視窗中分別含有元素0、1、4、5（5最大）；2、6（6最大）；8、9（9最大）；以及10。

![一個 $2\times 2$ 的興趣區域彙聚層](../img/roi.svg)
:label:`fig_roi`

下面，我們示範了興趣區域彙聚層的計算方法。
假設卷積神經網路抽取的特徵`X`的高度和寬度都是4，且只有單通道。

```{.python .input}
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.vision as paddlevision

X = paddle.reshape(paddle.arange(16, dtype='float32'), (1,1,4,4))
X
```

讓我們進一步假設輸入影象的高度和寬度都是40畫素，且選擇性搜尋在此影象上生成了兩個提議區域。
每個區域由5個元素表示：區域目標類別別、左上角和右下角的$(x, y)$坐標。

```{.python .input}
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab paddle
rois = paddle.to_tensor([[0, 0, 20, 20], [0, 10, 30, 30]]).astype('float32')
```

由於`X`的高和寬是輸入影象高和寬的$1/10$，因此，兩個提議區域的坐標先按`spatial_scale`乘以0.1。
然後，在`X`上分別標出這兩個興趣區域`X[:, :, 0:3, 0:3]`和`X[:, :, 1:4, 0:4]`。
最後，在$2\times 2$的興趣區域彙聚層中，每個興趣區域被劃分為子視窗網格，並進一步抽取相同形狀$2\times 2$的特徵。

```{.python .input}
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab paddle
boxes_num = paddle.to_tensor([len(rois)]).astype('int32')
paddlevision.ops.roi_pool(X, rois, boxes_num, output_size=(2, 2), spatial_scale=0.1)
```

## Faster R-CNN

為了較精確地檢測目標結果，Fast R-CNN模型通常需要在選擇性搜尋中生成大量的提議區域。
*Faster R-CNN* :cite:`Ren.He.Girshick.ea.2015`提出將選擇性搜尋替換為*區域提議網路*（region proposal network），從而減少提議區域的生成數量，並保證目標檢測的精度。

![Faster R-CNN 模型](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`

 :numref:`fig_faster_r-cnn`描述了Faster R-CNN模型。
與Fast R-CNN相比，Faster R-CNN只將生成提議區域的方法從選擇性搜尋改為了區域提議網路，模型的其餘部分保持不變。具體來說，區域提議網路的計算步驟如下：

1. 使用填充為1的$3\times 3$的卷積層變換卷積神經網路的輸出，並將輸出通道數記為$c$。這樣，卷積神經網路為影象抽取的特徵圖中的每個單元均得到一個長度為$c$的新特徵。
1. 以特徵圖的每個畫素為中心，生成多個不同大小和寬高比的錨框並標註它們。
1. 使用錨框中心單元長度為$c$的特徵，分別預測該錨框的二元類別（含目標還是背景）和邊界框。
1. 使用非極大值抑制，從預測類別為目標的預測邊界框中移除相似的結果。最終輸出的預測邊界框即是興趣區域彙聚層所需的提議區域。

值得一提的是，區域提議網路作為Faster R-CNN模型的一部分，是和整個模型一起訓練得到的。
換句話說，Faster R-CNN的目標函式不僅包括目標檢測中的類別和邊界框預測，還包括區域提議網路中錨框的二元類別和邊界框預測。
作為端到端訓練的結果，區域提議網路能夠學習到如何生成高質量的提議區域，從而在減少了從資料中學習的提議區域的數量的情況下，仍保持目標檢測的精度。

## Mask R-CNN

如果在訓練集中還標註了每個目標在影象上的畫素級位置，那麼*Mask R-CNN* :cite:`He.Gkioxari.Dollar.ea.2017`能夠有效地利用這些詳盡的標註訊息進一步提升目標檢測的精度。

![Mask R-CNN 模型](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

如 :numref:`fig_mask_r-cnn`所示，Mask R-CNN是基於Faster R-CNN修改而來的。
具體來說，Mask R-CNN將興趣區域彙聚層替換為了
*興趣區域對齊*層，使用*雙線性內嵌*（bilinear interpolation）來保留特徵圖上的空間訊息，從而更適於畫素級預測。
興趣區域對齊層的輸出包含了所有與興趣區域的形狀相同的特徵圖。
它們不僅被用於預測每個興趣區域的類別和邊界框，還透過額外的全卷積網路預測目標的畫素級位置。
本章的後續章節將更詳細地介紹如何使用全卷積網路預測影象中畫素級的語義。

## 小結

* R-CNN對影象選取若干提議區域，使用卷積神經網路對每個提議區域執行前向傳播以抽取其特徵，然後再用這些特徵來預測提議區域的類別和邊界框。
* Fast R-CNN對R-CNN的一個主要改進：只對整個影象做卷積神經網路的前向傳播。它還引入了興趣區域彙聚層，從而為具有不同形狀的興趣區域抽取相同形狀的特徵。
* Faster R-CNN將Fast R-CNN中使用的選擇性搜尋替換為參與訓練的區域提議網路，這樣後者可以在減少提議區域數量的情況下仍保證目標檢測的精度。
* Mask R-CNN在Faster R-CNN的基礎上引入了一個全卷積網路，從而藉助目標的畫素級位置進一步提升目標檢測的精度。

## 練習

1. 我們能否將目標檢測視為迴歸問題（例如預測邊界框和類別的機率）？可以參考YOLO模型 :cite:`Redmon.Divvala.Girshick.ea.2016`的設計。
1. 將單發多框檢測與本節介紹的方法進行比較。他們的主要區別是什麼？可以參考 :cite:`Zhao.Zheng.Xu.ea.2019`中的圖2。

:begin_tab:`mxnet`
[討論區](https://discuss.d2l.ai/t/3206)
:end_tab:

:begin_tab:`pytorch`
[討論區](https://discuss.d2l.ai/t/3207)
:end_tab:

:begin_tab:`paddle`
[討論區](https://discuss.d2l.ai/t/11808)
:end_tab:
