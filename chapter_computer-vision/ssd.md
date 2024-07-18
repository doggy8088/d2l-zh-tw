# 單發多框檢測（SSD）
:label:`sec_ssd`

在 :numref:`sec_bbox`— :numref:`sec_object-detection-dataset`中，我們分別介紹了邊界框、錨框、多尺度目標檢測和用於目標檢測的資料集。
現在我們已經準備好使用這樣的背景知識來設計一個目標檢測模型：單發多框檢測（SSD） :cite:`Liu.Anguelov.Erhan.ea.2016`。
該模型簡單、快速且被廣泛使用。儘管這只是其中一種目標檢測模型，但本節中的一些設計原則和實現細節也適用於其他模型。

## 模型

 :numref:`fig_ssd`描述了單發多框檢測模型的設計。
此模型主要由基礎網路組成，其後是幾個多尺度特徵區塊。
基本網路用於從輸入影象中提取特徵，因此它可以使用深度卷積神經網路。
單發多框檢測論文中選用了在分類層之前截斷的VGG :cite:`Liu.Anguelov.Erhan.ea.2016`，現在也常用ResNet替代。
我們可以設計基礎網路，使它輸出的高和寬較大。
這樣一來，基於該特徵圖生成的錨框數量較多，可以用來檢測尺寸較小的目標。
接下來的每個多尺度特徵區塊將上一層提供的特徵圖的高和寬縮小（如減半），並使特徵圖中每個單元在輸入影象上的感受野變得更廣闊。

回想一下在 :numref:`sec_multiscale-object-detection`中，透過深度神經網路分層表示影象的多尺度目標檢測的設計。
由於接近 :numref:`fig_ssd`頂部的多尺度特徵圖較小，但具有較大的感受野，它們適合檢測較少但較大的物體。
簡而言之，透過多尺度特徵區塊，單發多框檢測生成不同大小的錨框，並透過預測邊界框的類別和偏移量來檢測大小不同的目標，因此這是一個多尺度目標檢測模型。

![單發多框檢測模型主要由一個基礎網路區塊和若干多尺度特徵區塊串聯而成。](../img/ssd.svg)
:label:`fig_ssd`

在下面，我們將介紹 :numref:`fig_ssd`中不同區塊的實施細節。
首先，我們將討論如何實施類別和邊界框預測。

### [**類別預測層**]

設目標類別別的數量為$q$。這樣一來，錨框有$q+1$個類別，其中0類是背景。
在某個尺度下，設特徵圖的高和寬分別為$h$和$w$。
如果以其中每個單元為中心生成$a$個錨框，那麼我們需要對$hwa$個錨框進行分類。
如果使用全連線層作為輸出，很容易導致模型引數過多。
回憶 :numref:`sec_nin`一節介紹的使用卷積層的通道來輸出類別預測的方法，
單發多框檢測採用同樣的方法來降低模型複雜度。

具體來說，類別預測層使用一個保持輸入高和寬的卷積層。
這樣一來，輸出和輸入在特徵圖寬和高上的空間坐標一一對應。
考慮輸出和輸入同一空間坐標（$x$、$y$）：輸出特徵圖上（$x$、$y$）坐標的通道裡包含了以輸入特徵圖（$x$、$y$）坐標為中心生成的所有錨框的類別預測。
因此輸出通道數為$a(q+1)$，其中索引為$i(q+1) + j$（$0 \leq j \leq q$）的通道代表了索引為$i$的錨框有關類別索引為$j$的預測。

在下面，我們定義了這樣一個類別預測層，透過引數`num_anchors`和`num_classes`分別指定了$a$和$q$。
該圖層使用填充為1的$3\times3$的卷積層。此卷積層的輸入和輸出的寬度和高度保持不變。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.nn import functional as F
import paddle.vision as paddlevision

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2D(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**邊界框預測層**)

邊界框預測層的設計與類別預測層的設計類似。
唯一不同的是，這裡需要為每個錨框預測4個偏移量，而不是$q+1$個類別。

```{.python .input}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab paddle
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2D(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**連結多尺度的預測**]

正如我們所提到的，單發多框檢測使用多尺度特徵圖來生成錨框並預測其類別和偏移量。
在不同的尺度下，特徵圖的形狀或以同一單元為中心的錨框的數量可能會有所不同。
因此，不同尺度下預測輸出的形狀可能會有所不同。

在以下範例中，我們為同一個小批次建立兩個不同比例（`Y1`和`Y2`）的特徵圖，其中`Y2`的高度和寬度是`Y1`的一半。
以類別預測為例，假設`Y1`和`Y2`的每個單元分別生成了$5$個和$3$個錨框。
進一步假設目標類別別的數量為$10$，對於特徵圖`Y1`和`Y2`，類別預測輸出中的通道數分別為$5\times(10+1)=55$和$3\times(10+1)=33$，其中任一輸出的形狀是（批次大小，通道數，高度，寬度）。

```{.python .input}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab paddle
def forward(x, block):
    return block(x)

Y1 = forward(paddle.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(paddle.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

正如我們所看到的，除了批次大小這一維度外，其他三個維度都具有不同的尺寸。
為了將這兩個預測輸出連結起來以提高計算效率，我們將把這些張量轉換為更一致的格式。

通道維包含中心相同的錨框的預測結果。我們首先將通道維移到最後一維。
因為不同尺度下批次大小仍保持不變，我們可以將預測結果轉成二維的（批次大小，高$\times$寬$\times$通道數）的格式，以方便之後在維度$1$上的連結。

```{.python .input}
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

```{.python .input}
#@tab paddle
def flatten_pred(pred):
    return paddle.flatten(pred.transpose([0, 2, 3, 1]), start_axis=1)

def concat_preds(preds):
    return paddle.concat([flatten_pred(p) for p in preds], axis=1)
```

這樣一來，儘管`Y1`和`Y2`在通道數、高度和寬度方面具有不同的大小，我們仍然可以在同一個小批次的兩個不同尺度上連線這兩個預測輸出。

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**高和寬減半區塊**]

為了在多個尺度下檢測目標，我們在下面定義了高和寬減半區塊`down_sample_blk`，該模組將輸入特徵圖的高度和寬度減半。
事實上，該區塊應用了在 :numref:`subsec_vgg-blocks`中的VGG模組設計。
更具體地說，每個高和寬減半區塊由兩個填充為$1$的$3\times3$的卷積層、以及步幅為$2$的$2\times2$最大彙聚層組成。
我們知道，填充為$1$的$3\times3$卷積層不改變特徵圖的形狀。但是，其後的$2\times2$的最大彙聚層將輸入特徵圖的高度和寬度減少了一半。
對於此高和寬減半區塊的輸入和輸出特徵圖，因為$1\times 2+(3-1)+(3-1)=6$，所以輸出中的每個單元在輸入上都有一個$6\times6$的感受野。因此，高和寬減半區塊會擴大每個單元在其輸出特徵圖中的感受野。

```{.python .input}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

```{.python .input}
#@tab paddle
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2D(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2D(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2D(2))
    return nn.Sequential(*blk)
```

在以下範例中，我們建立的高和寬減半區塊會更改輸入通道的數量，並將輸入特徵圖的高度和寬度減半。

```{.python .input}
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

```{.python .input}
#@tab paddle
forward(paddle.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**基本網路區塊**]

基本網路區塊用於從輸入影象中抽取特徵。
為了計算簡潔，我們構造了一個小的基礎網路，該網路串聯3個高和寬減半區塊，並逐步將通道數翻倍。
給定輸入影象的形狀為$256\times256$，此基本網路區塊輸出的特徵圖形狀為$32 \times 32$（$256/2^3=32$）。

```{.python .input}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab paddle
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(paddle.zeros((2, 3, 256, 256)), base_net()).shape
```

### 完整的模型

[**完整的單發多框檢測模型由五個模組組成**]。每個區塊生成的特徵圖既用於生成錨框，又用於預測這些錨框的類別和偏移量。在這五個模組中，第一個是基本網路區塊，第二個到第四個是高和寬減半區塊，最後一個模組使用全域最大池將高度和寬度都降到1。從技術上講，第二到第五個區塊都是 :numref:`fig_ssd`中的多尺度特徵區塊。

```{.python .input}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

```{.python .input}
#@tab paddle
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2D((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

現在我們[**為每個區塊定義前向傳播**]。與影象分類任務不同，此處的輸出包括：CNN特徵圖`Y`；在當前尺度下根據`Y`生成的錨框；預測的這些錨框的類別和偏移量（基於`Y`）。

```{.python .input}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab paddle
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

回想一下，在 :numref:`fig_ssd`中，一個較接近頂部的多尺度特徵區塊是用於檢測較大目標的，因此需要生成更大的錨框。
在上面的前向傳播中，在每個多尺度特徵區塊上，我們透過呼叫的`multibox_prior`函式（見 :numref:`sec_anchor`）的`sizes`引數傳遞兩個比例值的清單。
在下面，0.2和1.05之間的區間被均勻分成五個部分，以確定五個模組的在不同尺度下的較小值：0.2、0.37、0.54、0.71和0.88。
之後，他們較大的值由$\sqrt{0.2 \times 0.37} = 0.272$、$\sqrt{0.37 \times 0.54} = 0.447$等給出。

[~~超引數~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

現在，我們就可以按如下方式[**定義完整的模型**]`TinySSD`了。

```{.python .input}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 即賦值語句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即訪問self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即賦值語句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即訪問self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab paddle
class TinySSD(nn.Layer):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即賦值語句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即訪問self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = paddle.concat(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            (cls_preds.shape[0], -1, self.num_classes + 1))
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

我們[**建立一個模型實例，然後使用它**]對一個$256 \times 256$畫素的小批次影象`X`(**執行前向傳播**)。

如本節前面部分所示，第一個模組輸出特徵圖的形狀為$32 \times 32$。
回想一下，第二到第四個模組為高和寬減半區塊，第五個模組為全域彙聚層。
由於以特徵圖的每個單元為中心有$4$個錨框生成，因此在所有五個尺度下，每個影象總共生成$(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$個錨框。

```{.python .input}
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab paddle
net = TinySSD(num_classes=1)
X = paddle.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 訓練模型

現在，我們將描述如何訓練用於目標檢測的單發多框檢測模型。

### 讀取資料集和初始化

首先，讓我們[**讀取**] :numref:`sec_object-detection-dataset`中描述的(**香蕉檢測資料集**)。

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

香蕉檢測資料集中，目標的類別數為1。
定義好模型後，我們需要(**初始化其引數並定義最佳化演算法**)。

```{.python .input}
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

```{.python .input}
#@tab paddle
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = paddle.optimizer.SGD(learning_rate=0.2, 
                               parameters=net.parameters(), 
                               weight_decay=5e-4)
```

### [**定義損失函式和評價函式**]

目標檢測有兩種型別的損失。
第一種有關錨框類別的損失：我們可以簡單地複用之前影象分類問題裡一直使用的交叉熵損失函式來計算；
第二種有關正類錨框偏移量的損失：預測偏移量是一個迴歸問題。
但是，對於這個迴歸問題，我們在這裡不使用 :numref:`subsec_normal_distribution_and_squared_loss`中描述的平方損失，而是使用$L_1$範數損失，即預測值和真實值之差的絕對值。
掩碼變數`bbox_masks`令負類錨框和填充錨框不參與損失的計算。
最後，我們將錨框類別和偏移量的損失相加，以獲得模型的最終損失函式。

```{.python .input}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

```{.python .input}
#@tab paddle
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape((-1, num_classes)),
                   cls_labels.reshape([-1])).reshape((batch_size, -1)).mean(axis=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(axis=1)
    return cls + bbox
```

我們可以沿用準確率評價分類結果。
由於偏移量使用了$L_1$範數損失，我們使用*平均絕對誤差*來評價邊界框的預測結果。這些預測結果是從生成的錨框及其預測偏移量中獲得的。

```{.python .input}
def cls_eval(cls_preds, cls_labels):
    # 由於類別預測結果放在最後一維，argmax需要指定最後一維。
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # 由於類別預測結果放在最後一維，argmax需要指定最後一維。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab paddle
def cls_eval(cls_preds, cls_labels):
    # 由於類別預測結果放在最後一維，argmax需要指定最後一維。
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((paddle.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**訓練模型**]

在訓練模型時，我們需要在模型的前向傳播過程中生成多尺度錨框（`anchors`），並預測其類別（`cls_preds`）和偏移量（`bbox_preds`）。
然後，我們根據標籤訊息`Y`為生成的錨框標記類別（`cls_labels`）和偏移量（`bbox_labels`）。
最後，我們根據類別和偏移量的預測和標註值計算損失函式。為了程式碼簡潔，這裡沒有評價測試資料集。

```{.python .input}
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # 指標包括：訓練精確度的和，訓練精確度的和中的範例數，
    # 絕對誤差的和，絕對誤差的和中的範例數
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # 生成多尺度的錨框，為每個錨框預測類別和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 為每個錨框標註類別和偏移量
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # 根據類別和偏移量的預測和標註值計算損失函式
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 訓練精確度的和，訓練精確度的和中的範例數
    # 絕對誤差的和，絕對誤差的和中的範例數
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的錨框，為每個錨框預測類別和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 為每個錨框標註類別和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根據類別和偏移量的預測和標註值計算損失函式
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab paddle
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # 訓練精確度的和，訓練精確度的和中的範例數
    # 絕對誤差的和，絕對誤差的和中的範例數
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.clear_grad()
        X, Y = features, target
        # 生成多尺度的錨框，為每個錨框預測類別和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 為每個錨框標註類別和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根據類別和偏移量的預測和標註值計算損失函式
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**預測目標**]

在預測階段，我們希望能把影象裡面所有我們感興趣的目標檢測出來。在下面，我們讀取並調整測試影象的大小，然後將其轉成卷積層需要的四維格式。

```{.python .input}
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

```{.python .input}
#@tab paddle
X = paddle.to_tensor(
            paddlevision.image.image_load(
                '../img/banana.jpg', backend="cv2"
                )[..., ::-1].transpose([2,0,1])
                ).unsqueeze(0).astype(paddle.float32)
img = X.squeeze(0).transpose([1, 2, 0]).astype(paddle.int64)
```

使用下面的`multibox_detection`函式，我們可以根據錨框及其預測偏移量得到預測邊界框。然後，透過非極大值抑制來移除相似的預測邊界框。

```{.python .input}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab paddle
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X)
    cls_probs = F.softmax(cls_preds, axis=2).transpose([0, 2, 1])
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, :][idx]

output = predict(X)
```

最後，我們[**篩選所有置信度不低於0.9的邊界框，做為最終輸出**]。

```{.python .input}
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

```{.python .input}
#@tab paddle
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * paddle.to_tensor((w, h, w, h))]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## 小結

* 單發多框檢測是一種多尺度目標檢測模型。基於基礎網路區塊和各個多尺度特徵區塊，單發多框檢測生成不同數量和不同大小的錨框，並透過預測這些錨框的類別和偏移量檢測不同大小的目標。
* 在訓練單發多框檢測模型時，損失函式是根據錨框的類別和偏移量的預測及標註值計算得出的。

## 練習

1. 能透過改進損失函式來改進單發多框檢測嗎？例如，將預測偏移量用到的$L_1$範數損失替換為平滑$L_1$範數損失。它在零點附近使用平方函式從而更加平滑，這是透過一個超引數$\sigma$來控制平滑區域的：

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

當$\sigma$非常大時，這種損失類似於$L_1$範數損失。當它的值較小時，損失函式較平滑。

```{.python .input}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab paddle
def smooth_l1(data, scalar):
    out = []
    for i in data.numpy():
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return paddle.to_tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = paddle.arange(-2.0, 2.0, 0.1, dtype=paddle.float32)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

此外，在類別預測時，實驗中使用了交叉熵損失：設真實類別$j$的預測機率是$p_j$，交叉熵損失為$-\log p_j$。我們還可以使用焦點損失 :cite:`Lin.Goyal.Girshick.ea.2017`。給定超引數$\gamma > 0$和$\alpha > 0$，此損失的定義為：

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

可以看到，增大$\gamma$可以有效地減少正類預測機率較大時（例如$p_j > 0.5$）的相對損失，因此訓練可以更集中在那些錯誤分類的困難範例上。

```{.python .input}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab paddle
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * paddle.log(x)

x = paddle.arange(0.01, 1, 0.01, dtype=paddle.float32)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. 由於篇幅限制，我們在本節中省略了單發多框檢測模型的一些實現細節。能否從以下幾個方面進一步改進模型：
    1. 當目標比影象小得多時，模型可以將輸入影象調大；
    1. 通常會存在大量的負錨框。為了使類別分佈更加平衡，我們可以將負錨框的高和寬減半；
    1. 在損失函式中，給類別損失和偏移損失設定不同比重的超引數；
    1. 使用其他方法評估目標檢測模型，例如單發多框檢測論文 :cite:`Liu.Anguelov.Erhan.ea.2016`中的方法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3205)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3204)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11807)
:end_tab:
