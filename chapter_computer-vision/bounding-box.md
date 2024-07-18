# 目標檢測和邊界框
:label:`sec_bbox`

前面的章節（例如 :numref:`sec_alexnet`— :numref:`sec_googlenet`）介紹了各種影象分類模型。
在影象分類任務中，我們假設影象中只有一個主要物體物件，我們只關注如何識別其類別。
然而，很多時候影象裡有多個我們感興趣的目標，我們不僅想知道它們的類別，還想得到它們在影象中的具體位置。
在電腦視覺裡，我們將這類任務稱為*目標檢測*（object detection）或*目標識別*（object recognition）。

目標檢測在多個領域中被廣泛使用。
例如，在無人駕駛裡，我們需要透過識別拍攝到的影片影象裡的車輛、行人、道路和障礙物的位置來規劃行進線路。
機器人也常透過該任務來檢測感興趣的目標。安防領域則需要檢測異常目標，如歹徒或者炸彈。

接下來的幾節將介紹幾種用於目標檢測的深度學習方法。
我們將首先介紹目標的*位置*。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
```

下面載入本節將使用的範例影象。可以看到影象左邊是一隻狗，右邊是一隻貓。
它們是這張影象裡的兩個主要目標。

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow, paddle
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## 邊界框

在目標檢測中，我們通常使用*邊界框*（bounding box）來描述物件的空間位置。
邊界框是矩形的，由矩形左上角的以及右下角的$x$和$y$坐標決定。
另一種常用的邊界框表示方法是邊界框中心的$(x, y)$軸坐標以及框的寬度和高度。

在這裡，我們[**定義在這兩種表示法之間進行轉換的函式**]：`box_corner_to_center`從兩角表示法轉換為中心寬度表示法，而`box_center_to_corner`反之亦然。
輸入引數`boxes`可以是長度為4的張量，也可以是形狀為（$n$，4）的二維張量，其中$n$是邊界框的數量。

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """從（左上，右下）轉換到（中間，寬度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """從（中間，寬度，高度）轉換到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

我們將根據坐標訊息[**定義影象中狗和貓的邊界框**]。
影象中坐標的原點是影象的左上角，向右的方向為$x$軸的正方向，向下的方向為$y$軸的正方向。

```{.python .input}
#@tab all
# bbox是邊界框的英文縮寫
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

我們可以透過轉換兩次來驗證邊界框轉換函式的正確性。

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

我們可以[**將邊界框在圖中畫出**]，以檢查其是否準確。
畫之前，我們定義一個輔助函式`bbox_to_rect`。
它將邊界框表示成`matplotlib`的邊界框格式。

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    # 將邊界框(左上x,左上y,右下x,右下y)格式轉換成matplotlib格式：
    # ((左上x,左上y),寬,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

在影象上新增邊界框之後，我們可以看到兩個物體的主要輪廓基本上在兩個框內。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 小結

* 目標檢測不僅可以識別影象中所有感興趣的物體，還能識別它們的位置，該位置通常由矩形邊界框表示。
* 我們可以在兩種常用的邊界框表示（中間，寬度，高度）和（左上，右下）坐標之間進行轉換。

## 練習

1. 找到另一張影象，然後嘗試標記包含該物件的邊界框。比較標註邊界框和標註類別哪個需要更長的時間？
1. 為什麼`box_corner_to_center`和`box_center_to_corner`的輸入引數的最內層維度總是4？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2943)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2944)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11803)
:end_tab:
