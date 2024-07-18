# 錨框
:label:`sec_anchor`

目標檢測演算法通常會在輸入影象中取樣大量的區域，然後判斷這些區域中是否包含我們感興趣的目標，並調整區域邊界從而更準確地預測目標的*真實邊界框*（ground-truth bounding box）。
不同的模型使用的區域取樣方法可能不同。
這裡我們介紹其中的一種方法：以每個畫素為中心，生成多個縮放比和寬高比（aspect ratio）不同的邊界框。
這些邊界框被稱為*錨框*（anchor box）我們將在 :numref:`sec_ssd`中設計一個基於錨框的目標檢測模型。

首先，讓我們修改輸出精度，以獲得更簡潔的輸出。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # 精簡輸出精度
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # 精簡輸出精度
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import numpy as np

paddle.set_printoptions(2)  # 精簡輸出精度
```

## 生成多個錨框

假設輸入影象的高度為$h$，寬度為$w$。
我們以影象的每個畫素為中心生成不同形狀的錨框：*縮放比*為$s\in (0, 1]$，*寬高比*為$r > 0$。
那麼[**錨框的寬度和高度分別是$hs\sqrt{r}$和$hs/\sqrt{r}$。**]
請注意，當中心位置給定時，已知寬和高的錨框是確定的。

要生成多個不同形狀的錨框，讓我們設定許多縮放比（scale）取值$s_1,\ldots, s_n$和許多寬高比（aspect ratio）取值$r_1,\ldots, r_m$。
當使用這些比例和長寬比的所有組合以每個畫素為中心時，輸入影象將總共有$whnm$個錨框。
儘管這些錨框可能會覆蓋所有真實邊界框，但計算複雜性很容易過高。
在實踐中，(**我們只考慮**)包含$s_1$或$r_1$的(**組合：**)

(**
$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$
**)

也就是說，以同一畫素為中心的錨框的數量是$n+m-1$。
對於整個輸入影象，將共生成$wh(n+m-1)$個錨框。

上述生成錨框的方法在下面的`multibox_prior`函式中實現。
我們指定輸入影象、尺寸清單和寬高比清單，然後此函式將回傳所有的錨框。

```{.python .input}
#@save
def multibox_prior(data, sizes, ratios):
    """生成以每個畫素為中心具有不同形狀的錨框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)

    # 為了將錨點移動到畫素的中心，需要設定偏移量。
    # 因為一個畫素的高為1且寬為1，我們選擇偏移我們的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y軸上縮放步長
    steps_w = 1.0 / in_width  # 在x軸上縮放步長

    # 生成錨框的所有中心點
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # 生成『boxes_per_pixel』個高和寬，
    # 之後用於建立錨框的四角坐標(xmin,xmax,ymin,ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # 處理矩形輸入
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # 除以2來獲得半高和半寬
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # 每個中心點都將有『boxes_per_pixel』個錨框，
    # 所以生成含所有錨框中心的網格，重複了『boxes_per_pixel』次
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """生成以每個畫素為中心具有不同形狀的錨框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)

    # 為了將錨點移動到畫素的中心，需要設定偏移量。
    # 因為一個畫素的高為1且寬為1，我們選擇偏移我們的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y軸上縮放步長
    steps_w = 1.0 / in_width  # 在x軸上縮放步長

    # 生成錨框的所有中心點
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成『boxes_per_pixel』個高和寬，
    # 之後用於建立錨框的四角坐標(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 處理矩形輸入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2來獲得半高和半寬
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每個中心點都將有『boxes_per_pixel』個錨框，
    # 所以生成含所有錨框中心的網格，重複了『boxes_per_pixel』次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

```{.python .input}
#@tab paddle
#@save
def multibox_prior(data, sizes, ratios):
    """生成以每個畫素為中心具有不同形狀的錨框"""
    in_height, in_width = data.shape[-2:]
    place, num_sizes, num_ratios = data.place, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = paddle.to_tensor(sizes, place=place)
    ratio_tensor = paddle.to_tensor(ratios, place=place)

    # 為了將錨點移動到畫素的中心，需要設定偏移量。
    # 因為一個畫素的的高為1且寬為1，我們選擇偏移我們的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y軸上縮放步長
    steps_w = 1.0 / in_width  # 在x軸上縮放步長

    # 生成錨框的所有中心點
    center_h = (paddle.arange(in_height) + offset_h) * steps_h
    center_w = (paddle.arange(in_width) + offset_w) * steps_w
    shift_y, shift_x = paddle.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape([-1]), shift_x.reshape([-1])

    # 生成『boxes_per_pixel』個高和寬，
    # 之後用於建立錨框的四角坐標(xmin,xmax,ymin,ymax)
    w = paddle.concat((size_tensor * paddle.sqrt(ratio_tensor[0]),
                       sizes[0] * paddle.sqrt(ratio_tensor[1:])))\
                       * in_height / in_width  # 處理矩形輸入
    h = paddle.concat((size_tensor / paddle.sqrt(ratio_tensor[0]),
                   sizes[0] / paddle.sqrt(ratio_tensor[1:])))
    # 除以2來獲得半高和半寬
    anchor_manipulations = paddle.tile(paddle.stack((-w, -h, w, h)).T,
                                        (in_height * in_width, 1)) / 2

    # 每個中心點都將有『boxes_per_pixel』個錨框，
    # 所以生成含所有錨框中心的網格，重複了『boxes_per_pixel』次
    out_grid = paddle.stack([shift_x, shift_y, shift_x, shift_y], axis=1)
    out_grid = paddle.tile(out_grid, repeat_times=[boxes_per_pixel]).reshape((-1, out_grid.shape[1]))
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

可以看到[**回傳的錨框變數`Y`的形狀**]是（批次大小，錨框的數量，4）。

```{.python .input}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab paddle
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = paddle.rand(shape=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

將錨框變數`Y`的形狀更改為(影象高度,影象寬度,以同一畫素為中心的錨框的數量,4)後，我們可以獲得以指定畫素的位置為中心的所有錨框。
在接下來的內容中，我們[**訪問以（250,250）為中心的第一個錨框**]。
它有四個元素：錨框左上角的$(x, y)$軸坐標和右下角的$(x, y)$軸坐標。
輸出中兩個軸的坐標各分別除以了影象的寬度和高度。

```{.python .input}
#@tab mxnet, pytorch
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

```{.python .input}
#@tab paddle
boxes = Y.reshape([h, w, 5, 4])
boxes[250, 250, 0, :]
```

為了[**顯示以影象中以某個畫素為中心的所有錨框**]，定義下面的`show_bboxes`函式來在影象上繪製多個邊界框。

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """顯示所有邊界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
        
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

正如從上面程式碼中所看到的，變數`boxes`中$x$軸和$y$軸的坐標值已分別除以影象的寬度和高度。
繪製錨框時，我們需要恢復它們原始的坐標值。
因此，在下面定義了變數`bbox_scale`。
現在可以繪製出影象中所有以(250,250)為中心的錨框了。
如下所示，縮放比為0.75且寬高比為1的藍色錨框很好地圍繞著影象中的狗。

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**交併比（IoU）**]

我們剛剛提到某個錨框『較好地』覆蓋了影象中的狗。
如果已知目標的真實邊界框，那麼這裡的『好』該如何如何量化呢？
直觀地說，可以衡量錨框和真實邊界框之間的相似性。
*傑卡德係數*（Jaccard）可以衡量兩組之間的相似性。
給定集合$\mathcal{A}$和$\mathcal{B}$，他們的傑卡德係數是他們交集的大小除以他們並集的大小：

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

事實上，我們可以將任何邊界框的畫素區域視為一組畫素。通
過這種方式，我們可以透過其畫素集的傑卡德係數來測量兩個邊界框的相似性。
對於兩個邊界框，它們的傑卡德係數通常稱為*交併比*（intersection over union，IoU），即兩個邊界框相交面積與相併面積之比，如 :numref:`fig_iou`所示。
交併比的取值範圍在0和1之間：0表示兩個邊界框無重合畫素，1表示兩個邊界框完全重合。

![交併比是兩個邊界框相交面積與相併面積之比。](../img/iou.svg)
:label:`fig_iou`

接下來部分將使用交併比來衡量錨框和真實邊界框之間、以及不同錨框之間的相似度。
給定兩個錨框或邊界框的清單，以下`box_iou`函式將在這兩個清單中計算它們成對的交併比。

```{.python .input}
#@save
def box_iou(boxes1, boxes2):
    """計算兩個錨框或邊界框清單中成對的交併比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形狀:
    # boxes1：(boxes1的數量,4),
    # boxes2：(boxes2的數量,4),
    # areas1：(boxes1的數量,),
    # areas2：(boxes2的數量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # inter_upperlefts,inter_lowerrights,inters的形狀:
    # (boxes1的數量,boxes2的數量,2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # inter_areasandunion_areas的形狀:(boxes1的數量,boxes2的數量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """計算兩個錨框或邊界框清單中成對的交併比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形狀:
    # boxes1：(boxes1的數量,4),
    # boxes2：(boxes2的數量,4),
    # areas1：(boxes1的數量,),
    # areas2：(boxes2的數量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形狀:
    # (boxes1的數量,boxes2的數量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形狀:(boxes1的數量,boxes2的數量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab paddle
#@save
def box_iou(boxes1, boxes2):
    """計算兩個錨框或邊界框清單中成對的交併比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形狀:
    # boxes1：(boxes1的數量,4),
    # boxes2：(boxes2的數量,4),
    # areas1：(boxes1的數量,),
    # areas2：(boxes2的數量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形狀:
    # (boxes1的數量,boxes2的數量,2)
    inter_upperlefts = paddle.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = paddle.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # inter_areasandunion_areas的形狀:(boxes1的數量,boxes2的數量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## 在訓練資料中標注錨框
:label:`subsec_labeling-anchor-boxes`

在訓練集中，我們將每個錨框視為一個訓練樣本。
為了訓練目標檢測模型，我們需要每個錨框的*類別*（class）和*偏移量*（offset）標籤，其中前者是與錨框相關的物件的類別，後者是真實邊界框相對於錨框的偏移量。
在預測時，我們為每個影象生成多個錨框，預測所有錨框的類別和偏移量，根據預測的偏移量調整它們的位置以獲得預測的邊界框，最後只輸出符合特定條件的預測邊界框。

目標檢測訓練集帶有*真實邊界框*的位置及其包圍物體類別的標籤。
要標記任何生成的錨框，我們可以參考分配到的最接近此錨框的真實邊界框的位置和類別標籤。
下文將介紹一個演算法，它能夠把最接近的真實邊界框分配給錨框。

### [**將真實邊界框分配給錨框**]

給定影象，假設錨框是$A_1, A_2, \ldots, A_{n_a}$，真實邊界框是$B_1, B_2, \ldots, B_{n_b}$，其中$n_a \geq n_b$。
讓我們定義一個矩陣$\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$，其中第$i$行、第$j$列的元素$x_{ij}$是錨框$A_i$和真實邊界框$B_j$的IoU。
該演算法包含以下步驟。

1. 在矩陣$\mathbf{X}$中找到最大的元素，並將它的行索引和列索引分別表示為$i_1$和$j_1$。然後將真實邊界框$B_{j_1}$分配給錨框$A_{i_1}$。這很直觀，因為$A_{i_1}$和$B_{j_1}$是所有錨框和真實邊界框配對中最相近的。在第一個分配完成後，丟棄矩陣中${i_1}^\mathrm{th}$行和${j_1}^\mathrm{th}$列中的所有元素。
1. 在矩陣$\mathbf{X}$中找到剩餘元素中最大的元素，並將它的行索引和列索引分別表示為$i_2$和$j_2$。我們將真實邊界框$B_{j_2}$分配給錨框$A_{i_2}$，並丟棄矩陣中${i_2}^\mathrm{th}$行和${j_2}^\mathrm{th}$列中的所有元素。
1. 此時，矩陣$\mathbf{X}$中兩行和兩列中的元素已被丟棄。我們繼續，直到丟棄掉矩陣$\mathbf{X}$中$n_b$列中的所有元素。此時已經為這$n_b$個錨框各自分配了一個真實邊界框。
1. 只遍歷剩下的$n_a - n_b$個錨框。例如，給定任何錨框$A_i$，在矩陣$\mathbf{X}$的第$i^\mathrm{th}$行中找到與$A_i$的IoU最大的真實邊界框$B_j$，只有當此IoU大於預定義的閾值時，才將$B_j$分配給$A_i$。

下面用一個具體的例子來說明上述演算法。
如 :numref:`fig_anchor_label`（左）所示，假設矩陣$\mathbf{X}$中的最大值為$x_{23}$，我們將真實邊界框$B_3$分配給錨框$A_2$。
然後，我們丟棄矩陣第2行和第3列中的所有元素，在剩餘元素（陰影區域）中找到最大的$x_{71}$，然後將真實邊界框$B_1$分配給錨框$A_7$。
接下來，如 :numref:`fig_anchor_label`（中）所示，丟棄矩陣第7行和第1列中的所有元素，在剩餘元素（陰影區域）中找到最大的$x_{54}$，然後將真實邊界框$B_4$分配給錨框$A_5$。
最後，如 :numref:`fig_anchor_label`（右）所示，丟棄矩陣第5行和第4列中的所有元素，在剩餘元素（陰影區域）中找到最大的$x_{92}$，然後將真實邊界框$B_2$分配給錨框$A_9$。
之後，我們只需要遍歷剩餘的錨框$A_1, A_3, A_4, A_6, A_8$，然後根據閾值確定是否為它們分配真實邊界框。

![將真實邊界框分配給錨框。](../img/anchor-label.svg)
:label:`fig_anchor_label`

此演算法在下面的`assign_anchor_to_bbox`函式中實現。

```{.python .input}
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """將最接近的真實邊界框分配給錨框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位於第i行和第j列的元素x_ij是錨框i和真實邊界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 對於每個錨框，分配的真實邊界框的張量
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # 根據閾值，決定是否分配真實邊界框
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """將最接近的真實邊界框分配給錨框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位於第i行和第j列的元素x_ij是錨框i和真實邊界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 對於每個錨框，分配的真實邊界框的張量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根據閾值，決定是否分配真實邊界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab paddle
#@save
def assign_anchor_to_bbox(ground_truth, anchors, place, iou_threshold=0.5):
    """將最接近的真實邊界框分配給錨框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位於第i行和第j列的元素x_ij是錨框i和真實邊界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 對於每個錨框，分配的真實邊界框的張量
    anchors_bbox_map = paddle.full((num_anchors,), -1, dtype=paddle.int64)
    # 根據閾值，決定是否分配真實邊界框
    max_ious = paddle.max(jaccard, axis=1)
    indices = paddle.argmax(jaccard, axis=1)
    anc_i = paddle.nonzero(max_ious >= 0.5).reshape([-1])
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = paddle.full((num_anchors,), -1)
    row_discard = paddle.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = paddle.argmax(jaccard)
        box_idx = paddle.cast((max_idx % num_gt_boxes), dtype='int64')
        anc_idx = paddle.cast((max_idx / num_gt_boxes), dtype='int64')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### 標記類別和偏移量

現在我們可以為每個錨框標記類別和偏移量了。
假設一個錨框$A$被分配了一個真實邊界框$B$。
一方面，錨框$A$的類別將被標記為與$B$相同。
另一方面，錨框$A$的偏移量將根據$B$和$A$中心坐標的相對位置以及這兩個框的相對大小進行標記。
鑑於資料集內不同的框的位置和大小不同，我們可以對那些相對位置和大小應用變換，使其獲得分佈更均勻且易於擬合的偏移量。
這裡介紹一種常見的變換。
[**給定框$A$和$B$，中心坐標分別為$(x_a, y_a)$和$(x_b, y_b)$，寬度分別為$w_a$和$w_b$，高度分別為$h_a$和$h_b$，可以將$A$的偏移量標記為：

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
其中常數的預設值為 $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$ ， $\sigma_w=\sigma_h=0.2$。
這種轉換在下面的 `offset_boxes` 函式中實現。

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """對錨框偏移量的轉換"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

如果一個錨框沒有被分配真實邊界框，我們只需將錨框的類別標記為*背景*（background）。
背景類別的錨框通常被稱為*負類*錨框，其餘的被稱為*正類*錨框。
我們使用真實邊界框（`labels`引數）實現以下`multibox_target`函式，來[**標記錨框的類別和偏移量**]（`anchors`引數）。
此函式將背景類別的索引設定為零，然後將新類別的整數索引遞增一。

```{.python .input}
#@save
def multibox_target(anchors, labels):
    """使用真實邊界框標記錨框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # 將類標籤和分配的邊界框坐標初始化為零
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # 使用真實邊界框來標記錨框的類別。
        # 如果一個錨框沒有被分配，標記其為背景（值為零）
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量轉換
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """使用真實邊界框標記錨框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 將類標籤和分配的邊界框坐標初始化為零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真實邊界框來標記錨框的類別。
        # 如果一個錨框沒有被分配，標記其為背景（值為零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量轉換
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab paddle
#@save
def multibox_target(anchors, labels):
    """使用真實邊界框標記錨框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    place, num_anchors = anchors.place, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, place)
        bbox_mask = paddle.tile(paddle.to_tensor((anchors_bbox_map >= 0), dtype='float32').unsqueeze(-1), (1, 4))
        # 將類標籤和分配的邊界框坐標初始化為零
        class_labels = paddle.zeros(paddle.to_tensor(num_anchors), dtype=paddle.int64)
        assigned_bb = paddle.zeros(paddle.to_tensor((num_anchors, 4)), dtype=paddle.float32)
        # 使用真實邊界框來標記錨框的類別。
        # 如果一個錨框沒有被分配，我們標記其為背景（值為零）
        indices_true = paddle.nonzero(anchors_bbox_map >= 0).numpy()
        bb_idx = anchors_bbox_map[indices_true].numpy()
        class_labels[indices_true] = label.numpy()[bb_idx, 0][:] + 1
        assigned_bb[indices_true] = label.numpy()[bb_idx, 1:]
        class_labels = paddle.to_tensor(class_labels)
        assigned_bb = paddle.to_tensor(assigned_bb)
        # 偏移量轉換
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape([-1]))
        batch_mask.append(bbox_mask.reshape([-1]))
        batch_class_labels.append(class_labels)
    bbox_offset = paddle.stack(batch_offset)
    bbox_mask = paddle.stack(batch_mask)
    class_labels = paddle.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### 一個例子

下面透過一個具體的例子來說明錨框標籤。
我們已經為載入影象中的狗和貓定義了真實邊界框，其中第一個元素是類別（0代表狗，1代表貓），其餘四個元素是左上角和右下角的$(x, y)$軸坐標（範圍介於0和1之間）。
我們還建立了五個錨框，用左上角和右下角的坐標進行標記：$A_0, \ldots, A_4$（索引從0開始）。
然後我們[**在影象中繪製這些真實邊界框和錨框**]。

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

使用上面定義的`multibox_target`函式，我們可以[**根據狗和貓的真實邊界框，標註這些錨框的分類和偏移量**]。
在這個例子中，背景、狗和貓的類索引分別為0、1和2。
下面我們為錨框和真實邊界框樣本新增一個維度。

```{.python .input}
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

```{.python .input}
#@tab paddle
labels = multibox_target(anchors.unsqueeze(axis=0),
                         ground_truth.unsqueeze(axis=0))
```

回傳的結果中有三個元素，都是張量格式。第三個元素包含標記的輸入錨框的類別。

讓我們根據影象中的錨框和真實邊界框的位置來分析下面回傳的類別標籤。
首先，在所有的錨框和真實邊界框配對中，錨框$A_4$與貓的真實邊界框的IoU是最大的。
因此，$A_4$的類別被標記為貓。
去除包含$A_4$或貓的真實邊界框的配對，在剩下的配對中，錨框$A_1$和狗的真實邊界框有最大的IoU。
因此，$A_1$的類別被標記為狗。
接下來，我們需要遍歷剩下的三個未標記的錨框：$A_0$、$A_2$和$A_3$。
對於$A_0$，與其擁有最大IoU的真實邊界框的類別是狗，但IoU低於預定義的閾值（0.5），因此該類別別被標記為背景；
對於$A_2$，與其擁有最大IoU的真實邊界框的類別是貓，IoU超過閾值，所以類別被標記為貓；
對於$A_3$，與其擁有最大IoU的真實邊界框的類別是貓，但值低於閾值，因此該類別別被標記為背景。

```{.python .input}
#@tab all
labels[2]
```

回傳的第二個元素是掩碼（mask）變數，形狀為（批次大小，錨框數的四倍）。
掩碼變數中的元素與每個錨框的4個偏移量一一對應。
由於我們不關心對背景的檢測，負類別的偏移量不應影響目標函式。
透過元素乘法，掩碼變數中的零將在計算目標函式之前過濾掉負類偏移量。

```{.python .input}
#@tab all
labels[1]
```

回傳的第一個元素包含了為每個錨框標記的四個偏移值。
請注意，負類錨框的偏移量被標記為零。

```{.python .input}
#@tab all
labels[0]
```

## 使用非極大值抑制預測邊界框
:label:`subsec_predicting-bounding-boxes-nms`

在預測時，我們先為影象生成多個錨框，再為這些錨框一一預測類別和偏移量。
一個*預測好的邊界框*則根據其中某個帶有預測偏移量的錨框而生成。
下面我們實現了`offset_inverse`函式，該函式將錨框和偏移量預測作為輸入，並[**應用逆偏移變換來回傳預測的邊界框坐標**]。

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """根據帶有預測偏移量的錨框來預測邊界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

當有許多錨框時，可能會輸出許多相似的具有明顯重疊的預測邊界框，都圍繞著同一目標。
為了簡化輸出，我們可以使用*非極大值抑制*（non-maximum suppression，NMS）合併屬於同一目標的類似的預測邊界框。

以下是非極大值抑制的工作原理。
對於一個預測邊界框$B$，目標檢測模型會計算每個類別的預測機率。
假設最大的預測機率為$p$，則該機率所對應的類別$B$即為預測的類別。
具體來說，我們將$p$稱為預測邊界框$B$的*置信度*（confidence）。
在同一張影象中，所有預測的非背景邊界框都按置信度降序排序，以生成清單$L$。然後我們透過以下步驟操作排序清單$L$。

1. 從$L$中選取置信度最高的預測邊界框$B_1$作為基準，然後將所有與$B_1$的IoU超過預定閾值$\epsilon$的非基準預測邊界框從$L$中移除。這時，$L$保留了置信度最高的預測邊界框，去除了與其太過相似的其他預測邊界框。簡而言之，那些具有*非極大值*置信度的邊界框被*抑制*了。
1. 從$L$中選取置信度第二高的預測邊界框$B_2$作為又一個基準，然後將所有與$B_2$的IoU大於$\epsilon$的非基準預測邊界框從$L$中移除。
1. 重複上述過程，直到$L$中的所有預測邊界框都曾被用作基準。此時，$L$中任意一對預測邊界框的IoU都小於閾值$\epsilon$；因此，沒有一對邊界框過於相似。
1. 輸出清單$L$中的所有預測邊界框。

[**以下`nms`函式按降序對置信度進行排序並回傳其索引**]。

```{.python .input}
#@save
def nms(boxes, scores, iou_threshold):
    """對預測邊界框的置信度進行排序"""
    B = scores.argsort()[::-1]
    keep = []  # 保留預測邊界框的指標
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """對預測邊界框的置信度進行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留預測邊界框的指標
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

```{.python .input}
#@tab paddle
#@save
def nms(boxes, scores, iou_threshold):
    """對預測邊界框的置信度進行排序"""
    B = paddle.argsort(scores, axis=-1, descending=True)
    keep = []  # 保留預測邊界框的指標
    while B.numel().item() > 0:
        i = B[0]
        keep.append(i.item())
        if B.numel().item() == 1: break
        iou = box_iou(boxes[i.numpy(), :].reshape([-1, 4]),
                      paddle.to_tensor(boxes.numpy()[B[1:].numpy(), :]).reshape([-1, 4])).reshape([-1])
        inds = paddle.nonzero(iou <= iou_threshold).numpy().reshape([-1])
        B = paddle.to_tensor(B.numpy()[inds + 1])
    return paddle.to_tensor(keep, place=boxes.place, dtype='int64')
```

我們定義以下`multibox_detection`函式來[**將非極大值抑制應用於預測邊界框**]。
這裡的實現有點複雜，請不要擔心。我們將在實現之後，馬上用一個具體的例子來展示它是如何工作的。

```{.python .input}
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非極大值抑制來預測邊界框"""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，並將類設定為背景
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一個用於非背景預測的閾值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非極大值抑制來預測邊界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，並將類設定為背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一個用於非背景預測的閾值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab paddle
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非極大值抑制來預測邊界框"""
    batch_size = cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape([-1, 4])
        conf = paddle.max(cls_prob[1:], 0)
        class_id = paddle.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，並將類設定為背景
        all_idx = paddle.arange(num_anchors, dtype='int64')
        combined = paddle.concat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = paddle.concat([keep, non_keep])
        class_id[non_keep.numpy()] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一個用於非背景預測的閾值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx.numpy()] = -1
        conf[below_min_idx.numpy()] = 1 - conf[below_min_idx.numpy()]
        pred_info = paddle.concat((paddle.to_tensor(class_id, dtype='float32').unsqueeze(1),
                               paddle.to_tensor(conf, dtype='float32').unsqueeze(1),
                               predicted_bb), axis=1)
        out.append(pred_info)
    return paddle.stack(out)
```

現在讓我們[**將上述演算法應用到一個帶有四個錨框的具體範例中**]。
為簡單起見，我們假設預測的偏移量都是零，這意味著預測的邊界框即是錨框。
對於背景、狗和貓其中的每個類，我們還定義了它的預測機率。

```{.python .input}
#@tab mxnet, pytorch
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # 背景的預測機率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的預測機率
                      [0.1, 0.2, 0.3, 0.9]])  # 貓的預測機率
```

```{.python .input}
#@tab paddle
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * anchors.numel().item())
cls_probs = d2l.tensor([[0] * 4,  # 背景的預測機率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的預測機率
                      [0.1, 0.2, 0.3, 0.9]])  # 貓的預測機率
```

我們可以[**在影象上繪製這些預測邊界框和置信度**]。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

現在我們可以呼叫`multibox_detection`函式來執行非極大值抑制，其中閾值設定為0.5。
請注意，我們在範例的張量輸入中新增了維度。

我們可以看到[**回傳結果的形狀是（批次大小，錨框的數量，6）**]。
最內層維度中的六個元素提供了同一預測邊界框的輸出訊息。
第一個元素是預測的類索引，從0開始（0代表狗，1代表貓），值-1表示背景或在非極大值抑制中被移除了。
第二個元素是預測的邊界框的置信度。
其餘四個元素分別是預測邊界框左上角和右下角的$(x, y)$軸坐標（範圍介於0和1之間）。

```{.python .input}
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab paddle
output = multibox_detection(cls_probs.unsqueeze(axis=0),
                            offset_preds.unsqueeze(axis=0),
                            anchors.unsqueeze(axis=0),
                            nms_threshold=0.5)
output
```

刪除-1類別（背景）的預測邊界框後，我們可以[**輸出由非極大值抑制儲存的最終預測邊界框**]。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

實踐中，在執行非極大值抑制前，我們甚至可以將置信度較低的預測邊界框移除，從而減少此演算法中的計算量。
我們也可以對非極大值抑制的輸出結果進行後處理。例如，只保留置信度更高的結果作為最終輸出。

## 小結

* 我們以影象的每個畫素為中心生成不同形狀的錨框。
* 交併比（IoU）也被稱為傑卡德係數，用於衡量兩個邊界框的相似性。它是相交面積與相併面積的比率。
* 在訓練集中，我們需要給每個錨框兩種型別的標籤。一個是與錨框中目標檢測的類別，另一個是錨框真實相對於邊界框的偏移量。
* 預測期間可以使用非極大值抑制（NMS）來移除類似的預測邊界框，從而簡化輸出。

## 練習

1. 在`multibox_prior`函式中更改`sizes`和`ratios`的值。生成的錨框有什麼變化？
1. 建立並可視化兩個IoU為0.5的邊界框。它們是怎樣重疊的？
1. 在 :numref:`subsec_labeling-anchor-boxes`和 :numref:`subsec_predicting-bounding-boxes-nms`中修改變數`anchors`，結果如何變化？
1. 非極大值抑制是一種貪心演算法，它透過*移除*來抑制預測的邊界框。是否存在一種可能，被移除的一些框實際上是有用的？如何修改這個演算法來柔和地抑制？可以參考Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`。
1. 如果非手動，非最大限度的抑制可以被學習嗎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2945)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2946)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11804)
:end_tab:
