# 目標檢測資料集
:label:`sec_object-detection-dataset`

目標檢測領域沒有像MNIST和Fashion-MNIST那樣的小數據集。
為了快速測試目標檢測模型，[**我們收集並標記了一個小型資料集**]。
首先，我們拍攝了一組香蕉的照片，並生成了1000張不同角度和大小的香蕉影象。
然後，我們在一些背景圖片的隨機位置上放一張香蕉的影象。
最後，我們在圖片上為這些香蕉標記了邊界框。

## [**下載資料集**]

包含所有影象和CSV標籤檔案的香蕉檢測資料集可以直接從網際網路下載。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import paddle
import paddle.vision as paddlevision
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 讀取資料集

透過`read_data_bananas`函式，我們[**讀取香蕉檢測資料集**]。
該資料集包括一個的CSV檔案，內含目標類別別標籤和位於左上角和右下角的真實邊界框坐標。

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """讀取香蕉檢測資料集中的影象和標籤"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 這裡的target包含（類別，左上角x，左上角y，右下角x，右下角y），
        # 其中所有影象都具有相同的香蕉類（索引為0）
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """讀取香蕉檢測資料集中的影象和標籤"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 這裡的target包含（類別，左上角x，左上角y，右下角x，右下角y），
        # 其中所有影象都具有相同的香蕉類（索引為0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

```{.python .input}
#@tab paddle
#@save
def read_data_bananas(is_train=True):
    """讀取香蕉檢測資料集中的影象和標籤"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        paddle.vision.set_image_backend('cv2')
        images.append(paddlevision.image_load(os.path.join(data_dir, 'bananas_train' if is_train else
        'bananas_val', 'images', f'{img_name}'))[..., ::-1])
        # 這裡的target包含（類別，左上角x，左上角y，右下角x，右下角y）
        # 其中所有影象都具有相同的香蕉類（索引為0）
        targets.append(list(target))
    return images, paddle.to_tensor(targets).unsqueeze(1) / 256
```

透過使用`read_data_bananas`函式讀取影象和標籤，以下`BananasDataset`類別將允許我們[**建立一個自定義`Dataset`實例**]來載入香蕉檢測資料集。

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """一個用於載入香蕉檢測資料集的自定義資料集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """一個用於載入香蕉檢測資料集的自定義資料集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab paddle
#@save
class BananasDataset(paddle.io.Dataset):
    """一個用於載入香蕉檢測資料集的自定義資料集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (paddle.to_tensor(self.features[idx], dtype='float32').transpose([2, 0, 1]), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

最後，我們定義`load_data_bananas`函式，來[**為訓練集和測試集回傳兩個資料載入器實例**]。對於測試集，無須按隨機順序讀取它。

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """載入香蕉檢測資料集"""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """載入香蕉檢測資料集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab paddle
#@save
def load_data_bananas(batch_size):
    """載入香蕉檢測資料集"""
    train_iter = paddle.io.DataLoader(BananasDataset(is_train=True),
                                      batch_size=batch_size, return_list=True, shuffle=True)
    val_iter = paddle.io.DataLoader(BananasDataset(is_train=False),
                                    batch_size=batch_size, return_list=True)
    return train_iter, val_iter
```

讓我們[**讀取一個小批次，並列印其中的影象和標籤的形狀**]。
影象的小批次的形狀為（批次大小、通道數、高度、寬度），看起來很眼熟：它與我們之前影象分類任務中的相同。
標籤的小批次的形狀為（批次大小，$m$，5），其中$m$是資料集的任何影象中邊界框可能出現的最大數量。

小批次計算雖然高效，但它要求每張影象含有相同數量的邊界框，以便放在同一個批次中。
通常來說，影象可能擁有不同數量個邊界框；因此，在達到$m$之前，邊界框少於$m$的影象將被非法邊界框填充。
這樣，每個邊界框的標籤將被長度為5的陣列表示。
陣列中的第一個元素是邊界框中物件的類別，其中-1表示用於填充的非法邊界框。
陣列的其餘四個元素是邊界框左上角和右下角的（$x$，$y$）坐標值（值域在0～1之間）。
對於香蕉資料集而言，由於每張影象上只有一個邊界框，因此$m=1$。

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**示範**]

讓我們展示10幅帶有真實邊界框的影象。
我們可以看到在所有這些影象中香蕉的旋轉角度、大小和位置都有所不同。
當然，這只是一個簡單的人工資料集，實踐中真實世界的資料集通常要複雜得多。

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab paddle
imgs = (batch[0][0:10].transpose([0, 2, 3, 1])) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 小結

* 我們收集的香蕉檢測資料集可用於示範目標檢測模型。
* 用於目標檢測的資料載入與影象分類的資料載入類似。但是，在目標檢測中，標籤還包含真實邊界框的訊息，它不出現在影象分類中。

## 練習

1. 在香蕉檢測資料集中示範其他帶有真實邊界框的影象。它們在邊界框和目標方面有什麼不同？
1. 假設我們想要將資料增強（例如隨機裁剪）應用於目標檢測。它與影象分類中的有什麼不同？提示：如果裁剪的影象只包含物體的一小部分會怎樣？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3203)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3202)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11806)
:end_tab:
