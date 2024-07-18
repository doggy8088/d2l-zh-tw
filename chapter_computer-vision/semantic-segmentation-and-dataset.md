# 語義分割和數據集
:label:`sec_semantic_segmentation`

在 :numref:`sec_bbox`— :numref:`sec_rcnn`中討論的目標檢測問題中，我們一直使用方形邊界框來標註和預測影象中的目標。
本節將探討*語義分割*（semantic segmentation）問題，它重點關注於如何將影象分割成屬於不同語義類別的區域。
與目標檢測不同，語義分割可以識別並理解影象中每一個畫素的內容：其語義區域的標註和預測是畫素級的。
 :numref:`fig_segmentation`展示了語義分割中影象有關狗、貓和背景的標籤。
與目標檢測相比，語義分割標註的畫素級的邊框顯然更加精細。

![語義分割中影象有關狗、貓和背景的標籤](../img/segmentation.svg)
:label:`fig_segmentation`

## 影象分割和實例分割

電腦視覺領域還有2個與語義分割相似的重要問題，即*影象分割*（image segmentation）和*實例分割*（instance segmentation）。
我們在這裡將它們同語義分割簡單區分一下。

* *影象分割*將影象劃分為若干組成區域，這類問題的方法通常利用影象中畫素之間的相關性。它在訓練時不需要有關影象畫素的標籤訊息，在預測時也無法保證分割出的區域具有我們希望得到的語義。以 :numref:`fig_segmentation`中的影象作為輸入，影象分割可能會將狗分為兩個區域：一個覆蓋以黑色為主的嘴和眼睛，另一個覆蓋以黃色為主的其餘部分身體。
* *實例分割*也叫*同時檢測並分割*（simultaneous detection and segmentation），它研究如何識別影象中各個目標實例的畫素級區域。與語義分割不同，實例分割不僅需要區分語義，還要區分不同的目標實例。例如，如果影象中有兩條狗，則實例分割需要區分畫素屬於的兩條狗中的哪一條。

## Pascal VOC2012 語義分割資料集

[**最重要的語義分割資料集之一是[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)。**]
下面我們深入瞭解一下這個資料集。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.vision as paddlevision
import os
```

資料集的tar檔案大約為2GB，所以下載可能需要一段時間。
提取出的資料集位於`../data/VOCdevkit/VOC2012`。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

進入路徑`../data/VOCdevkit/VOC2012`之後，我們可以看到資料集的不同元件。
`ImageSets/Segmentation`路徑包含用於訓練和測試樣本的文字檔案，而`JPEGImages`和`SegmentationClass`路徑分別儲存著每個範例的輸入影象和標籤。
此處的標籤也採用影象格式，其尺寸和它所標註的輸入影象的尺寸相同。
此外，標籤中顏色相同的畫素屬於同一個語義類別。
下面將`read_voc_images`函式定義為[**將所有輸入的影象和標籤讀入記憶體**]。

```{.python .input}
#@save
def read_voc_images(voc_dir, is_train=True):
    """讀取所有VOC影象並標註"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """讀取所有VOC影象並標註"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab paddle
#@save
def read_voc_images(voc_dir, is_train=True):
    """Defined in :numref:`sec_semantic_segmentation`"""
    """讀取所有VOC影象並標註
    Defined in :numref:`sec_semantic_segmentation`"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(paddle.vision.image.image_load(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg'), backend='cv2')[..., ::-1].transpose(
            [2, 0, 1]))
        labels.append(paddle.vision.image.image_load(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png'), backend='cv2')[..., ::-1].transpose(
            [2, 0, 1]))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

下面我們[**繪製前5個輸入影象及其標籤**]。
在標籤影象中，白色和黑色分別表示邊框和背景，而其他顏色則對應不同的類別。

```{.python .input}
n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab paddle
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.transpose([1, 2, 0]) for img in imgs]
d2l.show_images(imgs, 2, n);
```

接下來，我們[**列舉RGB顏色值和類別名稱**]。

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

透過上面定義的兩個常數，我們可以方便地[**查詢標籤中每個畫素的類索引**]。
我們定義了`voc_colormap2label`函式來建立從上述RGB顏色值到類別索引的對映，而`voc_label_indices`函式將RGB值對映到在Pascal VOC2012資料集中的類別索引。

```{.python .input}
#@save
def voc_colormap2label():
    """建立從RGB到VOC類別索引的對映"""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """將VOC標籤中的RGB值對映到它們的類別索引"""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """建立從RGB到VOC類別索引的對映"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """將VOC標籤中的RGB值對映到它們的類別索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab paddle
#@save
def voc_colormap2label():
    """建立從RGB到VOC類別索引的對映"""
    colormap2label = paddle.zeros([256 ** 3], dtype=paddle.int64)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label
    
#@save
def voc_label_indices(colormap, colormap2label):
    """將VOC標籤中的RGB值對映到它們的類別索引"""
    colormap = colormap.transpose([1, 2, 0]).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**例如**]，在第一張樣本影象中，飛機頭部區域的類別索引為1，而背景索引為0。

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### 預處理資料

在之前的實驗，例如 :numref:`sec_alexnet`— :numref:`sec_googlenet`中，我們透過再縮放影象使其符合模型的輸入形狀。
然而在語義分割中，這樣做需要將預測的畫素類別重新對映回原始尺寸的輸入影象。
這樣的對映可能不夠精確，尤其在不同語義的分割區域。
為了避免這個問題，我們將影象裁剪為固定尺寸，而不是再縮放。
具體來說，我們[**使用影象增廣中的隨機裁剪，裁剪輸入影象和標籤的相同區域**]。

```{.python .input}
#@save
def voc_rand_crop(feature, label, height, width):
    """隨機裁剪特徵和標籤影象"""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """隨機裁剪特徵和標籤影象"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab paddle
#@save
def voc_rand_crop(feature, label, height, width):
    """隨機裁剪特徵和標籤影象"""
    rect = paddle.vision.transforms.RandomCrop((height, width))._get_param(
        img=feature, output_size=(height, width))
    feature = paddle.vision.transforms.crop(feature, *rect)
    label = paddle.vision.transforms.crop(label, *rect)
    return feature, label
```

```{.python .input}
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab paddle
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0].transpose([1, 2, 0]), train_labels[0].transpose([1, 2, 0]), 200, 300)
    
imgs = [img for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**自定義語義分割資料集類**]

我們透過繼承高階API提供的`Dataset`類，自定義了一個語義分割資料集類`VOCSegDataset`。
透過實現`__getitem__`函式，我們可以任意訪問資料集中索引為`idx`的輸入影象及其每個畫素的類別索引。
由於資料集中有些影象的尺寸可能小於隨機裁剪所指定的輸出尺寸，這些樣本可以透過自定義的`filter`函式移除掉。
此外，我們還定義了`normalize_image`函式，從而對輸入影象的RGB三個通道的值分別做標準化。

```{.python .input}
#@save
class VOCSegDataset(gluon.data.Dataset):
    """一個用於載入VOC資料集的自定義資料集"""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """一個用於載入VOC資料集的自定義資料集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab paddle
#@save
class VOCSegDataset(paddle.io.Dataset):
    """一個用於載入VOC資料集的自定義資料集
    Defined in :numref:`sec_semantic_segmentation`"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = paddle.vision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.astype("float32") / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature = paddle.to_tensor(self.features[idx],dtype='float32')
        label = paddle.to_tensor(self.labels[idx],dtype='float32')
        feature, label = voc_rand_crop(feature,label,
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**讀取資料集**]

我們透過自定義的`VOCSegDataset`類來分別建立訓練集和測試集的實例。
假設我們指定隨機裁剪的輸出影象的形狀為$320\times 480$，
下面我們可以檢視訓練集和測試集所保留的樣本個數。

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

設批次大小為64，我們定義訓練集的迭代器。
列印第一個小批次的形狀會發現：與影象分類或目標檢測不同，這裡的標籤是一個三維陣列。

```{.python .input}
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab paddle
batch_size = 64
train_iter = paddle.io.DataLoader(voc_train, batch_size=batch_size, shuffle=True,
                                  drop_last=True,
                                  return_list=True,
                                  num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**整合所有元件**]

最後，我們定義以下`load_data_voc`函式來下載並讀取Pascal VOC2012語義分割資料集。
它回傳訓練集和測試集的資料迭代器。

```{.python .input}
#@save
def load_data_voc(batch_size, crop_size):
    """載入VOC語義分割資料集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """載入VOC語義分割資料集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab paddle
#@save
def load_data_voc(batch_size, crop_size):
    """載入VOC語義分割資料集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = paddle.io.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size=batch_size,
        shuffle=True, return_list=True, drop_last=True, num_workers=num_workers)
    test_iter = paddle.io.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size=batch_size,
        drop_last=True, return_list=True, num_workers=num_workers)
    return train_iter, test_iter
```

## 小結

* 語義分割透過將影象劃分為屬於不同語義類別的區域，來識別並理解影象中畫素級別的內容。
* 語義分割的一個重要的資料集叫做Pascal VOC2012。
* 由於語義分割的輸入影象和標籤在畫素上一一對應，輸入影象會被隨機裁剪為固定尺寸而不是縮放。

## 練習

1. 如何在自動駕駛和醫療影象診斷中應用語義分割？還能想到其他領域的應用嗎？
1. 回想一下 :numref:`sec_image_augmentation`中對資料增強的描述。影象分類中使用的哪種影象增強方法是難以用於語義分割的？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3296)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3295)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11809)
:end_tab:
