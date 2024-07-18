# 影象分類資料集
:label:`sec_fashion_mnist`

(**MNIST資料集**) :cite:`LeCun.Bottou.Bengio.ea.1998`
(**是影象分類中廣泛使用的資料集之一，但作為基準資料集過於簡單。
我們將使用類似但更復雜的Fashion-MNIST資料集**) :cite:`Xiao.Rasul.Vollgraf.2017`。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import sys
import paddle
from paddle.vision import transforms

d2l.use_svg_display()
```

## 讀取資料集

我們可以[**透過框架中的內建函式將Fashion-MNIST資料集下載並讀取到記憶體中**]。

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# 透過ToTensor實例將影象資料從PIL型別變換成32位浮點數格式，
# 併除以255使得所有畫素的數值均在0～1之間
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

```{.python .input}
#@tab paddle
trans = transforms.ToTensor()
mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                  transform=trans)
mnist_test = paddle.vision.datasets.FashionMNIST(mode="test", transform=trans)
```

Fashion-MNIST由10個類別的影象組成，
每個類別由*訓練資料集*（train dataset）中的6000張影象
和*測試資料集*（test dataset）中的1000張影象組成。
因此，訓練集和測試集分別包含60000和10000張影象。
測試資料集不會用於訓練，只用於評估模型效能。

```{.python .input}
#@tab mxnet, pytorch, paddle
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

每個輸入影象的高度和寬度均為28畫素。
資料集由灰度影象組成，其通道數為1。
為了簡潔起見，本書將高度$h$畫素、寬度$w$畫素影象的形狀記為$h \times w$或（$h$,$w$）。

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

[~~兩個視覺化資料集的函式~~]

Fashion-MNIST中包含的10個類別，分別為t-shirt（T恤）、trouser（褲子）、pullover（套衫）、dress（連衣裙）、coat（外套）、sandal（涼鞋）、shirt（襯衫）、sneaker（運動鞋）、bag（套件）和ankle boot（短靴）。
以下函式用於在數字標籤索引及其文字名稱之間進行轉換。

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """回傳Fashion-MNIST資料集的文字標籤"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

我們現在可以建立一個函式來視覺化這些樣本。

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """繪製圖像清單"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """繪製圖像清單"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 圖片張量
            ax.imshow(img.numpy())
        else:
            # PIL圖片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab paddle
#@save
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """繪製圖像清單"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if paddle.is_tensor(img):
            # 圖片張量
            ax.imshow(img.numpy())
        else:
            # PIL圖片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

以下是訓練資料集中前[**幾個樣本的影象及其相應的標籤**]。

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab paddle
X, y = next(iter(paddle.io.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape([18, 28, 28]), 2, 9, titles=get_fashion_mnist_labels(y));
```

## 讀取小批次

為了使我們在讀取訓練集和測試集時更容易，我們使用內建的資料迭代器，而不是從零開始建立。
回顧一下，在每次迭代中，資料載入器每次都會[**讀取一小批次資料，大小為`batch_size`**]。
透過內建資料迭代器，我們可以隨機打亂了所有樣本，從而無偏見地讀取小批次。

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """在非Windows的平台上，使用4個程序來讀取資料"""
    return 0 if sys.platform.startswith('win') else 4

# 透過ToTensor實例將影象資料從uint8格式變換成32位浮點數格式，併除以255使得所有畫素的數值
# 均在0～1之間
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4個程序來讀取資料"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

```{.python .input}
#@tab paddle
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4個程序來讀取資料"""
    return 4

train_iter = paddle.io.DataLoader(dataset=mnist_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  return_list=True,
                                  num_workers=get_dataloader_workers())
```

我們看一下讀取訓練資料所需的時間。

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## 整合所有元件

現在我們[**定義`load_data_fashion_mnist`函式**]，用於獲取和讀取Fashion-MNIST資料集。
這個函式回傳訓練集和驗證集的資料迭代器。
此外，這個函式還接受一個可選引數`resize`，用來將影象大小調整為另一種形狀。

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下載Fashion-MNIST資料集，然後將其載入到記憶體中"""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下載Fashion-MNIST資料集，然後將其載入到記憶體中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """下載Fashion-MNIST資料集，然後將其載入到記憶體中"""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # 將所有數字除以255，使所有畫素值介於0和1之間，在最後新增一個批處理維度，
    # 並將標籤轉換為int32。
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

```{.python .input}
#@tab paddle
#@save
def load_data_fashion_mnist(batch_size, resize=None):  
    """下載Fashion-MNIST資料集，然後將其載入到記憶體中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                      transform=trans)
    mnist_test = paddle.vision.datasets.FashionMNIST(mode="test",
                                                     transform=trans)
    return (paddle.io.DataLoader(dataset=mnist_train,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 return_list=True,
                                 num_workers=get_dataloader_workers()),
            paddle.io.DataLoader(dataset=mnist_test,
                                 batch_size=batch_size,
                                 return_list=True,
                                 shuffle=True,
                                 num_workers=get_dataloader_workers()))
```

下面，我們透過指定`resize`引數來測試`load_data_fashion_mnist`函式的影象大小調整功能。

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

我們現在已經準備好使用Fashion-MNIST資料集，便於下面的章節呼叫來評估各種分類演算法。

## 小結

* Fashion-MNIST是一個服裝分類資料集，由10個類別的影象組成。我們將在後續章節中使用此資料集來評估各種分類演算法。
* 我們將高度$h$畫素，寬度$w$畫素影象的形狀記為$h \times w$或（$h$,$w$）。
* 資料迭代器是獲得更高效能的關鍵元件。依靠實現良好的資料迭代器，利用高效能計算來避免減慢訓練過程。

## 練習

1. 減少`batch_size`（如減少到1）是否會影響讀取效能？
1. 資料迭代器的效能非常重要。當前的實現足夠快嗎？探索各種選擇來改進它。
1. 查閱框架的線上API文件。還有哪些其他資料集可用？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1788)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1787)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1786)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11692)
:end_tab:
