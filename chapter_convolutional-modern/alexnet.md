# 深度卷積神經網路（AlexNet）
:label:`sec_alexnet`

在LeNet提出後，卷積神經網路在電腦視覺和機器學習領域中很有名氣。但卷積神經網路並沒有主導這些領域。這是因為雖然LeNet在小數據集上取得了很好的效果，但是在更大、更真實的資料集上訓練卷積神經網路的效能和可行性還有待研究。事實上，在上世紀90年代初到2012年之間的大部分時間裡，神經網路往往被其他機器學習方法超越，如支援向量機（support vector machines）。

在電腦視覺中，直接將神經網路與其他機器學習方法進行比較也許不公平。這是因為，卷積神經網路的輸入是由原始畫素值或是經過簡單預處理（例如居中、縮放）的畫素值組成的。但在使用傳統機器學習方法時，從業者永遠不會將原始畫素作為輸入。在傳統機器學習方法中，電腦視覺流水線是由經過人的手工精心設計的特徵流水線組成的。對於這些傳統方法，大部分的進展都來自於對特徵有了更聰明的想法，並且學習到的演算法往往歸於事後的解釋。

雖然上世紀90年代就有了一些神經網路加速卡，但僅靠它們還不足以開發出有大量引數的深層多通道多層卷積神經網路。此外，當時的資料集仍然相對較小。除了這些障礙，訓練神經網路的一些關鍵技巧仍然缺失，包括啟發式引數初始化、隨機梯度下降的變體、非擠壓啟用函式和有效的正則化技術。

因此，與訓練*端到端*（從畫素到分類結果）系統不同，經典機器學習的流水線看起來更像下面這樣：

1. 獲取一個有趣的資料集。在早期，收集這些資料集需要昂貴的傳感器（在當時最先進的影象也就100萬畫素）。
2. 根據光學、幾何學、其他知識以及偶然的發現，手工對特徵資料集進行預處理。
3. 透過標準的特徵提取演算法，如SIFT（尺度不變特徵變換） :cite:`Lowe.2004`和SURF（加速魯棒特徵） :cite:`Bay.Tuytelaars.Van-Gool.2006`或其他手動調整的流水線來輸入資料。
4. 將提取的特徵送入最喜歡的分類器中（例如線性模型或其它核方法），以訓練分類器。

當人們和機器學習研究人員交談時，會發現機器學習研究人員相信機器學習既重要又美麗：優雅的理論去證明各種模型的性質。機器學習是一個正在蓬勃發展、嚴謹且非常有用的領域。然而，當人們和電腦視覺研究人員交談，會聽到一個完全不同的故事。電腦視覺研究人員會告訴一個詭異事實————推動領域進步的是資料特徵，而不是學習演算法。電腦視覺研究人員相信，從對最終模型精度的影響來說，更大或更乾淨的資料集、或是稍微改進的特徵提取，比任何學習演算法帶來的進步要大得多。

## 學習表徵

另一種預測這個領域發展的方法————觀察影象特徵的提取方法。在2012年前，影象特徵都是機械地計算出來的。事實上，設計一套新的特徵函式、改進結果，並撰寫論文是盛極一時的潮流。SIFT :cite:`Lowe.2004`、SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`、HOG（定向梯度直方圖） :cite:`Dalal.Triggs.2005`、[bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)和類似的特徵提取方法佔據了主導地位。

另一組研究人員，包括Yann LeCun、Geoff Hinton、Yoshua Bengio、Andrew Ng、Shun ichi Amari和Juergen Schmidhuber，想法則與眾不同：他們認為特徵本身應該被學習。此外，他們還認為，在合理地複雜性前提下，特徵應該由多個共同學習的神經網路層組成，每個層都有可學習的引數。在機器視覺中，最底層可能檢測邊緣、顏色和紋理。事實上，Alex Krizhevsky、Ilya Sutskever和Geoff Hinton提出了一種新的卷積神經網路變體*AlexNet*。在2012年ImageNet挑戰賽中取得了轟動一時的成績。AlexNet以Alex Krizhevsky的名字命名，他是論文 :cite:`Krizhevsky.Sutskever.Hinton.2012`的第一作者。

有趣的是，在網路的最底層，模型學習到了一些類似於傳統濾波器的特徵抽取器。 :numref:`fig_filters`是從AlexNet論文 :cite:`Krizhevsky.Sutskever.Hinton.2012`複製的，描述了底層影象特徵。

![AlexNet第一層學習到的特徵抽取器。](../img/filters.png)
:width:`400px`
:label:`fig_filters`

AlexNet的更高層建立在這些底層表示的基礎上，以表示更大的特徵，如眼睛、鼻子、草葉等等。而更高的層可以檢測整個物體，如人、飛機、狗或飛盤。最終的隱藏神經元可以學習影象的綜合表示，從而使屬於不同類別的資料易於區分。儘管一直有一群執著的研究者不斷鑽研，試圖學習視覺資料的逐級表徵，然而很長一段時間裡這些嘗試都未有突破。深度卷積神經網路的突破出現在2012年。突破可歸因於兩個關鍵因素。

### 缺少的成分：資料

包含許多特徵的深度模型需要大量的有標籤資料，才能顯著優於基於凸最佳化的傳統方法（如線性方法和核方法）。
然而，限於早期電腦有限的儲存和90年代有限的研究預算，大部分研究只基於小的公開資料集。例如，不少研究論文基於加州大學歐文分校（UCI）提供的若干個公開資料集，其中許多資料集只有幾百至幾千張在非自然環境下以低解析度拍攝的影象。這一狀況在2010年前後興起的大資料浪潮中得到改善。2009年，ImageNet資料集發布，併發起ImageNet挑戰賽：要求研究人員從100萬個樣本中訓練模型，以區分1000個不同類別的物件。ImageNet資料集由斯坦福教授李飛飛小組的研究人員開發，利用谷歌影象搜尋（Google Image Search）對每一類影象進行預篩選，並利用亞馬遜眾套件（Amazon Mechanical Turk）來標註每張圖片的相關類別。這種規模是前所未有的。這項被稱為ImageNet的挑戰賽推動了電腦視覺和機器學習研究的發展，挑戰研究人員確定哪些模型能夠在更大的資料規模下表現最好。

### 缺少的成分：硬體

深度學習對計算資源要求很高，訓練可能需要數百個迭代輪數，每次迭代都需要透過代價高昂的許多線性代數層傳遞資料。這也是為什麼在20世紀90年代至21世紀初，最佳化凸目標的簡單演算法是研究人員的首選。然而，用GPU訓練神經網路改變了這一格局。*圖形處理器*（Graphics Processing Unit，GPU）早年用來加速圖形處理，使電腦遊戲玩家受益。GPU可最佳化高吞吐量的$4 \times 4$矩陣和向量乘法，從而服務於基本的圖形任務。幸運的是，這些數學運算與卷積層的計算驚人地相似。由此，英偉達（NVIDIA）和ATI已經開始為通用計算操作最佳化gpu，甚至把它們作為*通用GPU*（general-purpose GPUs，GPGPU）來銷售。

那麼GPU比CPU強在哪裡呢？

首先，我們深度理解一下中央處理器（Central Processing Unit，CPU）的*核心*。
CPU的每個核心都擁有高時鐘頻率的執行能力，和高達數MB的三級快取（L3Cache）。
它們非常適合執行各種指令，具有分支預測器、深層流水線和其他使CPU能夠執行各種程式的功能。
然而，這種明顯的優勢也是它的致命弱點：通用核心的製造成本非常高。
它們需要大量的晶片面積、複雜的支援結構（記憶體介面、核心之間的快取邏輯、高速互連等等），而且它們在任何單個任務上的效能都相對較差。
現代筆記型電腦最多有4核，即使是高端伺服器也很少超過64核，因為它們的性價比不高。

相比於CPU，GPU由$100 \sim 1000$個小的處理單元組成（NVIDIA、ATI、ARM和其他晶片供應商之間的細節稍有不同），通常被分成更大的組（NVIDIA稱之為warps）。
雖然每個GPU核心都相對較弱，有時甚至以低於1GHz的時鐘頻率執行，但龐大的核心數量使GPU比CPU快幾個數量級。
例如，NVIDIA最近一代的Ampere GPU架構為每個晶片提供了高達312 TFlops的浮點效能，而CPU的浮點效能到目前為止還沒有超過1 TFlops。
之所以有如此大的差距，原因其實很簡單：首先，功耗往往會隨時鍾頻率呈二次方增長。
對於一個CPU核心，假設它的執行速度比GPU快4倍，但可以使用16個GPU核代替，那麼GPU的綜合性能就是CPU的$16 \times 1/4 = 4$倍。
其次，GPU核心要簡單得多，這使得它們更節能。
此外，深度學習中的許多操作需要相對較高的記憶體頻寬，而GPU擁有10倍於CPU的頻寬。

回到2012年的重大突破，當Alex Krizhevsky和Ilya Sutskever實現了可以在GPU硬體上執行的深度卷積神經網路時，一個重大突破出現了。他們意識到卷積神經網路中的計算瓶頸：卷積和矩陣乘法，都是可以在硬體上並行化的操作。
於是，他們使用兩個顯存為3GB的NVIDIA GTX580 GPU實現了快速卷積運算。他們的創新[cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)幾年來它一直是行業標準，並推動了深度學習熱潮。

## AlexNet

2012年，AlexNet橫空出世。它首次證明了學習到的特徵可以超越手工設計的特徵。它一舉打破了電腦視覺研究的現狀。
AlexNet使用了8層卷積神經網路，並以很大的優勢贏得了2012年ImageNet影象識別挑戰賽。

AlexNet和LeNet的架構非常相似，如 :numref:`fig_alexnet`所示。
注意，本書在這裡提供的是一個稍微精簡版本的AlexNet，去除了當年需要兩個小型GPU同時運算的設計特點。

![從LeNet（左）到AlexNet（右）](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet和LeNet的設計理念非常相似，但也存在顯著差異。

1. AlexNet比相對較小的LeNet5要深得多。AlexNet由八層組成：五個卷積層、兩個全連線隱藏層和一個全連線輸出層。
2. AlexNet使用ReLU而不是sigmoid作為其啟用函式。

下面的內容將深入研究AlexNet的細節。

### 模型設計

在AlexNet的第一層，卷積視窗的形狀是$11\times11$。
由於ImageNet中大多數影象的寬和高比MNIST影象的多10倍以上，因此，需要一個更大的卷積視窗來捕獲目標。
第二層中的卷積視窗形狀被縮減為$5\times5$，然後是$3\times3$。
此外，在第一層、第二層和第五層卷積層之後，加入視窗形狀為$3\times3$、步幅為2的最大彙聚層。
而且，AlexNet的卷積通道數目是LeNet的10倍。

在最後一個卷積層後有兩個全連線層，分別有4096個輸出。
這兩個巨大的全連線層擁有將近1GB的模型引數。
由於早期GPU顯存有限，原版的AlexNet採用了雙數據流設計，使得每個GPU只負責儲存和計算模型的一半引數。
幸運的是，現在GPU顯存相對充裕，所以現在很少需要跨GPU分解模型（因此，本書的AlexNet模型在這方面與原始論文稍有不同）。

### 啟用函式

此外，AlexNet將sigmoid啟用函式改為更簡單的ReLU啟用函式。
一方面，ReLU啟用函式的計算更簡單，它不需要如sigmoid啟用函式那般複雜的求冪運算。
另一方面，當使用不同的引數初始化方法時，ReLU啟用函式使訓練模型更加容易。
當sigmoid啟用函式的輸出非常接近於0或1時，這些區域的梯度幾乎為0，因此反向傳播無法繼續更新一些模型引數。
相反，ReLU啟用函式在正區間的梯度總是1。
因此，如果模型引數沒有正確初始化，sigmoid函式可能在正區間內得到幾乎為0的梯度，從而使模型無法得到有效的訓練。

### 容量控制和預處理

AlexNet透過暫退法（ :numref:`sec_dropout`）控制全連線層的模型複雜度，而LeNet只使用了權重衰減。
為了進一步擴充資料，AlexNet在訓練時增加了大量的影象增強資料，如翻轉、裁切和變色。
這使得模型更健壯，更大的樣本量有效地減少了過擬合。
在 :numref:`sec_image_augmentation`中更詳細地討論資料擴增。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()

net.add(
    # 這裡使用一個11*11的更大視窗來捕捉物件。
    # 同時，步幅為4，以減少輸出的高度和寬度。
    # 另外，輸出通道的數目遠大於LeNet
    nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 減小卷積視窗，使用填充為2來使得輸入與輸出的高和寬一致，且增大輸出通道數
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 使用三個連續的卷積層和一個較小的卷積視窗。
    # 除了最後的卷積層，輸出通道的數量進一步增加。
    # 前兩個卷積層後不使用彙聚層來減小輸入的高和寬
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 這裡，全連線層的輸出數量是LeNet中的好幾倍。使用dropout層來減輕過擬合
    nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
    nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
    # 最後是輸出層。由於這裡使用Fashion-MNIST，所以用類別數為10，而非論文中的1000
    nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # 這裡使用一個11*11的更大視窗來捕捉物件。
    # 同時，步幅為4，以減少輸出的高度和寬度。
    # 另外，輸出通道的數目遠大於LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 減小卷積視窗，使用填充為2來使得輸入與輸出的高和寬一致，且增大輸出通道數
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三個連續的卷積層和較小的卷積視窗。
    # 除了最後的卷積層，輸出通道的數量進一步增加。
    # 在前兩個卷積層之後，彙聚層不用於減少輸入的高度和寬度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 這裡，全連線層的輸出數量是LeNet中的好幾倍。使用dropout層來減輕過擬合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最後是輸出層。由於這裡使用Fashion-MNIST，所以用類別數為10，而非論文中的1000
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # 這裡使用一個11*11的更大視窗來捕捉物件。
        # 同時，步幅為4，以減少輸出的高度和寬度。
        # 另外，輸出通道的數目遠大於LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 減小卷積視窗，使用填充為2來使得輸入與輸出的高和寬一致，且增大輸出通道數
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 使用三個連續的卷積層和較小的卷積視窗。
        # 除了最後的卷積層，輸出通道的數量進一步增加。
        # 在前兩個卷積層之後，彙聚層不用於減少輸入的高度和寬度
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # 這裡，全連線層的輸出數量是LeNet中的好幾倍。使用dropout層來減輕過擬合
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # 最後是輸出層。由於這裡使用Fashion-MNIST，所以用類別數為10，而非論文中的1000
        tf.keras.layers.Dense(10)
    ])
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn

net = nn.Sequential(
    # 這裡，我們使用一個11*11的更大視窗來捕捉物件。
    # 同時，步幅為4，以減少輸出的高度和寬度。
    # 另外，輸出通道的數目遠大於LeNet
    nn.Conv2D(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2D(kernel_size=3, stride=2),
    # 減小卷積視窗，使用填充為2來使得輸入與輸出的高和寬一致，且增大輸出通道數
    nn.Conv2D(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2D(kernel_size=3, stride=2),
    # 使用三個連續的卷積層和較小的卷積視窗。
    # 除了最後的卷積層，輸出通道的數量進一步增加。
    # 在前兩個卷積層之後，池化層不用於減少輸入的高度和寬度
    nn.Conv2D(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2D(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2D(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2D(kernel_size=3, stride=2), nn.Flatten(),
    # 這裡，全連線層的輸出數量是LeNet中的好幾倍。使用dropout層來減輕過度擬合
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    # 最後是輸出層。由於這裡使用Fashion-MNIST，所以用類別數為10，而非論文中的1000
    nn.Linear(4096, 10)
)
```

[**我們構造一個**]高度和寬度都為224的(**單通道資料，來觀察每一層輸出的形狀**)。
它與 :numref:`fig_alexnet`中的AlexNet架構相對應。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab paddle
X = paddle.randn(shape=(1, 1, 224, 224))
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

## 讀取資料集

儘管原文中AlexNet是在ImageNet上進行訓練的，但本書在這裡使用的是Fashion-MNIST資料集。因為即使在現代GPU上，訓練ImageNet模型，同時使其收斂可能需要數小時或數天的時間。
將AlexNet直接應用於Fashion-MNIST的一個問題是，[**Fashion-MNIST影象的解析度**]（$28 \times 28$畫素）(**低於ImageNet影象。**)
為了解決這個問題，(**我們將它們增加到$224 \times 224$**)（通常來講這不是一個明智的做法，但在這裡這樣做是為了有效使用AlexNet架構）。
這裡需要使用`d2l.load_data_fashion_mnist`函式中的`resize`引數執行此調整。

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## [**訓練AlexNet**]

現在AlexNet可以開始被訓練了。與 :numref:`sec_lenet`中的LeNet相比，這裡的主要變化是使用更小的學習速率訓練，這是因為網路更深更廣、影象解析度更高，訓練卷積神經網路就更昂貴。

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* AlexNet的架構與LeNet相似，但使用了更多的卷積層和更多的引數來擬合大規模的ImageNet資料集。
* 今天，AlexNet已經被更有效的架構所超越，但它是從淺層網路到深層網路的關鍵一步。
* 儘管AlexNet的程式碼只比LeNet多出幾行，但學術界花了很多年才接受深度學習這一概念，並應用其出色的實驗結果。這也是由於缺乏有效的計算工具。
* Dropout、ReLU和預處理是提升電腦視覺任務效能的其他關鍵步驟。

## 練習

1. 試著增加迭代輪數。對比LeNet的結果有什麼不同？為什麼？
1. AlexNet對Fashion-MNIST資料集來說可能太複雜了。
    1. 嘗試簡化模型以加快訓練速度，同時確保準確性不會顯著下降。
    1. 設計一個更好的模型，可以直接在$28 \times 28$影象上工作。
1. 修改批次大小，並觀察模型精度和GPU顯存變化。
1. 分析了AlexNet的計算效能。
    1. 在AlexNet中主要是哪部分佔用顯存？
    1. 在AlexNet中主要是哪部分需要更多的計算？
    1. 計算結果時顯存頻寬如何？
1. 將dropout和ReLU應用於LeNet-5，效果有提升嗎？再試試預處理會怎麼樣？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1864)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1863)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1862)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11788)
:end_tab:
