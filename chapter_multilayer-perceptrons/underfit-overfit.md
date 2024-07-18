# 模型選擇、欠擬合和過擬合
:label:`sec_model_selection`

作為機器學習科學家，我們的目標是發現*模式*（pattern）。
但是，我們如何才能確定模型是真正發現了一種泛化的模式，
而不是簡單地記住了資料呢？
例如，我們想要在患者的基因數據與痴呆狀態之間尋找模式，
其中標籤是從集合$\{\text{痴呆}, \text{輕度認知障礙}, \text{健康}\}$中提取的。
因為基因可以唯一確定每個個體（不考慮雙胞胎），
所以在這個任務中是有可能記住整個資料集的。

我們不想讓模型只會做這樣的事情：『那是鮑勃！我記得他！他有痴呆症！』。
原因很簡單：當我們將來部署該模型時，模型需要判斷從未見過的患者。
只有當模型真正發現了一種泛化模式時，才會作出有效的預測。

更正式地說，我們的目標是發現某些模式，
這些模式捕捉到了我們訓練集潛在總體的規律。
如果成功做到了這點，即使是對以前從未遇到過的個體，
模型也可以成功地評估風險。
如何發現可以泛化的模式是機器學習的根本問題。

困難在於，當我們訓練模型時，我們只能訪問資料中的小部分樣本。
最大的公開影象資料集包含大約一百萬張影象。
而在大部分時候，我們只能從數千或數萬個數據樣本中學習。
在大型醫院系統中，我們可能會訪問數十萬份醫療記錄。
當我們使用有限的樣本時，可能會遇到這樣的問題：
當收集到更多的資料時，會發現之前找到的明顯關係並不成立。

將模型在訓練資料上擬合的比在潛在分佈中更接近的現象稱為*過擬合*（overfitting），
用於對抗過擬合的技術稱為*正則化*（regularization）。
在前面的章節中，有些讀者可能在用Fashion-MNIST資料集做實驗時已經觀察到了這種過擬合現象。
在實驗中調整模型架構或超引數時會發現：
如果有足夠多的神經元、層數和訓練迭代週期，
模型最終可以在訓練集上達到完美的精度，此時測試集的準確性卻下降了。

## 訓練誤差和泛化誤差

為了進一步討論這一現象，我們需要瞭解訓練誤差和泛化誤差。
*訓練誤差*（training error）是指，
模型在訓練資料集上計算得到的誤差。
*泛化誤差*（generalization error）是指，
模型應用在同樣從原始樣本的分佈中抽取的無限多數據樣本時，模型誤差的期望。

問題是，我們永遠不能準確地計算出泛化誤差。
這是因為無限多的資料樣本是一個虛構的物件。
在實際中，我們只能透過將模型應用於一個獨立的測試集來估計泛化誤差，
該測試集由隨機選取的、未曾在訓練集中出現的資料樣本構成。

下面的三個思維實驗將有助於更好地說明這種情況。
假設一個大學生正在努力準備期末考試。
一個勤奮的學生會努力做好練習，並利用往年的考試題目來測試自己的能力。
儘管如此，在過去的考試題目上取得好成績並不能保證他會在真正考試時發揮出色。
例如，學生可能試圖透過死記硬背考題的答案來做準備。
他甚至可以完全記住過去考試的答案。
另一名學生可能會通過試圖理解給出某些答案的原因來做準備。
在大多數情況下，後者會考得更好。

類似地，考慮一個簡單地使用查表法來回答問題的模型。
如果允許的輸入集合是離散的並且相當小，
那麼也許在檢視許多訓練樣本後，該方法將執行得很好。
但當這個模型面對從未見過的例子時，它表現的可能比隨機猜測好不到哪去。
這是因為輸入空間太大了，遠遠不可能記住每一個可能的輸入所對應的答案。
例如，考慮$28\times28$的灰度影象。
如果每個畫素可以取$256$個灰度值中的一個，
則有$256^{784}$個可能的影象。
這意味著指甲大小的低解析度灰度影象的數量比宇宙中的原子要多得多。
即使我們可能遇到這樣的資料，我們也不可能儲存整個查詢表。

最後，考慮對擲硬幣的結果（類別0：正面，類別1：反面）進行分類的問題。
假設硬幣是公平的，無論我們想出什麼演算法，泛化誤差始終是$\frac{1}{2}$。
然而，對於大多數演算法，我們應該期望訓練誤差會更低（取決於運氣）。
考慮資料集{0，1，1，1，0，1}。
我們的演算法不需要額外的特徵，將傾向於總是預測*多數類*，
從我們有限的樣本來看，它似乎是1佔主流。
在這種情況下，總是預測類1的模型將產生$\frac{1}{3}$的誤差，
這比我們的泛化誤差要好得多。
當我們逐漸增加資料量，正面比例明顯偏離$\frac{1}{2}$的可能性將會降低，
我們的訓練誤差將與泛化誤差相對應。

### 統計學習理論

由於泛化是機器學習中的基本問題，
許多數學家和理論家畢生致力於研究描述這一現象的形式理論。
在[同名定理（eponymous theorem）](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem)中，
格里文科和坎特利推導出了訓練誤差收斂到泛化誤差的速率。
在一系列開創性的論文中，
[Vapnik和Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory)
將這一理論擴充套件到更一般種類的函式。
這項工作為統計學習理論奠定了基礎。

在我們目前已探討、並將在之後繼續探討的監督學習情景中，
我們假設訓練資料和測試資料都是從相同的分佈中獨立提取的。
這通常被稱為*獨立同分布假設*（i.i.d. assumption），
這意味著對資料進行取樣的過程沒有進行『記憶』。
換句話說，抽取的第2個樣本和第3個樣本的相關性，
並不比抽取的第2個樣本和第200萬個樣本的相關性更強。

要成為一名優秀的機器學習科學家需要具備批判性思考能力。
假設是存在漏洞的，即很容易找出假設失效的情況。
如果我們根據從加州大學舊金山分校醫學中心的患者資料訓練死亡風險預測模型，
並將其應用於麻薩諸塞州綜合醫院的患者資料，結果會怎麼樣？
這兩個資料的分佈可能不完全一樣。
此外，抽樣過程可能與時間有關。
比如當我們對微博的主題進行分類時，
新聞週期會使得正在討論的話題產生時間依賴性，從而違反獨立性假設。

有時候我們即使輕微違背獨立同分布假設，模型仍將繼續執行得非常好。
比如，我們有許多有用的工具已經應用於現實，如人臉識別、語音識別和語言翻譯。
畢竟，幾乎所有現實的應用都至少涉及到一些違背獨立同分布假設的情況。

有些違背獨立同分布假設的行為肯定會帶來麻煩。
比如，我們試圖只用來自大學生的人臉資料來訓練一個人臉識別系統，
然後想要用它來監測療養院中的老人。
這不太可能有效，因為大學生看起來往往與老年人有很大的不同。

在接下來的章節中，我們將討論因違背獨立同分布假設而引起的問題。
目前，即使認為獨立同分布假設是理所當然的，理解泛化性也是一個困難的問題。
此外，能夠解釋深層神經網路泛化效能的理論基礎，
也仍在繼續困擾著學習理論領域最偉大的學者們。

當我們訓練模型時，我們試圖找到一個能夠儘可能擬合訓練資料的函式。
但是如果它執行地『太好了』，而不能對看不見的資料做到很好泛化，就會導致過擬合。
這種情況正是我們想要避免或控制的。
深度學習中有許多啟發式的技術旨在防止過擬合。

### 模型複雜性

當我們有簡單的模型和大量的資料時，我們期望泛化誤差與訓練誤差相近。
當我們有更復雜的模型和更少的樣本時，我們預計訓練誤差會下降，但泛化誤差會增大。
模型複雜性由什麼構成是一個複雜的問題。
一個模型是否能很好地泛化取決於很多因素。
例如，具有更多引數的模型可能被認為更復雜，
引數有更大取值範圍的模型可能更為複雜。
通常對於神經網路，我們認為需要更多訓練迭代的模型比較複雜，
而需要*早停*（early stopping）的模型（即較少訓練迭代週期）就不那麼複雜。

我們很難比較本質上不同大類別的模型之間（例如，決策樹與神經網路）的複雜性。
就目前而言，一條簡單的經驗法則相當有用：
統計學家認為，能夠輕鬆解釋任意事實的模型是複雜的，
而表達能力有限但仍能很好地解釋資料的模型可能更有現實用途。
在哲學上，這與波普爾的科學理論的可證偽性標準密切相關：
如果一個理論能擬合數據，且有具體的測試可以用來證明它是錯誤的，那麼它就是好的。
這一點很重要，因為所有的統計估計都是*事後歸納*。
也就是說，我們在觀察事實之後進行估計，因此容易受到相關謬誤的影響。
目前，我們將把哲學放在一邊，堅持更切實的問題。

本節為了給出一些直觀的印象，我們將重點介紹幾個傾向於影響模型泛化的因素。

1. 可調整引數的數量。當可調整引數的數量（有時稱為*自由度*）很大時，模型往往更容易過擬合。
1. 引數採用的值。當權重的取值範圍較大時，模型可能更容易過擬合。
1. 訓練樣本的數量。即使模型很簡單，也很容易過擬合只包含一兩個樣本的資料集。而過擬合一個有數百萬個樣本的資料集則需要一個極其靈活的模型。

## 模型選擇

在機器學習中，我們通常在評估幾個候選模型後選擇最終的模型。
這個過程叫做*模型選擇*。
有時，需要進行比較的模型在本質上是完全不同的（比如，決策樹與線性模型）。
又有時，我們需要比較不同的超引數設定下的同一類模型。

例如，訓練多層感知機模型時，我們可能希望比較具有
不同數量的隱藏層、不同數量的隱藏單元以及不同的啟用函式組合的模型。
為了確定候選模型中的最佳模型，我們通常會使用驗證集。

### 驗證集

原則上，在我們確定所有的超引數之前，我們不希望用到測試集。
如果我們在模型選擇過程中使用測試資料，可能會有過擬合測試資料的風險，那就麻煩大了。
如果我們過擬合了訓練資料，還可以在測試資料上的評估來判斷過擬合。
但是如果我們過擬合了測試資料，我們又該怎麼知道呢？

因此，我們決不能依靠測試資料進行模型選擇。
然而，我們也不能僅僅依靠訓練資料來選擇模型，因為我們無法估計訓練資料的泛化誤差。

在實際應用中，情況變得更加複雜。
雖然理想情況下我們只會使用測試資料一次，
以評估最好的模型或比較一些模型效果，但現實是測試資料很少在使用一次後被丟棄。
我們很少能有充足的資料來對每一輪實驗採用全新測試集。

解決此問題的常見做法是將我們的資料分成三份，
除了訓練和測試資料集之外，還增加一個*驗證資料集*（validation dataset），
也叫*驗證集*（validation set）。
但現實是驗證資料和測試資料之間的邊界模糊得令人擔憂。
除非另有明確說明，否則在這本書的實驗中，
我們實際上是在使用應該被正確地稱為訓練資料和驗證資料的資料集，
並沒有真正的測試資料集。
因此，書中每次實驗報告的準確度都是驗證集準確度，而不是測試集準確度。

### $K$折交叉驗證

當訓練資料稀缺時，我們甚至可能無法提供足夠的資料來構成一個合適的驗證集。
這個問題的一個流行的解決方案是採用$K$*折交叉驗證*。
這裡，原始訓練資料被分成$K$個不重疊的子集。
然後執行$K$次模型訓練和驗證，每次在$K-1$個子集上進行訓練，
並在剩餘的一個子集（在該輪中沒有用於訓練的子集）上進行驗證。
最後，透過對$K$次實驗的結果取平均來估計訓練和驗證誤差。

## 欠擬合還是過擬合？

當我們比較訓練和驗證誤差時，我們要注意兩種常見的情況。
首先，我們要注意這樣的情況：訓練誤差和驗證誤差都很嚴重，
但它們之間僅有一點差距。
如果模型不能降低訓練誤差，這可能意味著模型過於簡單（即表達能力不足），
無法捕獲試圖學習的模式。
此外，由於我們的訓練和驗證誤差之間的*泛化誤差*很小，
我們有理由相信可以用一個更復雜的模型降低訓練誤差。
這種現象被稱為*欠擬合*（underfitting）。

另一方面，當我們的訓練誤差明顯低於驗證誤差時要小心，
這表明嚴重的*過擬合*（overfitting）。
注意，*過擬合*並不總是一件壞事。
特別是在深度學習領域，眾所周知，
最好的預測模型在訓練資料上的表現往往比在保留（驗證）資料上好得多。
最終，我們通常更關心驗證誤差，而不是訓練誤差和驗證誤差之間的差距。

是否過擬合或欠擬合可能取決於模型複雜性和可用訓練資料集的大小，
這兩個點將在下面進行討論。

### 模型複雜性

為了說明一些關於過擬合和模型複雜性的經典直覺，
我們給出一個多項式的例子。
給定由單個特徵$x$和對應實數標籤$y$組成的訓練資料，
我們試圖找到下面的$d$階多項式來估計標籤$y$。

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

這只是一個線性迴歸問題，我們的特徵是$x$的冪給出的，
模型的權重是$w_i$給出的，偏置是$w_0$給出的
（因為對於所有的$x$都有$x^0 = 1$）。
由於這只是一個線性迴歸問題，我們可以使用平方誤差作為我們的損失函式。

高階多項式函式比低階多項式函式複雜得多。
高階多項式的引數較多，模型函式的選擇範圍較廣。
因此在固定訓練資料集的情況下，
高階多項式函式相對於低階多項式的訓練誤差應該始終更低（最壞也是相等）。
事實上，當資料樣本包含了$x$的不同值時，
函式階數等於資料樣本數量的多項式函式可以完美擬合訓練集。
在 :numref:`fig_capacity_vs_error`中，
我們直觀地描述了多項式的階數和欠擬合與過擬合之間的關係。


![模型複雜度對欠擬合和過擬合的影響](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`

### 資料集大小

另一個重要因素是資料集的大小。
訓練資料集中的樣本越少，我們就越有可能（且更嚴重地）過擬合。
隨著訓練資料量的增加，泛化誤差通常會減小。
此外，一般來說，更多的資料不會有什麼壞處。
對於固定的任務和數據分佈，模型複雜性和數據集大小之間通常存在關係。
給出更多的資料，我們可能會嘗試擬合一個更復雜的模型。
能夠擬合更復雜的模型可能是有益的。
如果沒有足夠的資料，簡單的模型可能更有用。
對於許多任務，深度學習只有在有數千個訓練樣本時才優於線性模型。
從一定程度上來說，深度學習目前的生機要歸功於
廉價儲存、互聯裝置以及數字化經濟帶來的海量資料集。

## 多項式迴歸

我們現在可以(**透過多項式擬合來探索這些概念**)。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import math
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import numpy as np
import math
```

### 生成資料集

給定$x$，我們將[**使用以下三階多項式來生成訓練和測試資料的標籤：**]

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

噪聲項$\epsilon$服從均值為0且標準差為0.1的常態分佈。
在最佳化的過程中，我們通常希望避免非常大的梯度值或損失值。
這就是我們將特徵從$x^i$調整為$\frac{x^i}{i!}$的原因，
這樣可以避免很大的$i$帶來的特別大的指數值。
我們將為訓練集和測試集各生成100個樣本。

```{.python .input}
#@tab all
max_degree = 20  # 多項式的最大階數
n_train, n_test = 100, 100  # 訓練和測試資料集大小
true_w = np.zeros(max_degree)  # 分配大量的空間
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的維度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

同樣，儲存在`poly_features`中的單項式由gamma函式重新縮放，
其中$\Gamma(n)=(n-1)!$。
從生成的資料集中[**檢視一下前2個樣本**]，
第一個值是與偏置相對應的常數特徵。

```{.python .input}
#@tab pytorch, tensorflow, paddle
# NumPy ndarray轉換為tensor
true_w, features, poly_features, labels = [d2l.tensor(x, dtype=
    d2l.float32) for x in [true_w, features, poly_features, labels]]
```

```{.python .input}
#@tab all
features[:2], poly_features[:2, :], labels[:2]
```

### 對模型進行訓練和測試

首先讓我們[**實現一個函式來評估模型在給定資料集上的損失**]。

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_loss(net, data_iter, loss):  #@save
    """評估給定資料集上模型的損失"""
    metric = d2l.Accumulator(2)  # 損失的總和,樣本數量
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_loss(net, data_iter, loss):  #@save
    """評估給定資料集上模型的損失"""
    metric = d2l.Accumulator(2)  # 損失的總和,樣本數量
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab paddle
def evaluate_loss(net, data_iter, loss):  #@save
    """評估給定資料集上模型的損失。"""
    metric = d2l.Accumulator(2)  # 損失的總和, 樣本數量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
```

現在[**定義訓練函式**]。

```{.python .input}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # 不設定偏置，因為我們已經在多項式中實現了它
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

```{.python .input}
#@tab pytorch
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不設定偏置，因為我們已經在多項式中實現了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```{.python .input}
#@tab tensorflow
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # 不設定偏置，因為我們已經在多項式中實現了它
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net.get_weights()[0].T)
```

```{.python .input}
#@tab paddle
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # 不設定偏置，因為我們已經在多項式特徵中實現了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias_attr=False))
    batch_size = min(10, train_labels.shape[0])
    print(batch_size)
    train_iter = d2l.load_array(((train_features, train_labels.reshape([-1,1]))),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape([-1,1])),
                               batch_size, is_train=False)
    trainer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.numpy())
```

### [**三階多項式函式擬合(正常)**]

我們將首先使用三階多項式函式，它與資料生成函式的階數相同。
結果表明，該模型能有效降低訓練損失和測試損失。
學習到的模型引數也接近真實值$w = [5, 1.2, -3.4, 5.6]$。

```{.python .input}
#@tab all
# 從多項式特徵中選擇前4個維度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### [**線性函式擬合(欠擬合)**]

讓我們再看看線性函式擬合，減少該模型的訓練損失相對困難。
在最後一個迭代週期完成後，訓練損失仍然很高。
當用來擬合非線性模式（如這裡的三階多項式函式）時，線性模型容易欠擬合。

```{.python .input}
#@tab all
# 從多項式特徵中選擇前2個維度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### [**高階多項式函式擬合(過擬合)**]

現在，讓我們嘗試使用一個階數過高的多項式來訓練模型。
在這種情況下，沒有足夠的資料用於學到高階係數應該具有接近於零的值。
因此，這個過於複雜的模型會輕易受到訓練資料中噪聲的影響。
雖然訓練損失可以有效地降低，但測試損失仍然很高。
結果表明，複雜模型對資料造成了過擬合。

```{.python .input}
#@tab all
# 從多項式特徵中選取所有維度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

在接下來的章節中，我們將繼續討論過擬合問題和處理這些問題的方法，例如權重衰減和dropout。

## 小結

* 欠擬合是指模型無法繼續減少訓練誤差。過擬合是指訓練誤差遠小於驗證誤差。
* 由於不能基於訓練誤差來估計泛化誤差，因此簡單地最小化訓練誤差並不一定意味著泛化誤差的減小。機器學習模型需要注意防止過擬合，即防止泛化誤差過大。
* 驗證集可以用於模型選擇，但不能過於隨意地使用它。
* 我們應該選擇一個複雜度適當的模型，避免使用數量不足的訓練樣本。

## 練習

1. 這個多項式迴歸問題可以準確地解出嗎？提示：使用線性代數。
1. 考慮多項式的模型選擇。
    1. 繪製訓練損失與模型複雜度（多項式的階數）的關係圖。觀察到了什麼？需要多少階的多項式才能將訓練損失減少到0?
    1. 在這種情況下繪製測試的損失圖。
    1. 生成同樣的圖，作為資料量的函式。
1. 如果不對多項式特徵$x^i$進行標準化($1/i!$)，會發生什麼事情？能用其他方法解決這個問題嗎？
1. 泛化誤差可能為零嗎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1807)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1806)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1805)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11771)
:end_tab:
