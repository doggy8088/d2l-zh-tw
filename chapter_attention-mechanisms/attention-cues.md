# 注意力提示
:label:`sec_attention-cues`

感謝讀者對本書的關注，因為讀者的注意力是一種稀缺的資源：
此刻讀者正在閱讀本書（而忽略了其他的書），
因此讀者的注意力是用機會成本（與金錢類似）來支付的。
為了確保讀者現在投入的注意力是值得的，
作者們盡全力（全部的注意力）創作一本好書。

自經濟學研究稀缺資源分配以來，人們正處在『注意力經濟』時代，
即人類的注意力被視為可以交換的、有限的、有價值的且稀缺的商品。
許多商業模式也被開發出來去利用這一點：
在音樂或視頻流媒體服務上，人們要麼消耗注意力在廣告上，要麼付錢來隱藏廣告；
為了在網絡遊戲世界的成長，人們要麼消耗注意力在遊戲戰鬥中，
從而幫助吸引新的玩家，要麼付錢立即變得強大。
總之，注意力不是免費的。

注意力是稀缺的，而環境中的干擾注意力的訊息卻並不少。
比如人類的視覺神經系統大約每秒收到$10^8$位的訊息，
這遠遠超過了大腦能夠完全處理的水平。
幸運的是，人類的祖先已經從經驗（也稱為資料）中認識到
『並非感官的所有輸入都是一樣的』。
在整個人類歷史中，這種只將注意力引向感興趣的一小部分訊息的能力，
使人類的大腦能夠更明智地分配資源來生存、成長和社交，
例如發現天敵、找尋食物和伴侶。

## 生物學中的注意力提示

注意力是如何應用於視覺世界中的呢？
這要從當今十分普及的*雙元件*（two-component）的框架開始講起：
這個框架的出現可以追溯到19世紀90年代的威廉·詹姆斯，
他被認為是『美國心理學之父』 :cite:`James.2007`。
在這個框架中，受試者基於*非自主性提示*和*自主性提示*
有選擇地引導注意力的焦點。

非自主性提示是基於環境中物體的突出性和易見性。
想象一下，假如我們面前有五個物品：
一份報紙、一篇研究論文、一杯咖啡、一本筆記本和一本書，
就像 :numref:`fig_eye-coffee`。
所有紙製品都是黑白印刷的，但咖啡杯是紅色的。
換句話說，這個咖啡杯在這種視覺環境中是突出和顯眼的，
不由自主地引起人們的注意。
所以我們會把視力最敏銳的地方放到咖啡上，
如 :numref:`fig_eye-coffee`所示。

![由於突出性的非自主性提示（紅杯子），注意力不自主地指向了咖啡杯](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

喝咖啡後，我們會變得興奮並想讀書，
所以轉過頭，重新聚焦眼睛，然後看看書，
就像 :numref:`fig_eye-book`中描述那樣。
與 :numref:`fig_eye-coffee`中由於突出性導致的選擇不同，
此時選擇書是受到了認知和意識的控制，
因此注意力在基於自主性提示去輔助選擇時將更為謹慎。
受試者的主觀意願推動，選擇的力量也就更強大。

![依賴於任務的意志提示（想讀一本書），注意力被自主引導到書上](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`

## 查詢、鍵和值

自主性的與非自主性的注意力提示解釋了人類的注意力的方式，
下面來看看如何透過這兩種注意力提示，
用神經網路來設計注意力機制的框架，

首先，考慮一個相對簡單的狀況，
即只使用非自主性提示。
要想將選擇偏向於感官輸入，
則可以簡單地使用引數化的全連線層，
甚至是非引數化的最大彙聚層或平均彙聚層。

因此，『是否包含自主性提示』將注意力機制與全連線層或彙聚層區別開來。
在注意力機制的背景下，自主性提示被稱為*查詢*（query）。
給定任何查詢，注意力機制透過*注意力彙聚*（attention pooling）
將選擇引導至*感官輸入*（sensory inputs，例如中間特徵表示）。
在注意力機制中，這些感官輸入被稱為*值*（value）。
更通俗的解釋，每個值都與一個*鍵*（key）配對，
這可以想象為感官輸入的非自主提示。
如 :numref:`fig_qkv`所示，可以透過設計注意力彙聚的方式，
便於給定的查詢（自主性提示）與鍵（非自主性提示）進行對應，
這將引導得出最對應的值（感官輸入）。

![注意力機制透過注意力彙聚將*查詢*（自主性提示）和*鍵*（非自主性提示）結合在一起，實現對*值*（感官輸入）的選擇傾向](../img/qkv.svg)
:label:`fig_qkv`

鑑於上面所提框架在 :numref:`fig_qkv`中的主導地位，
因此這個框架下的模型將成為本章的中心。
然而，注意力機制的設計有許多替代方案。
例如可以設計一個不可微的注意力模型，
該模型可以使用強化學習方法 :cite:`Mnih.Heess.Graves.ea.2014`進行訓練。

## 注意力的視覺化

平均彙聚層可以被視為輸入的加權平均值，
其中各輸入的權重是一樣的。
實際上，注意力彙聚得到的是加權平均的總和值，
其中權重是在給定的查詢和不同的鍵之間計算得出的。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
```

為了視覺化注意力權重，需要定義一個`show_heatmaps`函式。
其輸入`matrices`的形狀是
（要顯示的行數，要顯示的列數，查詢的數目，鍵的數目）。

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """顯示矩陣熱圖"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

下面使用一個簡單的例子進行示範。
在本例子中，僅當查詢和鍵相同時，注意力權重為1，否則為0。

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

後面的章節內容將經常呼叫`show_heatmaps`函式來顯示注意力權重。

## 小結

* 人類的注意力是有限的、有價值和稀缺的資源。
* 受試者使用非自主性和自主性提示有選擇性地引導注意力。前者基於突出性，後者則依賴於意識。
* 注意力機制與全連線層或者彙聚層的區別源於增加的自主提示。
* 由於包含了自主性提示，注意力機制與全連線的層或彙聚層不同。
* 注意力機制透過注意力彙聚使選擇偏向於值（感官輸入），其中包含查詢（自主性提示）和鍵（非自主性提示）。鍵和值是成對的。
* 視覺化查詢和鍵之間的注意力權重是可行的。

## 練習

1. 在機器翻譯中透過解碼序列詞元時，其自主性提示可能是什麼？非自主性提示和感官輸入又是什麼？
1. 隨機生成一個$10 \times 10$矩陣並使用`softmax`運算來確保每行都是有效的機率分佈，然後視覺化輸出注意力權重。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5763)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5764)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/5765)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11839)
:end_tab: