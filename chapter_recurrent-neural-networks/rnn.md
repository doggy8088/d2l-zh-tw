# 迴圈神經網路
:label:`sec_rnn`

在 :numref:`sec_language_model`中，
我們介紹了$n$元語法模型，
其中單詞$x_t$在時間步$t$的條件機率僅取決於前面$n-1$個單詞。
對於時間步$t-(n-1)$之前的單詞，
如果我們想將其可能產生的影響合併到$x_t$上，
需要增加$n$，然而模型引數的數量也會隨之呈指數增長，
因為詞表$\mathcal{V}$需要儲存$|\mathcal{V}|^n$個數字，
因此與其將$P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$模型化，
不如使用隱變數模型：

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

其中$h_{t-1}$是*隱狀態*（hidden state），
也稱為*隱藏變數*（hidden variable），
它儲存了到時間步$t-1$的序列訊息。
通常，我們可以基於當前輸入$x_{t}$和先前隱狀態$h_{t-1}$
來計算時間步$t$處的任何時間的隱狀態：

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

對於 :eqref:`eq_ht_xt`中的函式$f$，隱變數模型不是近似值。
畢竟$h_t$是可以僅僅儲存到目前為止觀察到的所有資料，
然而這樣的操作可能會使計算和儲存的代價都變得昂貴。

回想一下，我們在 :numref:`chap_perceptrons`中
討論過的具有隱藏單元的隱藏層。
值得注意的是，隱藏層和隱狀態指的是兩個截然不同的概念。
如上所述，隱藏層是在從輸入到輸出的路徑上（以觀測角度來理解）的隱藏的層，
而隱狀態則是在給定步驟所做的任何事情（以技術角度來定義）的*輸入*，
並且這些狀態只能透過先前時間步的資料來計算。

*迴圈神經網路*（recurrent neural networks，RNNs）
是具有隱狀態的神經網路。
在介紹迴圈神經網路模型之前，
我們首先回顧 :numref:`sec_mlp`中介紹的多層感知機模型。

## 無隱狀態的神經網路

讓我們來看一看只有單隱藏層的多層感知機。
設隱藏層的啟用函式為$\phi$，
給定一個小批次樣本$\mathbf{X} \in \mathbb{R}^{n \times d}$，
其中批次大小為$n$，輸入維度為$d$，
則隱藏層的輸出$\mathbf{H} \in \mathbb{R}^{n \times h}$透過下式計算：

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

在 :eqref:`rnn_h_without_state`中，
我們擁有的隱藏層權重引數為$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$，
偏置引數為$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$，
以及隱藏單元的數目為$h$。
因此求和時可以應用廣播機制（見 :numref:`subsec_broadcasting`）。
接下來，將隱藏變數$\mathbf{H}$用作輸出層的輸入。
輸出層由下式給出：

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中，$\mathbf{O} \in \mathbb{R}^{n \times q}$是輸出變數，
$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$是權重引數，
$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是輸出層的偏置引數。
如果是分類問題，我們可以用$\text{softmax}(\mathbf{O})$
來計算輸出類別的機率分佈。

這完全類似於之前在 :numref:`sec_sequence`中解決的迴歸問題，
因此我們省略了細節。
無須多言，只要可以隨機選擇『特徵-標籤』對，
並且透過自動微分和隨機梯度下降能夠學習網路引數就可以了。

## 有隱狀態的迴圈神經網路
:label:`subsec_rnn_w_hidden_states`

有了隱狀態後，情況就完全不同了。
假設我們在時間步$t$有小批次輸入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$。
換言之，對於$n$個序列樣本的小批次，
$\mathbf{X}_t$的每一行對應於來自該序列的時間步$t$處的一個樣本。
接下來，用$\mathbf{H}_t  \in \mathbb{R}^{n \times h}$
表示時間步$t$的隱藏變數。
與多層感知機不同的是，
我們在這裡儲存了前一個時間步的隱藏變數$\mathbf{H}_{t-1}$，
並引入了一個新的權重引數$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$，
來描述如何在當前時間步中使用前一個時間步的隱藏變數。
具體地說，當前時間步隱藏變數由當前時間步的輸入
與前一個時間步的隱藏變數一起計算得出：

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

與 :eqref:`rnn_h_without_state`相比，
 :eqref:`rnn_h_with_state`多新增了一項
$\mathbf{H}_{t-1} \mathbf{W}_{hh}$，
從而實例化了 :eqref:`eq_ht_xt`。
從相鄰時間步的隱藏變數$\mathbf{H}_t$和
$\mathbf{H}_{t-1}$之間的關係可知，
這些變數捕獲並保留了序列直到其當前時間步的歷史訊息，
就如當前時間步下神經網路的狀態或記憶，
因此這樣的隱藏變數被稱為*隱狀態*（hidden state）。
由於在當前時間步中，
隱狀態使用的定義與前一個時間步中使用的定義相同，
因此 :eqref:`rnn_h_with_state`的計算是*迴圈的*（recurrent）。
於是基於迴圈計算的隱狀態神經網路被命名為
*迴圈神經網路*（recurrent neural network）。
在迴圈神經網路中執行 :eqref:`rnn_h_with_state`計算的層
稱為*迴圈層*（recurrent layer）。

有許多不同的方法可以建立迴圈神經網路，
由 :eqref:`rnn_h_with_state`定義的隱狀態的迴圈神經網路是非常常見的一種。
對於時間步$t$，輸出層的輸出類似於多層感知機中的計算：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

迴圈神經網路的引數包括隱藏層的權重
$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和偏置$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$，
以及輸出層的權重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$
和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$。
值得一提的是，即使在不同的時間步，迴圈神經網路也總是使用這些模型引數。
因此，迴圈神經網路的引數開銷不會隨著時間步的增加而增加。

 :numref:`fig_rnn`展示了迴圈神經網路在三個相鄰時間步的計算邏輯。
在任意時間步$t$，隱狀態的計算可以被視為：

1. 拼接當前時間步$t$的輸入$\mathbf{X}_t$和前一時間步$t-1$的隱狀態$\mathbf{H}_{t-1}$；
1. 將拼接的結果送入帶有啟用函式$\phi$的全連線層。
   全連線層的輸出是當前時間步$t$的隱狀態$\mathbf{H}_t$。
   
在本例中，模型引數是$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的拼接，
以及$\mathbf{b}_h$的偏置，所有這些引數都來自 :eqref:`rnn_h_with_state`。
當前時間步$t$的隱狀態$\mathbf{H}_t$
將參與計算下一時間步$t+1$的隱狀態$\mathbf{H}_{t+1}$。
而且$\mathbf{H}_t$還將送入全連線輸出層，
用於計算當前時間步$t$的輸出$\mathbf{O}_t$。

![具有隱狀態的迴圈神經網路](../img/rnn.svg)
:label:`fig_rnn`

我們剛才提到，隱狀態中
$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$的計算，
相當於$\mathbf{X}_t$和$\mathbf{H}_{t-1}$的拼接
與$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的拼接的矩陣乘法。
雖然這個性質可以透過數學證明，
但在下面我們使用一個簡單的程式碼來說明一下。
首先，我們定義矩陣`X`、`W_xh`、`H`和`W_hh`，
它們的形狀分別為$(3，1)$、$(1，4)$、$(3，4)$和$(4，4)$。
分別將`X`乘以`W_xh`，將`H`乘以`W_hh`，
然後將這兩個乘法相加，我們得到一個形狀為$(3，4)$的矩陣。

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

```{.python .input}
#@tab mxnet, pytorch, paddle
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

現在，我們沿列（軸1）拼接矩陣`X`和`H`，
沿行（軸0）拼接矩陣`W_xh`和`W_hh`。
這兩個拼接分別產生形狀$(3, 5)$和形狀$(5, 4)$的矩陣。
再將這兩個拼接的矩陣相乘，
我們得到與上面相同形狀$(3, 4)$的輸出矩陣。

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## 基於迴圈神經網路的字元級語言模型

回想一下 :numref:`sec_language_model`中的語言模型，
我們的目標是根據過去的和當前的詞元預測下一個詞元，
因此我們將原始序列移位一個詞元作為標籤。
Bengio等人首先提出使用神經網路進行語言建模
 :cite:`Bengio.Ducharme.Vincent.ea.2003`。
接下來，我們看一下如何使用迴圈神經網路來建立語言模型。
設小批次大小為1，批次中的文字序列為『machine』。
為了簡化後續部分的訓練，我們考慮使用
*字元級語言模型*（character-level language model），
將文字詞元化為字元而不是單詞。
 :numref:`fig_rnn_train`示範了
如何透過基於字元級語言建模的迴圈神經網路，
使用當前的和先前的字元預測下一個字元。

![基於迴圈神經網路的字元級語言模型：輸入序列和標籤序列分別為『machin』和『achine』](../img/rnn-train.svg)
:label:`fig_rnn_train`

在訓練過程中，我們對每個時間步的輸出層的輸出進行softmax操作，
然後利用交叉熵損失計算模型輸出和標籤之間的誤差。
由於隱藏層中隱狀態的迴圈計算，
 :numref:`fig_rnn_train`中的第$3$個時間步的輸出$\mathbf{O}_3$
由文字序列『m』『a』和『c』確定。
由於訓練資料中這個文字序列的下一個字元是『h』，
因此第$3$個時間步的損失將取決於下一個字元的機率分佈，
而下一個字元是基於特徵序列『m』『a』『c』和這個時間步的標籤『h』生成的。

在實踐中，我們使用的批次大小為$n>1$，
每個詞元都由一個$d$維向量表示。
因此，在時間步$t$輸入$\mathbf X_t$將是一個$n\times d$矩陣，
這與我們在 :numref:`subsec_rnn_w_hidden_states`中的討論相同。

## 困惑度（Perplexity）
:label:`subsec_perplexity`

最後，讓我們討論如何度量語言模型的質量，
這將在後續部分中用於評估基於迴圈神經網路的模型。
一個好的語言模型能夠用高度準確的詞元來預測我們接下來會看到什麼。
考慮一下由不同的語言模型給出的對『It is raining ...』（『...下雨了』）的續寫：

1. "It is raining outside"（外面下雨了）；
1. "It is raining banana tree"（香蕉樹下雨了）；
1. "It is raining piouw;kcj pwepoiut"（piouw;kcj pwepoiut下雨了）。

就質量而言，例$1$顯然是最合乎情理、在邏輯上最連貫的。
雖然這個模型可能沒有很準確地反映出後續詞的語義，
比如，『It is raining in San Francisco』（舊金山下雨了）
和『It is raining in winter』（冬天下雨了）
可能才是更完美的合理擴充套件，
但該模型已經能夠捕捉到跟在後麵的是哪類單詞。
例$2$則要糟糕得多，因為其產生了一個無意義的續寫。
儘管如此，至少該模型已經學會瞭如何拼寫單詞，
以及單詞之間的某種程度的相關性。
最後，例$3$表明了訓練不足的模型是無法正確地擬合數據的。

我們可以透過計算序列的似然機率來度量模型的質量。
然而這是一個難以理解、難以比較的數字。
畢竟，較短的序列比較長的序列更有可能出現，
因此評估模型產生托爾斯泰的巨著《戰爭與和平》的可能性
不可避免地會比產生聖埃克蘇佩裡的中篇小說《小王子》可能性要小得多。
而缺少的可能性值相當於平均數。

在這裡，資訊理論可以派上用場了。
我們在引入softmax迴歸
（ :numref:`subsec_info_theory_basics`）時定義了熵、驚異和交叉熵，
並在[資訊理論的線上附錄](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)
中討論了更多的資訊理論知識。
如果想要壓縮文字，我們可以根據當前詞元集預測的下一個詞元。
一個更好的語言模型應該能讓我們更準確地預測下一個詞元。
因此，它應該允許我們在壓縮序列時花費更少的比特。
所以我們可以透過一個序列中所有的$n$個詞元的交叉熵損失的平均值來衡量：

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

其中$P$由語言模型給出，
$x_t$是在時間步$t$從該序列中觀察到的實際詞元。
這使得不同長度的文件的效能具有了可比性。
由於歷史原因，自然語言處理的科學家更喜歡使用一個叫做*困惑度*（perplexity）的量。
簡而言之，它是 :eqref:`eq_avg_ce_for_lm`的指數：

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

困惑度的最好的理解是『下一個詞元的實際選擇數的調和平均數』。
我們看看一些案例。

* 在最好的情況下，模型總是完美地估計標籤詞元的機率為1。
  在這種情況下，模型的困惑度為1。
* 在最壞的情況下，模型總是預測標籤詞元的機率為0。
  在這種情況下，困惑度是正無窮大。
* 在基線上，該模型的預測是詞表的所有可用詞元上的均勻分佈。
  在這種情況下，困惑度等於詞表中唯一詞元的數量。
  事實上，如果我們在沒有任何壓縮的情況下儲存序列，
  這將是我們能做的最好的編碼方式。
  因此，這種方式提供了一個重要的上限，
  而任何實際模型都必須超越這個上限。

在接下來的小節中，我們將基於迴圈神經網路實現字元級語言模型，
並使用困惑度來評估這樣的模型。

## 小結

* 對隱狀態使用迴圈計算的神經網路稱為迴圈神經網路（RNN）。
* 迴圈神經網路的隱狀態可以捕獲直到當前時間步序列的歷史訊息。
* 迴圈神經網路模型的引數數量不會隨著時間步的增加而增加。
* 我們可以使用迴圈神經網路建立字元級語言模型。
* 我們可以使用困惑度來評價語言模型的質量。

## 練習

1. 如果我們使用迴圈神經網路來預測文字序列中的下一個字元，那麼任意輸出所需的維度是多少？
1. 為什麼迴圈神經網路可以基於文字序列中所有先前的詞元，在某個時間步表示當前詞元的條件機率？
1. 如果基於一個長序列進行反向傳播，梯度會發生什麼狀況？
1. 與本節中描述的語言模型相關的問題有哪些？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2099)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2101)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11798)
:end_tab:
