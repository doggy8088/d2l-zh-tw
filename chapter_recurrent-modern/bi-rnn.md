# 雙向迴圈神經網路
:label:`sec_bi_rnn`

在序列學習中，我們以往假設的目標是：
在給定觀測的情況下
（例如，在時間序列的上下文中或在語言模型的上下文中），
對下一個輸出進行建模。
雖然這是一個典型情景，但不是唯一的。
還可能發生什麼其它的情況呢？
我們考慮以下三個在文字序列中填空的任務。

* 我`___`。
* 我`___`餓了。
* 我`___`餓了，我可以吃半頭豬。

根據可獲得的訊息量，我們可以用不同的詞填空，
如『很高興』（"happy"）、『不』（"not"）和『非常』（"very"）。
很明顯，每個短語的『下文』傳達了重要訊息（如果有的話），
而這些訊息關乎到選擇哪個詞來填空，
所以無法利用這一點的序列模型將在相關任務上表現不佳。
例如，如果要做好命名實體識別
（例如，識別『Green』指的是『格林先生』還是綠色），
不同長度的上下文範圍重要性是相同的。
為了獲得一些解決問題的靈感，讓我們先迂迴到機率圖模型。

## 隱馬爾可夫模型中的動態規劃

這一小節是用來說明動態規劃問題的，
具體的技術細節對於理解深度學習模型並不重要，
但它有助於我們思考為什麼要使用深度學習，
以及為什麼要選擇特定的架構。

如果我們想用機率圖模型來解決這個問題，
可以設計一個隱變數模型：
在任意時間步$t$，假設存在某個隱變數$h_t$，
透過機率$P(x_t \mid h_t)$控制我們觀測到的$x_t$。
此外，任何$h_t \to h_{t+1}$轉移
都是由一些狀態轉移機率$P(h_{t+1} \mid h_{t})$給出。
這個機率圖模型就是一個*隱馬爾可夫模型*（hidden Markov model，HMM），
如 :numref:`fig_hmm`所示。

![隱馬爾可夫模型](../img/hmm.svg)
:label:`fig_hmm`

因此，對於有$T$個觀測值的序列，
我們在觀測狀態和隱狀態上具有以下聯合機率分佈：

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

現在，假設我們觀測到所有的$x_i$，除了$x_j$，
並且我們的目標是計算$P(x_j \mid x_{-j})$，
其中$x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$。
由於$P(x_j \mid x_{-j})$中沒有隱變數，
因此我們考慮對$h_1, \ldots, h_T$選擇構成的
所有可能的組合進行求和。
如果任何$h_i$可以接受$k$個不同的值（有限的狀態數），
這意味著我們需要對$k^T$個項求和，
這個任務顯然難於登天。
幸運的是，有個巧妙的解決方案：*動態規劃*（dynamic programming）。

要了解動態規劃的工作方式，
我們考慮對隱變數$h_1, \ldots, h_T$的依次求和。
根據 :eqref:`eq_hmm_jointP`，將得出：

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

通常，我們將*前向遞迴*（forward recursion）寫為：

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

遞迴被初始化為$\pi_1(h_1) = P(h_1)$。
符號簡化，也可以寫成$\pi_{t+1} = f(\pi_t, x_t)$，
其中$f$是一些可學習的函式。
這看起來就像我們在迴圈神經網路中討論的隱變數模型中的更新方程。

與前向遞迴一樣，我們也可以使用後向遞迴對同一組隱變數求和。這將得到：

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

因此，我們可以將*後向遞迴*（backward recursion）寫為：

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

初始化$\rho_T(h_T) = 1$。
前向和後向遞迴都允許我們對$T$個隱變數在$\mathcal{O}(kT)$
（線性而不是指數）時間內對$(h_1, \ldots, h_T)$的所有值求和。
這是使用圖模型進行機率推理的巨大好處之一。
它也是通用訊息傳遞演算法 :cite:`Aji.McEliece.2000`的一個非常特殊的例子。
結合前向和後向遞迴，我們能夠計算

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

因為符號簡化的需要，後向遞迴也可以寫為$\rho_{t-1} = g(\rho_t, x_t)$，
其中$g$是一個可以學習的函式。
同樣，這看起來非常像一個更新方程，
只是不像我們在迴圈神經網路中看到的那樣前向運算，而是後向計算。
事實上，知道未來資料何時可用對隱馬爾可夫模型是有益的。
訊號處理學家將是否知道未來觀測這兩種情況區分為內插和外推，
有關更多詳細訊息，請參閱 :cite:`Doucet.De-Freitas.Gordon.2001`。

## 雙向模型

如果我們希望在迴圈神經網路中擁有一種機制，
使之能夠提供與隱馬爾可夫模型類似的前瞻能力，
我們就需要修改迴圈神經網路的設計。
幸運的是，這在概念上很容易，
只需要增加一個『從最後一個詞元開始從後向前執行』的迴圈神經網路，
而不是只有一個在前向模式下『從第一個詞元開始執行』的迴圈神經網路。
*雙向迴圈神經網路*（bidirectional RNNs）
新增了反向傳遞訊息的隱藏層，以便更靈活地處理此類訊息。
 :numref:`fig_birnn`描述了具有單個隱藏層的雙向迴圈神經網路的架構。

![雙向迴圈神經網路架構](../img/birnn.svg)
:label:`fig_birnn`

事實上，這與隱馬爾可夫模型中的動態規劃的前向和後向遞迴沒有太大區別。
其主要區別是，在隱馬爾可夫模型中的方程具有特定的統計意義。
雙向迴圈神經網路沒有這樣容易理解的解釋，
我們只能把它們當作通用的、可學習的函式。
這種轉變集中體現了現代深度網路的設計原則：
首先使用經典統計模型的函式依賴型別，然後將其引數化為通用形式。

### 定義

雙向迴圈神經網路是由 :cite:`Schuster.Paliwal.1997`提出的，
關於各種架構的詳細討論請參閱 :cite:`Graves.Schmidhuber.2005`。
讓我們看看這樣一個網路的細節。

對於任意時間步$t$，給定一個小批次的輸入資料
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
（樣本數$n$，每個範例中的輸入數$d$），
並且令隱藏層啟用函式為$\phi$。
在雙向架構中，我們設該時間步的前向和反向隱狀態分別為
$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$和
$\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$，
其中$h$是隱藏單元的數目。
前向和反向隱狀態的更新如下：

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

其中，權重$\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$
和偏置$\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h}, \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$都是模型引數。

接下來，將前向隱狀態$\overrightarrow{\mathbf{H}}_t$
和反向隱狀態$\overleftarrow{\mathbf{H}}_t$連線起來，
獲得需要送入輸出層的隱狀態$\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$。
在具有多個隱藏層的深度雙向迴圈神經網路中，
該訊息作為輸入傳遞到下一個雙向層。
最後，輸出層計算得到的輸出為
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$（$q$是輸出單元的數目）：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

這裡，權重矩陣$\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$
和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$
是輸出層的模型引數。
事實上，這兩個方向可以擁有不同數量的隱藏單元。

### 模型的計算代價及其應用

雙向迴圈神經網路的一個關鍵屬性是：使用來自序列兩端的訊息來估計輸出。
也就是說，我們使用來自過去和未來的觀測訊息來預測當前的觀測。
但是在對下一個詞元進行預測的情況中，這樣的模型並不是我們所需的。
因為在預測下一個詞元時，我們終究無法知道下一個詞元的下文是什麼，
所以將不會得到很好的精度。
具體地說，在訓練期間，我們能夠利用過去和未來的資料來估計現在空缺的詞；
而在測試期間，我們只有過去的資料，因此精度將會很差。
下面的實驗將說明這一點。

另一個嚴重問題是，雙向迴圈神經網路的計算速度非常慢。
其主要原因是網路的前向傳播需要在雙向層中進行前向和後向遞迴，
並且網路的反向傳播還依賴於前向傳播的結果。
因此，梯度求解將有一個非常長的鏈。

雙向層的使用在實踐中非常少，並且僅僅應用於部分場合。
例如，填充缺失的單詞、詞元註解（例如，用於命名實體識別）
以及作為序列處理流水線中的一個步驟對序列進行編碼（例如，用於機器翻譯）。
在 :numref:`sec_bert`和 :numref:`sec_sentiment_rnn`中，
我們將介紹如何使用雙向迴圈神經網路編碼文字序列。

## (**雙向迴圈神經網路的錯誤應用**)

由於雙向迴圈神經網路使用了過去的和未來的資料，
所以我們不能盲目地將這一語言模型應用於任何預測任務。
儘管模型產出的困惑度是合理的，
該模型預測未來詞元的能力卻可能存在嚴重缺陷。
我們用下面的範例程式碼引以為戒，以防在錯誤的環境中使用它們。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# 載入資料
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 透過設定『bidirective=True』來定義雙向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# 訓練模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# 載入資料
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 透過設定『bidirective=True』來定義雙向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# 訓練模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn

#載入資料
batch_size, num_steps, device  = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
#透過設定『direction='bidirect'』來定義雙向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, direction='bidirect', time_major=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
#訓練模型
num_epochs, lr = 500, 1.0
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

上述結果顯然令人瞠目結舌。
關於如何更有效地使用雙向迴圈神經網路的討論，
請參閱 :numref:`sec_sentiment_rnn`中的情感分類應用。

## 小結

* 在雙向迴圈神經網路中，每個時間步的隱狀態由當前時間步的前後資料同時決定。
* 雙向迴圈神經網路與機率圖模型中的『前向-後向』演算法具有相似性。
* 雙向迴圈神經網路主要用於序列編碼和給定雙向上下文的觀測估計。
* 由於梯度鏈更長，因此雙向迴圈神經網路的訓練代價非常高。

## 練習

1. 如果不同方向使用不同數量的隱藏單位，$\mathbf{H_t}$的形狀會發生怎樣的變化？
1. 設計一個具有多個隱藏層的雙向迴圈神經網路。
1. 在自然語言中一詞多義很常見。例如，『bank』一詞在不同的上下文『i went to the bank to deposit cash』和『i went to the bank to sit down』中有不同的含義。如何設計一個神經網路模型，使其在給定上下文序列和單詞的情況下，回傳該單詞在此上下文中的向量表示？哪種型別的神經網路架構更適合處理一詞多義？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2774)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2773)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11835)
:end_tab:
