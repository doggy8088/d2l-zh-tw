# 深度迴圈神經網路

:label:`sec_deep_rnn`

到目前為止，我們只討論了具有一個單向隱藏層的迴圈神經網路。
其中，隱變數和觀測值與具體的函式形式的互動方式是相當隨意的。
只要互動型別建模具有足夠的靈活性，這就不是一個大問題。
然而，對一個單層來說，這可能具有相當的挑戰性。
之前線上性模型中，我們透過新增更多的層來解決這個問題。
而在迴圈神經網路中，我們首先需要確定如何新增更多的層，
以及在哪裡新增額外的非線性，因此這個問題有點棘手。

事實上，我們可以將多層迴圈神經網路堆疊在一起，
透過對幾個簡單層的組合，產生了一個靈活的機制。
特別是，資料可能與不同層的堆疊有關。
例如，我們可能希望保持有關金融市場狀況
（熊市或牛市）的宏觀資料可用，
而微觀資料只記錄較短期的時間動態。

 :numref:`fig_deep_rnn`描述了一個具有$L$個隱藏層的深度迴圈神經網路，
每個隱狀態都連續地傳遞到當前層的下一個時間步和下一層的當前時間步。

![深度迴圈神經網路結構](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

## 函式依賴關係

我們可以將深度架構中的函式依賴關係形式化，
這個架構是由 :numref:`fig_deep_rnn`中描述了$L$個隱藏層構成。
後續的討論主要集中在經典的迴圈神經網路模型上，
但是這些討論也適應於其他序列模型。

假設在時間步$t$有一個小批次的輸入資料
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
（樣本數：$n$，每個樣本中的輸入數：$d$）。
同時，將$l^\mathrm{th}$隱藏層（$l=1,\ldots,L$）
的隱狀態設為$\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$
（隱藏單元數：$h$），
輸出層變數設為$\mathbf{O}_t \in \mathbb{R}^{n \times q}$
（輸出數：$q$）。
設定$\mathbf{H}_t^{(0)} = \mathbf{X}_t$，
第$l$個隱藏層的隱狀態使用啟用函式$\phi_l$，則：

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

其中，權重$\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$，
$\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$和
偏置$\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$
都是第$l$個隱藏層的模型引數。

最後，輸出層的計算僅基於第$l$個隱藏層最終的隱狀態：

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中，權重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$都是輸出層的模型引數。

與多層感知機一樣，隱藏層數目$L$和隱藏單元數目$h$都是超引數。
也就是說，它們可以由我們調整的。
另外，用門控迴圈單元或長短期記憶網路的隱狀態
來代替 :eqref:`eq_deep_rnn_H`中的隱狀態進行計算，
可以很容易地得到深度門控迴圈神經網路或深度長短期記憶神經網路。

## 簡潔實現

實現多層迴圈神經網路所需的許多邏輯細節在高階API中都是現成的。
簡單起見，我們僅示範使用此類內建函式的實現方式。
以長短期記憶網路模型為例，
該程式碼與之前在 :numref:`sec_lstm`中使用的程式碼非常相似，
實際上唯一的區別是我們指定了層的數量，
而不是使用單一層這個預設值。
像往常一樣，我們從載入資料集開始。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

像選擇超引數這類架構決策也跟 :numref:`sec_lstm`中的決策非常相似。
因為我們有不同的詞元，所以輸入和輸出都選擇相同數量，即`vocab_size`。
隱藏單元的數量仍然是$256$。
唯一的區別是，我們現在(**透過`num_layers`的值來設定隱藏層數**)。

```{.python .input}
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
device = d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

```{.python .input}
#@tab paddle
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, time_major=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
```

## [**訓練**]與預測

由於使用了長短期記憶網路模型來實例化兩個層，因此訓練速度被大大降低了。

```{.python .input}
#@tab all
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
```

## 小結

* 在深度迴圈神經網路中，隱狀態的訊息被傳遞到當前層的下一時間步和下一層的當前時間步。
* 有許多不同風格的深度迴圈神經網路，
  如長短期記憶網路、門控迴圈單元、或經典迴圈神經網路。
  這些模型在深度學習框架的高階API中都有涵蓋。
* 總體而言，深度迴圈神經網路需要大量的調參（如學習率和修剪）
  來確保合適的收斂，模型的初始化也需要謹慎。

## 練習

1. 基於我們在 :numref:`sec_rnn_scratch`中討論的單層實現，
   嘗試從零開始實現兩層迴圈神經網路。
1. 在本節訓練模型中，比較使用門控迴圈單元替換長短期記憶網路後模型的精確度和訓練速度。
1. 如果增加訓練資料，能夠將困惑度降到多低？
1. 在為文字建模時，是否可以將不同作者的源資料合併？有何優劣呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2771)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2770)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11834)
:end_tab: