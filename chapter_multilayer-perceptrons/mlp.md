# 多層感知機
:label:`sec_mlp`

在 :numref:`chap_linear`中，
我們介紹了softmax迴歸（ :numref:`sec_softmax`），
然後我們從零開始實現了softmax迴歸（ :numref:`sec_softmax_scratch`），
接著使用高階API實現了演算法（ :numref:`sec_softmax_concise`），
並訓練分類器從低解析度影象中識別10類服裝。
在這個過程中，我們學習瞭如何處理資料，如何將輸出轉換為有效的機率分佈，
並應用適當的損失函式，根據模型引數最小化損失。
我們已經在簡單的線性模型背景下掌握了這些知識，
現在我們可以開始對深度神經網路的探索，這也是本書主要涉及的一類模型。

## 隱藏層

我們在 :numref:`subsec_linear_model`中描述了仿射變換，
它是一種帶有偏置項的線性變換。
首先，回想一下如 :numref:`fig_softmaxreg`中所示的softmax迴歸的模型架構。
該模型透過單個仿射變換將我們的輸入直接對映到輸出，然後進行softmax操作。
如果我們的標籤透過仿射變換後確實與我們的輸入資料相關，那麼這種方法確實足夠了。
但是，仿射變換中的*線性*是一個很強的假設。

### 線性模型可能會出錯

例如，線性意味著*單調*假設：
任何特徵的增大都會導致模型輸出的增大（如果對應的權重為正），
或者導致模型輸出的減小（如果對應的權重為負）。
有時這是有道理的。
例如，如果我們試圖預測一個人是否會償還貸款。
我們可以認為，在其他條件不變的情況下，
收入較高的申請人比收入較低的申請人更有可能償還貸款。
但是，雖然收入與還款機率存在單調性，但它們不是線性相關的。
收入從0增加到5萬，可能比從100萬增加到105萬帶來更大的還款可能性。
處理這一問題的一種方法是對我們的資料進行預處理，
使線性變得更合理，如使用收入的對數作為我們的特徵。

然而我們可以很容易找出違反單調性的例子。
例如，我們想要根據體溫預測死亡率。
對體溫高於37攝氏度的人來說，溫度越高風險越大。
然而，對體溫低於37攝氏度的人來說，溫度越高風險就越低。
在這種情況下，我們也可以透過一些巧妙的預處理來解決問題。
例如，我們可以使用與37攝氏度的距離作為特徵。

但是，如何對貓和狗的影象進行分類呢？
增加位置$(13, 17)$處畫素的強度是否總是增加（或降低）影象描繪狗的似然？
對線性模型的依賴對應於一個隱含的假設，
即區分貓和狗的唯一要求是評估單個畫素的強度。
在一個倒置影象後依然保留類別的世界裡，這種方法註定會失敗。

與我們前面的例子相比，這裡的線性很荒謬，
而且我們難以透過簡單的預處理來解決這個問題。
這是因為任何畫素的重要性都以複雜的方式取決於該畫素的上下文（周圍畫素的值）。
我們的資料可能會有一種表示，這種表示會考慮到我們在特徵之間的相關互動作用。
在此表示的基礎上建立一個線性模型可能會是合適的，
但我們不知道如何手動計算這麼一種表示。
對於深度神經網路，我們使用觀測資料來聯合學習隱藏層表示和應用於該表示的線性預測器。

### 在網路中加入隱藏層

我們可以透過在網路中加入一個或多個隱藏層來克服線性模型的限制，
使其能處理更普遍的函式關係型別。
要做到這一點，最簡單的方法是將許多全連線層堆疊在一起。
每一層都輸出到上面的層，直到生成最後的輸出。
我們可以把前$L-1$層看作表示，把最後一層看作線性預測器。
這種架構通常稱為*多層感知機*（multilayer perceptron），通常縮寫為*MLP*。
下面，我們以圖的方式描述了多層感知機（ :numref:`fig_mlp`）。

![一個單隱藏層的多層感知機，具有5個隱藏單元](../img/mlp.svg)
:label:`fig_mlp`

這個多層感知機有4個輸入，3個輸出，其隱藏層包含5個隱藏單元。
輸入層不涉及任何計算，因此使用此網路產生輸出只需要實現隱藏層和輸出層的計算。
因此，這個多層感知機中的層數為2。
注意，這兩個層都是全連線的。
每個輸入都會影響隱藏層中的每個神經元，
而隱藏層中的每個神經元又會影響輸出層中的每個神經元。

然而，正如 :numref:`subsec_parameterization-cost-fc-layers`所說，
具有全連線層的多層感知機的引數開銷可能會高得令人望而卻步。
即使在不改變輸入或輸出大小的情況下，
可能在引數節約和模型有效性之間進行權衡 :cite:`Zhang.Tay.Zhang.ea.2021`。

### 從線性到非線性

同之前的章節一樣，
我們透過矩陣$\mathbf{X} \in \mathbb{R}^{n \times d}$
來表示$n$個樣本的小批次，
其中每個樣本具有$d$個輸入特徵。
對於具有$h$個隱藏單元的單隱藏層多層感知機，
用$\mathbf{H} \in \mathbb{R}^{n \times h}$表示隱藏層的輸出，
稱為*隱藏表示*（hidden representations）。
在數學或程式碼中，$\mathbf{H}$也被稱為*隱藏層變數*（hidden-layer variable）
或*隱藏變數*（hidden variable）。
因為隱藏層和輸出層都是全連線的，
所以我們有隱藏層權重$\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$
和隱藏層偏置$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
以及輸出層權重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$
和輸出層偏置$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$。
形式上，我們按如下方式計算單隱藏層多層感知機的輸出
$\mathbf{O} \in \mathbb{R}^{n \times q}$：

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

注意在新增隱藏層之後，模型現在需要跟蹤和更新額外的引數。
可我們能從中得到什麼好處呢？在上面定義的模型裡，我們沒有好處！
原因很簡單：上面的隱藏單元由輸入的仿射函式給出，
而輸出（softmax操作前）只是隱藏單元的仿射函式。
仿射函式的仿射函式本身就是仿射函式，
但是我們之前的線性模型已經能夠表示任何仿射函式。

我們可以證明這一等價性，即對於任意權重值，
我們只需合併隱藏層，便可產生具有引數
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$
和$\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$
的等價單層模型：

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

為了發揮多層架構的潛力，
我們還需要一個額外的關鍵要素：
在仿射變換之後對每個隱藏單元應用非線性的*啟用函式*（activation function）$\sigma$。
啟用函式的輸出（例如，$\sigma(\cdot)$）被稱為*活性值*（activations）。
一般來說，有了啟用函式，就不可能再將我們的多層感知機退化成線性模型：

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

由於$\mathbf{X}$中的每一行對應於小批次中的一個樣本，
出於記號習慣的考量，
我們定義非線性函式$\sigma$也以按行的方式作用於其輸入，
即一次計算一個樣本。
我們在 :numref:`subsec_softmax_vectorization`中
以相同的方式使用了softmax符號來表示按行操作。
但是本節應用於隱藏層的啟用函式通常不僅按行操作，也按元素操作。
這意味著在計算每一層的線性部分之後，我們可以計算每個活性值，
而不需要檢視其他隱藏單元所取的值。對於大多數啟用函式都是這樣。

為了建立更通用的多層感知機，
我們可以繼續堆疊這樣的隱藏層，
例如$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$和$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$，
一層疊一層，從而產生更有表達能力的模型。

### 通用近似定理

多層感知機可以透過隱藏神經元，捕捉到輸入之間複雜的相互作用，
這些神經元依賴於每個輸入的值。
我們可以很容易地設計隱藏節點來執行任意計算。
例如，在一對輸入上進行基本邏輯操作，多層感知機是通用近似器。
即使是網路只有一個隱藏層，給定足夠的神經元和正確的權重，
我們可以對任意函式建模，儘管實際中學習該函式是很困難的。
神經網路有點像C語言。
C語言和任何其他現代程式語言一樣，能夠表達任何可計算的程式。
但實際上，想出一個符合規範的程式才是最困難的部分。

而且，雖然一個單隱層網路能學習任何函式，
但並不意味著我們應該嘗試使用單隱藏層網路來解決所有問題。
事實上，透過使用更深（而不是更廣）的網路，我們可以更容易地逼近許多函式。
我們將在後麵的章節中進行更細緻的討論。

## 啟用函式
:label:`subsec_activation_functions`

*啟用函式*（activation function）透過計算加權和並加上偏置來確定神經元是否應該被啟用，
它們將輸入訊號轉換為輸出的可微運算。
大多數啟用函式都是非線性的。
由於啟用函式是深度學習的基礎，下面(**簡要介紹一些常見的啟用函式**)。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
```

### ReLU函式

最受歡迎的啟用函式是*修正線性單元*（Rectified linear unit，*ReLU*），
因為它實現簡單，同時在各種預測任務中表現良好。
[**ReLU提供了一種非常簡單的非線性變換**]。
給定元素$x$，ReLU函式被定義為該元素與$0$的最大值：

(**$$\operatorname{ReLU}(x) = \max(x, 0).$$**)

通俗地說，ReLU函式透過將相應的活性值設為0，僅保留正元素並丟棄所有負元素。
為了直觀感受一下，我們可以畫出函式的曲線圖。
正如從圖中所看到，啟用函式是分段線性的。

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab paddle
x = paddle.arange(-8.0, 8.0, 0.1, dtype='float32')
x.stop_gradient = False
y = paddle.nn.functional.relu(x)
d2l.plot(x.detach().numpy(), y.detach().numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

當輸入為負時，ReLU函式的導數為0，而當輸入為正時，ReLU函式的導數為1。
注意，當輸入值精確等於0時，ReLU函式不可導。
在此時，我們預設使用左側的導數，即當輸入為0時導數為0。
我們可以忽略這種情況，因為輸入可能永遠都不會是0。
這裡引用一句古老的諺語，『如果微妙的邊界條件很重要，我們很可能是在研究數學而非工程』，
這個觀點正好適用於這裡。
下面我們繪製ReLU函式的導數。

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
#@tab paddle
y.backward(paddle.ones_like(x), retain_graph=True)
d2l.plot(x.detach().numpy(), x.grad.numpy(), 'x', 'grad of relu', figsize=(5, 2.5))
```

使用ReLU的原因是，它求導表現得特別好：要麼讓引數消失，要麼讓引數透過。
這使得最佳化表現得更好，並且ReLU減輕了困擾以往神經網路的梯度消失問題（稍後將詳細介紹）。

注意，ReLU函式有許多變體，包括*引數化ReLU*（Parameterized ReLU，*pReLU*）
函式 :cite:`He.Zhang.Ren.ea.2015`。
該變體為ReLU新增了一個線性項，因此即使引數是負的，某些訊息仍然可以透過：

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### sigmoid函式

[**對於一個定義域在$\mathbb{R}$中的輸入，
*sigmoid函式*將輸入變換為區間(0, 1)上的輸出**]。
因此，sigmoid通常稱為*擠壓函式*（squashing function）：
它將範圍（-inf, inf）中的任意輸入壓縮到區間（0, 1）中的某個值：

(**$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$**)

在最早的神經網路中，科學家們感興趣的是對『激發』或『不激發』的生物神經元進行建模。
因此，這一領域的先驅可以一直追溯到人工神經元的發明者麥卡洛克和皮茨，他們專注於閾值單元。
閾值單元在其輸入低於某個閾值時取值0，當輸入超過閾值時取值1。

當人們逐漸關注到到基於梯度的學習時，
sigmoid函式是一個自然的選擇，因為它是一個平滑的、可微的閾值單元近似。
當我們想要將輸出視作二元分類問題的機率時，
sigmoid仍然被廣泛用作輸出單元上的啟用函式
（sigmoid可以視為softmax的特例）。
然而，sigmoid在隱藏層中已經較少使用，
它在大部分時候被更簡單、更容易訓練的ReLU所取代。
在後面關於迴圈神經網路的章節中，我們將描述利用sigmoid單元來控制時序訊息流的架構。

下面，我們繪製sigmoid函式。
注意，當輸入接近0時，sigmoid函式接近線性變換。

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab paddle
y = paddle.nn.functional.sigmoid(x)
d2l.plot(x.detach().numpy(), y.detach().numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

sigmoid函式的導數為下面的公式：

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

sigmoid函式的導數影象如下所示。
注意，當輸入為0時，sigmoid函式的導數達到最大值0.25；
而輸入在任一方向上越遠離0點時，導數越接近0。

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
#@tab paddle
# 清除以前的梯度。
x.clear_gradient()
y.backward(paddle.ones_like(x), retain_graph=True)
d2l.plot(x.detach().numpy(), x.grad.numpy(), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### tanh函式

與sigmoid函式類似，
[**tanh(雙曲正切)函式也能將其輸入壓縮轉換到區間(-1, 1)上**]。
tanh函式的公式如下：

(**$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$**)

下面我們繪製tanh函式。
注意，當輸入在0附近時，tanh函式接近線性變換。
函式的形狀類似於sigmoid函式，
不同的是tanh函式關於座標系原點中心對稱。

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab paddle
y = paddle.tanh(x)
d2l.plot(x.detach().numpy(), y.detach().numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

tanh函式的導數是：

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

tanh函式的導數影象如下所示。
當輸入接近0時，tanh函式的導數接近最大值1。
與我們在sigmoid函式影象中看到的類似，
輸入在任一方向上越遠離0點，導數越接近0。

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
#@tab paddle
# 清除以前的梯度。
x.clear_gradient()
y.backward(paddle.ones_like(x), retain_graph=True)
d2l.plot(x.detach().numpy(), x.grad.numpy(), 'x', 'grad of tanh', figsize=(5, 2.5))
```

總結一下，我們現在瞭解瞭如何結合非線性函式來建立具有更強表達能力的多層神經網路架構。
順便說一句，這些知識已經讓你掌握了一個類似於1990年左右深度學習從業者的工具。
在某些方面，你比在20世紀90年代工作的任何人都有優勢，
因為你可以利用功能強大的開源深度學習框架，只需幾行程式碼就可以快速建立模型，
而以前訓練這些網路需要研究人員編寫數千行的C或Fortran程式碼。

## 小結

* 多層感知機在輸出層和輸入層之間增加一個或多個全連線隱藏層，並透過啟用函式轉換隱藏層的輸出。
* 常用的啟用函式包括ReLU函式、sigmoid函式和tanh函式。

## 練習

1. 計算pReLU啟用函式的導數。
1. 證明一個僅使用ReLU（或pReLU）的多層感知機構造了一個連續的分段線性函式。
1. 證明$\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$。
1. 假設我們有一個非線性單元，將它一次應用於一個小批次的資料。這會導致什麼樣的問題？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1797)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1796)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1795)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11768)
:end_tab:
