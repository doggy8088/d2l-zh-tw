# 數值穩定性和模型初始化
:label:`sec_numerical_stability`

到目前為止，我們實現的每個模型都是根據某個預先指定的分佈來初始化模型的引數。
有人會認為初始化方案是理所當然的，忽略瞭如何做出這些選擇的細節。甚至有人可能會覺得，初始化方案的選擇並不是特別重要。
相反，初始化方案的選擇在神經網路學習中起著舉足輕重的作用，
它對保持數值穩定性至關重要。
此外，這些初始化方案的選擇可以與非線性啟用函式的選擇有趣的結合在一起。
我們選擇哪個函式以及如何初始化引數可以決定最佳化演算法收斂的速度有多快。
糟糕選擇可能會導致我們在訓練時遇到梯度爆炸或梯度消失。
本節將更詳細地探討這些主題，並討論一些有用的啟發式方法。
這些啟發式方法在整個深度學習生涯中都很有用。

## 梯度消失和梯度爆炸

考慮一個具有$L$層、輸入$\mathbf{x}$和輸出$\mathbf{o}$的深層網路。
每一層$l$由變換$f_l$定義，
該變換的引數為權重$\mathbf{W}^{(l)}$，
其隱藏變數是$\mathbf{h}^{(l)}$（令 $\mathbf{h}^{(0)} = \mathbf{x}$）。
我們的網路可以表示為：

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ 因此 } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

如果所有隱藏變數和輸入都是向量，
我們可以將$\mathbf{o}$關於任何一組引數$\mathbf{W}^{(l)}$的梯度寫為下式：

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

換言之，該梯度是$L-l$個矩陣
$\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$
與梯度向量 $\mathbf{v}^{(l)}$的乘積。
因此，我們容易受到數值下溢問題的影響.
當將太多的機率乘在一起時，這些問題經常會出現。
在處理機率時，一個常見的技巧是切換到對數空間，
即將數值表示的壓力從尾數轉移到指數。
不幸的是，上面的問題更為嚴重：
最初，矩陣 $\mathbf{M}^{(l)}$ 可能具有各種各樣的特徵值。
他們可能很小，也可能很大；
他們的乘積可能非常大，也可能非常小。

不穩定梯度帶來的風險不止在於數值表示；
不穩定梯度也威脅到我們最佳化演算法的穩定性。
我們可能面臨一些問題。
要麼是*梯度爆炸*（gradient exploding）問題：
引數更新過大，破壞了模型的穩定收斂；
要麼是*梯度消失*（gradient vanishing）問題：
引數更新過小，在每次更新時幾乎不會移動，導致模型無法學習。

### (**梯度消失**)

曾經sigmoid函式$1/(1 + \exp(-x))$（ :numref:`sec_mlp`提到過）很流行，
因為它類似於閾值函式。
由於早期的人工神經網路受到生物神經網路的啟發，
神經元要麼完全啟用要麼完全不啟用（就像生物神經元）的想法很有吸引力。
然而，它卻是導致梯度消失問題的一個常見的原因，
讓我們仔細看看sigmoid函式為什麼會導致梯度消失。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle

x = paddle.arange(start=-8.0, end=8.0, step=0.1, dtype='float32')
x.stop_gradient = False
y = paddle.nn.functional.sigmoid(x)
y.backward(paddle.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

正如上圖，當sigmoid函式的輸入很大或是很小時，它的梯度都會消失。
此外，當反向傳播透過許多層時，除非我們在剛剛好的地方，
這些地方sigmoid函式的輸入接近於零，否則整個乘積的梯度可能會消失。
當我們的網路有很多層時，除非我們很小心，否則在某一層可能會切斷梯度。
事實上，這個問題曾經困擾著深度網路的訓練。
因此，更穩定的ReLU系列函式已經成為從業者的預設選擇（雖然在神經科學的角度看起來不太合理）。

### [**梯度爆炸**]

相反，梯度爆炸可能同樣令人煩惱。
為了更好地說明這一點，我們生成100個高斯隨機矩陣，並將它們與某個初始矩陣相乘。
對於我們選擇的尺度（方差$\sigma^2=1$），矩陣乘積發生爆炸。
當這種情況是由於深度網路的初始化所導致時，我們沒有機會讓梯度下降最佳化器收斂。

```{.python .input}
M = np.random.normal(size=(4, 4))
print('一個矩陣 \n', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('乘以100個矩陣後\n', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('一個矩陣 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100個矩陣後\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('一個矩陣 \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('乘以100個矩陣後\n', M.numpy())
```

```{.python .input}
#@tab paddle
M = paddle.normal(0, 1, shape=(4,4))
print('一個矩陣 \n',M)
for i in range(100):
    M = paddle.mm(M, paddle.normal(0, 1, shape=(4, 4)))

print('乘以100個矩陣後\n', M)
```

### 打破對稱性

神經網路設計中的另一個問題是其引數化所固有的對稱性。
假設我們有一個簡單的多層感知機，它有一個隱藏層和兩個隱藏單元。
在這種情況下，我們可以對第一層的權重$\mathbf{W}^{(1)}$進行重排列，
並且同樣對輸出層的權重進行重排列，可以獲得相同的函式。
第一個隱藏單元與第二個隱藏單元沒有什麼特別的區別。
換句話說，我們在每一層的隱藏單元之間具有排列對稱性。

假設輸出層將上述兩個隱藏單元的多層感知機轉換為僅一個輸出單元。
想象一下，如果我們將隱藏層的所有引數初始化為$\mathbf{W}^{(1)} = c$，
$c$為常數，會發生什麼？
在這種情況下，在前向傳播期間，兩個隱藏單元採用相同的輸入和引數，
產生相同的啟用，該啟用被送到輸出單元。
在反向傳播期間，根據引數$\mathbf{W}^{(1)}$對輸出單元進行微分，
得到一個梯度，其元素都取相同的值。
因此，在基於梯度的迭代（例如，小批次隨機梯度下降）之後，
$\mathbf{W}^{(1)}$的所有元素仍然採用相同的值。
這樣的迭代永遠不會打破對稱性，我們可能永遠也無法實現網路的表達能力。
隱藏層的行為就好像只有一個單元。
請注意，雖然小批次隨機梯度下降不會打破這種對稱性，但暫退法正則化可以。

## 引數初始化

解決（或至少減輕）上述問題的一種方法是進行引數初始化，
最佳化期間的注意和適當的正則化也可以進一步提高穩定性。

### 預設初始化

在前面的部分中，例如在 :numref:`sec_linear_concise`中，
我們使用常態分佈來初始化權重值。如果我們不指定初始化方法，
框架將使用預設的隨機初始化方法，對於中等難度的問題，這種方法通常很有效。

### Xavier初始化
:label:`subsec_xavier`

讓我們看看某些*沒有非線性*的全連線層輸出（例如，隱藏變數）$o_{i}$的尺度分佈。
對於該層$n_\mathrm{in}$輸入$x_j$及其相關權重$w_{ij}$，輸出由下式給出

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

權重$w_{ij}$都是從同一分佈中獨立抽取的。
此外，讓我們假設該分佈具有零均值和方差$\sigma^2$。
請注意，這並不意味著分佈必須是高斯的，只是均值和方差需要存在。
現在，讓我們假設層$x_j$的輸入也具有零均值和方差$\gamma^2$，
並且它們獨立於$w_{ij}$並且彼此獨立。
在這種情況下，我們可以按如下方式計算$o_i$的平均值和方差：

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

保持方差不變的一種方法是設定$n_\mathrm{in} \sigma^2 = 1$。
現在考慮反向傳播過程，我們面臨著類似的問題，儘管梯度是從更靠近輸出的層傳播的。
使用與前向傳播相同的推斷，我們可以看到，除非$n_\mathrm{out} \sigma^2 = 1$，
否則梯度的方差可能會增大，其中$n_\mathrm{out}$是該層的輸出的數量。
這使得我們進退兩難：我們不可能同時滿足這兩個條件。
相反，我們只需滿足：

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ 或等價於 }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

這就是現在標準且實用的*Xavier初始化*的基礎，
它以其提出者 :cite:`Glorot.Bengio.2010` 第一作者的名字命名。
通常，Xavier初始化從均值為零，方差
$\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$
的高斯分佈中取樣權重。
我們也可以將其改為選擇從均勻分佈中抽取權重時的方差。
注意均勻分佈$U(-a, a)$的方差為$\frac{a^2}{3}$。
將$\frac{a^2}{3}$代入到$\sigma^2$的條件中，將得到初始化值域：

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

儘管在上述數學推理中，『不存在非線性』的假設在神經網路中很容易被違反，
但Xavier初始化方法在實踐中被證明是有效的。

### 額外閱讀

上面的推理僅僅觸及了現代引數初始化方法的皮毛。
深度學習框架通常實現十幾種不同的啟發式方法。
此外，引數初始化一直是深度學習基礎研究的熱點領域。
其中包括專門用於引數繫結（共享）、超解析度、序列模型和其他情況的啟發式演算法。
例如，Xiao等人示範了透過使用精心設計的初始化方法
 :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`，
可以無須架構上的技巧而訓練10000層神經網路的可能性。

如果有讀者對該主題感興趣，我們建議深入研究本模組的內容，
閱讀提出並分析每種啟發式方法的論文，然後探索有關該主題的最新出版物。
也許會偶然發現甚至發明一個聰明的想法，併為深度學習框架提供一個實現。

## 小結

* 梯度消失和梯度爆炸是深度網路中常見的問題。在引數初始化時需要非常小心，以確保梯度和引數可以得到很好的控制。
* 需要用啟發式的初始化方法來確保初始梯度既不太大也不太小。
* ReLU啟用函式緩解了梯度消失問題，這樣可以加速收斂。
* 隨機初始化是保證在進行最佳化前打破對稱性的關鍵。
* Xavier初始化表明，對於每一層，輸出的方差不受輸入數量的影響，任何梯度的方差不受輸出數量的影響。

## 練習

1. 除了多層感知機的排列對稱性之外，還能設計出其他神經網路可能會表現出對稱性且需要被打破的情況嗎？
2. 我們是否可以將線性迴歸或softmax迴歸中的所有權重引數初始化為相同的值？
3. 在相關資料中查詢兩個矩陣乘積特徵值的解析界。這對確保梯度條件合適有什麼啟示？
4. 如果我們知道某些項是發散的，我們能在事後修正嗎？看看關於按層自適應速率縮放的論文 :cite:`You.Gitman.Ginsburg.2017` 。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1819)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1818)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1817)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11776)
:end_tab:
