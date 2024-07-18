# 線性迴歸
:label:`sec_linear_regression`

*迴歸*（regression）是能為一個或多個自變量與因變量之間關係建模的一類方法。
在自然科學和社會科學領域，迴歸經常用來表示輸入和輸出之間的關係。

在機器學習領域中的大多數任務通常都與*預測*（prediction）有關。
當我們想預測一個數值時，就會涉及到迴歸問題。
常見的例子包括：預測價格（房屋、股票等）、預測住院時間（針對住院病人等）、
預測需求（零售銷量等）。
但不是所有的*預測*都是迴歸問題。
在後麵的章節中，我們將介紹分類問題。分類問題的目標是預測資料屬於一組類別中的哪一個。

## 線性迴歸的基本元素

*線性迴歸*（linear regression）可以追溯到19世紀初，
它在迴歸的各種標準工具中最簡單而且最流行。
線性迴歸基於幾個簡單的假設：
首先，假設自變量$\mathbf{x}$和因變量$y$之間的關係是線性的，
即$y$可以表示為$\mathbf{x}$中元素的加權和，這裡通常允許包含觀測值的一些噪聲；
其次，我們假設任何噪聲都比較正常，如噪聲遵循常態分佈。

為了解釋*線性迴歸*，我們舉一個實際的例子：
我們希望根據房屋的面積（平方英尺）和房齡（年）來估算房屋價格（美元）。
為了開發一個能預測房價的模型，我們需要收集一個真實的資料集。
這個資料集包括了房屋的銷售價格、面積和房齡。
在機器學習的術語中，該資料集稱為*訓練資料集*（training data set）
或*訓練集*（training set）。
每行數據（比如一次房屋交易相對應的資料）稱為*樣本*（sample），
也可以稱為*資料點*（data point）或*資料樣本*（data instance）。
我們把試圖預測的目標（比如預測房屋價格）稱為*標籤*（label）或*目標*（target）。
預測所依據的自變量（面積和房齡）稱為*特徵*（feature）或*協變數*（covariate）。

通常，我們使用$n$來表示資料集中的樣本數。
對索引為$i$的樣本，其輸入表示為$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，
其對應的標籤是$y^{(i)}$。

### 線性模型
:label:`subsec_linear_model`

線性假設是指目標（房屋價格）可以表示為特徵（面積和房齡）的加權和，如下麵的式子：

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

 :eqref:`eq_price-area`中的$w_{\mathrm{area}}$和$w_{\mathrm{age}}$
稱為*權重*（weight），權重決定了每個特徵對我們預測值的影響。
$b$稱為*偏置*（bias）、*偏移量*（offset）或*截距*（intercept）。
偏置是指當所有特徵都取值為0時，預測值應該為多少。
即使現實中不會有任何房子的面積是0或房齡正好是0年，我們仍然需要偏置項。
如果沒有偏置項，我們模型的表達能力將受到限制。
嚴格來說， :eqref:`eq_price-area`是輸入特徵的一個
*仿射變換*（affine transformation）。
仿射變換的特點是透過加權和對特徵進行*線性變換*（linear transformation），
並透過偏置項來進行*平移*（translation）。

給定一個資料集，我們的目標是尋找模型的權重$\mathbf{w}$和偏置$b$，
使得根據模型做出的預測大體符合資料裡的真實價格。
輸出的預測值由輸入特徵透過*線性模型*的仿射變換決定，仿射變換由所選權重和偏置確定。

而在機器學習領域，我們通常使用的是高維資料集，建模時採用線性代數表示法會比較方便。
當我們的輸入包含$d$個特徵時，我們將預測結果$\hat{y}$
（通常使用『尖角』符號表示$y$的估計值）表示為：

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

將所有特徵放到向量$\mathbf{x} \in \mathbb{R}^d$中，
並將所有權重放到向量$\mathbf{w} \in \mathbb{R}^d$中，
我們可以用點積形式來簡潔地表達模型：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

在 :eqref:`eq_linreg-y`中，
向量$\mathbf{x}$對應於單個資料樣本的特徵。
用符號表示的矩陣$\mathbf{X} \in \mathbb{R}^{n \times d}$
可以很方便地引用我們整個資料集的$n$個樣本。
其中，$\mathbf{X}$的每一行是一個樣本，每一列是一種特徵。

對於特徵集合$\mathbf{X}$，預測值$\hat{\mathbf{y}} \in \mathbb{R}^n$
可以透過矩陣-向量乘法表示為：

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b$$

這個過程中的求和將使用廣播機制
（廣播機制在 :numref:`subsec_broadcasting`中有詳細介紹）。
給定訓練資料特徵$\mathbf{X}$和對應的已知標籤$\mathbf{y}$，
線性迴歸的目標是找到一組權重向量$\mathbf{w}$和偏置$b$：
當給定從$\mathbf{X}$的同分布中取樣的新樣本特徵時，
這組權重向量和偏置能夠使得新樣本預測標籤的誤差儘可能小。

雖然我們相信給定$\mathbf{x}$預測$y$的最佳模型會是線性的，
但我們很難找到一個有$n$個樣本的真實資料集，其中對於所有的$1 \leq i \leq n$，$y^{(i)}$完全等於$\mathbf{w}^\top \mathbf{x}^{(i)}+b$。
無論我們使用什麼手段來觀察特徵$\mathbf{X}$和標籤$\mathbf{y}$，
都可能會出現少量的觀測誤差。
因此，即使確信特徵與標籤的潛在關係是線性的，
我們也會加入一個噪聲項來考慮觀測誤差帶來的影響。

在開始尋找最好的*模型引數*（model parameters）$\mathbf{w}$和$b$之前，
我們還需要兩個東西：
（1）一種模型質量的度量方式；
（2）一種能夠更新模型以提高模型預測質量的方法。

### 損失函式

在我們開始考慮如何用模型*擬合*（fit）資料之前，我們需要確定一個擬合程度的度量。
*損失函式*（loss function）能夠量化目標的*實際*值與*預測*值之間的差距。
通常我們會選擇非負數作為損失，且數值越小表示損失越小，完美預測時的損失為0。
迴歸問題中最常用的損失函式是平方誤差函式。
當樣本$i$的預測值為$\hat{y}^{(i)}$，其相應的真實標籤為$y^{(i)}$時，
平方誤差可以定義為以下公式：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

常數$\frac{1}{2}$不會帶來本質的差別，但這樣在形式上稍微簡單一些
（因為當我們對損失函式求導後常數係數為1）。
由於訓練資料集並不受我們控制，所以經驗誤差只是關於模型引數的函式。
為了進一步說明，來看下面的例子。
我們為一維情況下的迴歸問題繪製圖像，如 :numref:`fig_fit_linreg`所示。

![用線性模型擬合數據。](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

由於平方誤差函式中的二次方項，
估計值$\hat{y}^{(i)}$和觀測值$y^{(i)}$之間較大的差異將導致更大的損失。
為了度量模型在整個資料集上的質量，我們需計算在訓練集$n$個樣本上的損失均值（也等價於求和）。

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

在訓練模型時，我們希望尋找一組引數（$\mathbf{w}^*, b^*$），
這組引數能最小化在所有訓練樣本上的總損失。如下式：

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### 解析解

線性迴歸剛好是一個很簡單的最佳化問題。
與我們將在本書中所講到的其他大部分模型不同，線性迴歸的解可以用一個公式簡單地表達出來，
這類解叫作解析解（analytical solution）。
首先，我們將偏置$b$合併到引數$\mathbf{w}$中，合併方法是在包含所有引數的矩陣中附加一列。
我們的預測問題是最小化$\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$。
這在損失平面上只有一個臨界點，這個臨界點對應於整個區域的損失極小點。
將損失關於$\mathbf{w}$的導數設為0，得到解析解：

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

像線性迴歸這樣的簡單問題存在解析解，但並不是所有的問題都存在解析解。
解析解可以進行很好的數學分析，但解析解對問題的限制很嚴格，導致它無法廣泛應用在深度學習裡。

### 隨機梯度下降

即使在我們無法得到解析解的情況下，我們仍然可以有效地訓練模型。
在許多任務上，那些難以最佳化的模型效果要更好。
因此，弄清楚如何訓練這些難以最佳化的模型是非常重要的。

本書中我們用到一種名為*梯度下降*（gradient descent）的方法，
這種方法幾乎可以最佳化所有深度學習模型。
它透過不斷地在損失函式遞減的方向上更新引數來降低誤差。

梯度下降最簡單的用法是計算損失函式（資料集中所有樣本的損失均值）
關於模型引數的導數（在這裡也可以稱為梯度）。
但實際中的執行可能會非常慢：因為在每一次更新引數之前，我們必須遍歷整個資料集。
因此，我們通常會在每次需要計算更新的時候隨機抽取一小批樣本，
這種變體叫做*小批次隨機梯度下降*（minibatch stochastic gradient descent）。

在每次迭代中，我們首先隨機抽樣一個小批次$\mathcal{B}$，
它是由固定數量的訓練樣本組成的。
然後，我們計算小批次的平均損失關於模型引數的導數（也可以稱為梯度）。
最後，我們將梯度乘以一個預先確定的正數$\eta$，並從當前引數的值中減掉。

我們用下面的數學公式來表示這一更新過程（$\partial$表示偏導數）：

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

總結一下，演算法的步驟如下：
（1）初始化模型引數的值，如隨機初始化；
（2）從資料集中隨機抽取小批次樣本且在負梯度的方向上更新引數，並不斷迭代這一步驟。
對於平方損失和仿射變換，我們可以明確地寫成如下形式:

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

公式 :eqref:`eq_linreg_batch_update`中的$\mathbf{w}$和$\mathbf{x}$都是向量。
在這裡，更優雅的向量表示法比係數表示法（如$w_1, w_2, \ldots, w_d$）更具可讀性。
$|\mathcal{B}|$表示每個小批次中的樣本數，這也稱為*批次大小*（batch size）。
$\eta$表示*學習率*（learning rate）。
批次大小和學習率的值通常是手動預先指定，而不是透過模型訓練得到的。
這些可以調整但不在訓練過程中更新的引數稱為*超引數*（hyperparameter）。
*調參*（hyperparameter tuning）是選擇超引數的過程。
超引數通常是我們根據訓練迭代結果來調整的，
而訓練迭代結果是在獨立的*驗證資料集*（validation dataset）上評估得到的。

在訓練了預先確定的若干迭代次數後（或者直到滿足某些其他停止條件後），
我們記錄下模型引數的估計值，表示為$\hat{\mathbf{w}}, \hat{b}$。
但是，即使我們的函式確實是線性的且無噪聲，這些估計值也不會使損失函式真正地達到最小值。
因為演算法會使得損失向最小值緩慢收斂，但卻不能在有限的步數內非常精確地達到最小值。

線性迴歸恰好是一個在整個域中只有一個最小值的學習問題。
但是物件深度神經網路這樣複雜的模型來說，損失平面上通常包含多個最小值。
深度學習實踐者很少會去花費大力氣尋找這樣一組引數，使得在*訓練集*上的損失達到最小。
事實上，更難做到的是找到一組引數，這組引數能夠在我們從未見過的資料上實現較低的損失，
這一挑戰被稱為*泛化*（generalization）。

### 用模型進行預測

給定『已學習』的線性迴歸模型$\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$，
現在我們可以透過房屋面積$x_1$和房齡$x_2$來估計一個（未包含在訓練資料中的）新房屋價格。
給定特徵估計目標的過程通常稱為*預測*（prediction）或*推斷*（inference）。

本書將嘗試堅持使用*預測*這個詞。
雖然*推斷*這個詞已經成為深度學習的標準術語，但其實*推斷*這個詞有些用詞不當。
在統計學中，*推斷*更多地表示基於資料集估計引數。
當深度學習從業者與統計學家交談時，術語的誤用經常導致一些誤解。

## 矢量化加速

在訓練我們的模型時，我們經常希望能夠同時處理整個小批次的樣本。
為了實現這一點，需要(**我們對計算進行矢量化，
從而利用線性代數函式庫，而不是在Python中編寫開銷高昂的for迴圈**)。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
import time
import paddle
```

為了說明矢量化為什麼如此重要，我們考慮(**對向量相加的兩種方法**)。
我們實例化兩個全為1的10000維向量。
在一種方法中，我們將使用Python的for迴圈遍歷向量；
在另一種方法中，我們將依賴對`+`的呼叫。

```{.python .input}
#@tab all
n = 10000
a = d2l.ones([n])
b = d2l.ones([n])
```

由於在本書中我們將頻繁地進行執行時間的基準測試，所以[**我們定義一個計時器**]：

```{.python .input}
#@tab all
class Timer:  #@save
    """記錄多次執行時間"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """啟動計時器"""
        self.tik = time.time()

    def stop(self):
        """停止計時器並將時間記錄在清單中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """回傳平均時間"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """回傳時間總和"""
        return sum(self.times)

    def cumsum(self):
        """回傳累計時間"""
        return np.array(self.times).cumsum().tolist()
```

現在我們可以對工作負載進行基準測試。

首先，[**我們使用for迴圈，每次執行一位的加法**]。

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab paddle
c = d2l.zeros([n])
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

(**或者，我們使用過載的`+`運算符來計算按元素的和**)。

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

結果很明顯，第二種方法比第一種方法快得多。
矢量化程式碼通常會帶來數量級的加速。
另外，我們將更多的數學運算放到庫中，而無須自己編寫那麼多的計算，從而減少了出錯的可能性。

## 常態分佈與平方損失
:label:`subsec_normal_distribution_and_squared_loss`

接下來，我們透過對噪聲分佈的假設來解讀平方損失目標函式。

常態分佈和線性迴歸之間的關係很密切。
常態分佈（normal distribution），也稱為*高斯分佈*（Gaussian distribution），
最早由德國數學家高斯（Gauss）應用於天文學研究。
簡單的說，若隨機變數$x$具有均值$\mu$和方差$\sigma^2$（標準差$\sigma$），其常態分佈機率密度函式如下：

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

下面[**我們定義一個Python函式來計算常態分佈**]。

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

我們現在(**視覺化常態分佈**)。

```{.python .input}
#@tab mxnet
# 再次使用numpy進行視覺化
x = np.arange(-7, 7, 0.01)
# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.python .input}
#@tab pytorch, tensorflow, paddle
# 再次使用numpy進行視覺化
x = np.arange(-7, 7, 0.01)

# 均值和標準差對
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

就像我們所看到的，改變均值會產生沿$x$軸的偏移，增加方差將會分散分佈、降低其峰值。

均方誤差損失函式（簡稱均方損失）可以用於線性迴歸的一個原因是：
我們假設了觀測中包含噪聲，其中噪聲服從常態分佈。
噪聲常態分佈如下式:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon,$$

其中，$\epsilon \sim \mathcal{N}(0, \sigma^2)$。

因此，我們現在可以寫出透過給定的$\mathbf{x}$觀測到特定$y$的*似然*（likelihood）：

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

現在，根據極大似然估計法，引數$\mathbf{w}$和$b$的最優值是使整個資料集的*似然*最大的值：

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

根據極大似然估計法選擇的估計量稱為*極大似然估計量*。
雖然使許多指數函式的乘積最大化看起來很困難，
但是我們可以在不改變目標的前提下，透過最大化似然對數來簡化。
由於歷史原因，最佳化通常是說最小化而不是最大化。
我們可以改為*最小化負對數似然*$-\log P(\mathbf y \mid \mathbf X)$。
由此可以得到的數學公式是：

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

現在我們只需要假設$\sigma$是某個固定常數就可以忽略第一項，
因為第一項不依賴於$\mathbf{w}$和$b$。
現在第二項除了常數$\frac{1}{\sigma^2}$外，其餘部分和前面介紹的均方誤差是一樣的。
幸運的是，上面式子的解並不依賴於$\sigma$。
因此，在高斯噪聲的假設下，最小化均方誤差等價於對線性模型的極大似然估計。

## 從線性迴歸到深度網路

到目前為止，我們只談論了線性模型。
儘管神經網路涵蓋了更多更為豐富的模型，我們依然可以用描述神經網路的方式來描述線性模型，
從而把線性模型看作一個神經網路。
首先，我們用『層』符號來重寫這個模型。

### 神經網路圖

深度學習從業者喜歡繪製圖表來視覺化模型中正在發生的事情。
在 :numref:`fig_single_neuron`中，我們將線性迴歸模型描述為一個神經網路。
需要注意的是，該圖只顯示連線模式，即只顯示每個輸入如何連線到輸出，隱去了權重和偏置的值。

![線性迴歸是一個單層神經網路。](../img/singleneuron.svg)
:label:`fig_single_neuron`

在 :numref:`fig_single_neuron`所示的神經網路中，輸入為$x_1, \ldots, x_d$，
因此輸入層中的*輸入數*（或稱為*特徵維度*，feature dimensionality）為$d$。
網路的輸出為$o_1$，因此輸出層中的*輸出數*是1。
需要注意的是，輸入值都是已經給定的，並且只有一個*計算*神經元。
由於模型重點在發生計算的地方，所以通常我們在計算層數時不考慮輸入層。
也就是說， :numref:`fig_single_neuron`中神經網路的*層數*為1。
我們可以將線性迴歸模型視為僅由單個人工神經元組成的神經網路，或稱為單層神經網路。

對於線性迴歸，每個輸入都與每個輸出（在本例中只有一個輸出）相連，
我們將這種變換（ :numref:`fig_single_neuron`中的輸出層）
稱為*全連線層*（fully-connected layer）或稱為*稠密層*（dense layer）。
下一章將詳細討論由這些層組成的網路。

### 生物學

線性迴歸發明的時間（1795年）早於計算神經科學，所以將線性迴歸描述為神經網路似乎不合適。
當控制學家、神經生物學家沃倫·麥函式庫洛奇和沃爾特·皮茨開始開發人工神經元模型時，
他們為什麼將線性模型作為一個起點呢？
我們來看一張圖片 :numref:`fig_Neuron`：
這是一張由*樹突*（dendrites，輸入終端）、
*細胞核*（nucleus，CPU）組成的生物神經元圖片。
*軸突*（axon，輸出線）和*軸突端子*（axon terminal，輸出端子）
透過*突觸*（synapse）與其他神經元連線。

![真實的神經元。](../img/neuron.svg)
:label:`fig_Neuron`

樹突中接收到來自其他神經元（或視網膜等環境傳感器）的訊息$x_i$。
該訊息透過*突觸權重*$w_i$來加權，以確定輸入的影響（即，透過$x_i w_i$相乘來啟用或抑制）。
來自多個源的加權輸入以加權和$y = \sum_i x_i w_i + b$的形式彙聚在細胞核中，
然後將這些訊息傳送到軸突$y$中進一步處理，通常會通過$\sigma(y)$進行一些非線性處理。
之後，它要麼到達目的地（例如肌肉），要麼透過樹突進入另一個神經元。

當然，許多這樣的單元可以透過正確連線和正確的學習演算法拼湊在一起，
從而產生的行為會比單獨一個神經元所產生的行為更有趣、更復雜，
這種想法歸功於我們對真實生物神經系統的研究。

當今大多數深度學習的研究幾乎沒有直接從神經科學中獲得靈感。
我們援引斯圖爾特·羅素和彼得·諾維格在他們的經典人工智慧教科書
*Artificial Intelligence:A Modern Approach* :cite:`Russell.Norvig.2016`
中所說的：雖然飛機可能受到鳥類的啟發，但幾個世紀以來，鳥類學並不是航空創新的主要驅動力。
同樣地，如今在深度學習中的靈感同樣或更多地來自數學、統計學和電腦科學。

## 小結

* 機器學習模型中的關鍵要素是訓練資料、損失函式、最佳化演算法，還有模型本身。
* 矢量化使數學表達上更簡潔，同時執行的更快。
* 最小化目標函式和執行極大似然估計等價。
* 線性迴歸模型也是一個簡單的神經網路。

## 練習

1. 假設我們有一些資料$x_1, \ldots, x_n \in \mathbb{R}$。我們的目標是找到一個常數$b$，使得最小化$\sum_i (x_i - b)^2$。
    1. 找到最優值$b$的解析解。
    1. 這個問題及其解與常態分佈有什麼關係?
1. 推導出使用平方誤差的線性迴歸最佳化問題的解析解。為了簡化問題，可以忽略偏置$b$（我們可以透過向$\mathbf X$新增所有值為1的一列來做到這一點）。
    1. 用矩陣和向量表示法寫出最佳化問題（將所有資料視為單個矩陣，將所有目標值視為單個向量）。
    1. 計算損失對$w$的梯度。
    1. 透過將梯度設為0、求解矩陣方程來找到解析解。
    1. 什麼時候可能比使用隨機梯度下降更好？這種方法何時會失效？
1. 假定控制附加噪聲$\epsilon$的噪聲模型是指數分佈。也就是說，$p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$
    1. 寫出模型$-\log P(\mathbf y \mid \mathbf X)$下資料的負對數似然。
    1. 請試著寫出解析解。
    1. 提出一種隨機梯度下降演算法來解決這個問題。哪裡可能出錯？（提示：當我們不斷更新引數時，在駐點附近會發生什麼情況）請嘗試解決這個問題。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1774)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1775)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1776)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11688)
:end_tab:
