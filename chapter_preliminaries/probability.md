# 機率
:label:`sec_prob`

簡單地說，機器學習就是做出預測。

根據病人的臨床病史，我們可能想預測他們在下一年心臟病發作的*機率*。
在飛機噴氣發動機的異常檢測中，我們想要評估一組引擎讀數為正常執行情況的機率有多大。
在強化學習中，我們希望智慧體（agent）能在一個環境中智慧地行動。
這意味著我們需要考慮在每種可行的行為下獲得高獎勵的機率。
當我們建立推薦系統時，我們也需要考慮機率。
例如，假設我們為一家大型線上書店工作，我們可能希望估計某些使用者購買特定圖書的機率。
為此，我們需要使用機率學。
有完整的課程、專業、論文、職業、甚至院系，都致力於機率學的工作。
所以很自然地，我們在這部分的目標不是教授整個科目。
相反，我們希望教給讀者基礎的機率知識，使讀者能夠開始建立第一個深度學習模型，
以便讀者可以開始自己探索它。

現在讓我們更認真地考慮第一個例子：根據照片區分貓和狗。
這聽起來可能很簡單，但對於機器卻可能是一個艱鉅的挑戰。
首先，問題的難度可能取決於影象的解析度。

![不同解析度的影象 ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, 和 $160 \times 160$ pixels)](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

如 :numref:`fig_cat_dog`所示，雖然人類很容易以$160 \times 160$畫素的解析度識別貓和狗，
但它在$40\times40$畫素上變得具有挑戰性，而且在$10 \times 10$畫素下幾乎是不可能的。
換句話說，我們在很遠的距離（從而降低解析度）區分貓和狗的能力可能會變為猜測。
機率給了我們一種正式的途徑來說明我們的確定性水平。
如果我們完全肯定影象是一隻貓，我們說標籤$y$是"貓"的*機率*，表示為$P(y=$"貓"$)$等於$1$。
如果我們沒有證據表明$y=$『貓』或$y=$『狗』，那麼我們可以說這兩種可能性是相等的，
即$P(y=$"貓"$)=P(y=$"狗"$)=0.5$。
如果我們不十分確定影象描繪的是一隻貓，我們可以將機率賦值為$0.5<P(y=$"貓"$)<1$。

現在考慮第二個例子：給出一些天氣監測資料，我們想預測明天北京下雨的機率。
如果是夏天，下雨的機率是0.5。

在這兩種情況下，我們都不確定結果，但這兩種情況之間有一個關鍵區別。
在第一種情況中，影象實際上是狗或貓二選一。
在第二種情況下，結果實際上是一個隨機的事件。
因此，機率是一種靈活的語言，用於說明我們的確定程度，並且它可以有效地應用於廣泛的領域中。

## 基本機率論

假設我們擲骰子，想知道看到1的機率有多大，而不是看到另一個數字。
如果骰子是公平的，那麼所有六個結果$\{1, \ldots, 6\}$都有相同的可能發生，
因此我們可以說$1$發生的機率為$\frac{1}{6}$。

然而現實生活中，對於我們從工廠收到的真實骰子，我們需要檢查它是否有瑕疵。
檢查骰子的唯一方法是多次投擲並記錄結果。
對於每個骰子，我們將觀察到$\{1, \ldots, 6\}$中的一個值。
對於每個值，一種自然的方法是將它出現的次數除以投擲的總次數，
即此*事件*（event）機率的*估計值*。
*大數定律*（law of large numbers）告訴我們：
隨著投擲次數的增加，這個估計值會越來越接近真實的潛在機率。
讓我們用程式碼試一試！

首先，我們匯入必要的軟體套件。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import random
import numpy as np
```

在統計學中，我們把從機率分佈中抽取樣本的過程稱為*抽樣*（sampling）。
籠統來說，可以把*分佈*（distribution）看作對事件的機率分配，
稍後我們將給出的更正式定義。
將機率分配給一些離散選擇的分佈稱為*多項分佈*（multinomial distribution）。

為了抽取一個樣本，即擲骰子，我們只需傳入一個機率向量。
輸出是另一個相同長度的向量：它在索引$i$處的值是取樣結果中$i$出現的次數。

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab paddle
fair_probs = [1.0 / 6] * 6
paddle.distribution.Multinomial(1, paddle.to_tensor(fair_probs)).sample()
```

在估計一個骰子的公平性時，我們希望從同一分佈中生成多個樣本。
如果用Python的for迴圈來完成這個任務，速度會慢得驚人。
因此我們使用深度學習框架的函式同時抽取多個樣本，得到我們想要的任意形狀的獨立樣本數組。

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab paddle
paddle.distribution.Multinomial(10, paddle.to_tensor(fair_probs)).sample()
```

現在我們知道如何對骰子進行取樣，我們可以模擬1000次投擲。
然後，我們可以統計1000次投擲後，每個數字被投中了多少次。
具體來說，我們計算相對頻率，以作為真實機率的估計。

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# 將結果儲存為32位浮點數以進行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相對頻率作為估計值
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

```{.python .input}
#@tab paddle
counts = paddle.distribution.Multinomial(1000, paddle.to_tensor(fair_probs)).sample()
counts / 1000
```

因為我們是從一個公平的骰子中生成的資料，我們知道每個結果都有真實的機率$\frac{1}{6}$，
大約是$0.167$，所以上面輸出的估計值看起來不錯。

我們也可以看到這些機率如何隨著時間的推移收斂到真實機率。
讓我們進行500組實驗，每組抽取10個樣本。

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab paddle
counts = paddle.distribution.Multinomial(10, paddle.to_tensor(fair_probs)).sample((500,1))
cum_counts = counts.cumsum(axis=0)
cum_counts = cum_counts.squeeze(axis=1)
estimates = cum_counts / cum_counts.sum(axis=1, keepdim=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i],
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
```

每條實線對應於骰子的6個值中的一個，並給出骰子在每組實驗後出現值的估計機率。
當我們透過更多的實驗獲得更多的資料時，這$6$條實體曲線向真實機率收斂。

### 機率論公理

在處理骰子擲出時，我們將集合$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$
稱為*樣本空間*（sample space）或*結果空間*（outcome space），
其中每個元素都是*結果*（outcome）。
*事件*（event）是一組給定樣本空間的隨機結果。
例如，『看到$5$』（$\{5\}$）和『看到奇數』（$\{1, 3, 5\}$）都是擲出骰子的有效事件。
注意，如果一個隨機實驗的結果在$\mathcal{A}$中，則事件$\mathcal{A}$已經發生。
也就是說，如果投擲出$3$點，因為$3 \in \{1, 3, 5\}$，我們可以說，『看到奇數』的事件發生了。

*機率*（probability）可以被認為是將集合對映到真實值的函式。
在給定的樣本空間$\mathcal{S}$中，事件$\mathcal{A}$的機率，
表示為$P(\mathcal{A})$，滿足以下屬性：

* 對於任意事件$\mathcal{A}$，其機率從不會是負數，即$P(\mathcal{A}) \geq 0$；
* 整個樣本空間的機率為$1$，即$P(\mathcal{S}) = 1$；
* 對於*互斥*（mutually exclusive）事件（對於所有$i \neq j$都有$\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$）的任意一個可數序列$\mathcal{A}_1, \mathcal{A}_2, \ldots$，序列中任意一個事件發生的機率等於它們各自發生的機率之和，即$P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$。

以上也是機率論的公理，由科爾莫戈羅夫於1933年提出。
有了這個公理系統，我們可以避免任何關於隨機性的哲學爭論；
相反，我們可以用數學語言嚴格地推理。
例如，假設事件$\mathcal{A}_1$為整個樣本空間，
且當所有$i > 1$時的$\mathcal{A}_i = \emptyset$，
那麼我們可以證明$P(\emptyset) = 0$，即不可能發生事件的機率是$0$。

### 隨機變數

在我們擲骰子的隨機實驗中，我們引入了*隨機變數*（random variable）的概念。
隨機變數幾乎可以是任何數量，並且它可以在隨機實驗的一組可能性中取一個值。
考慮一個隨機變數$X$，其值在擲骰子的樣本空間$\mathcal{S}=\{1,2,3,4,5,6\}$中。
我們可以將事件『看到一個$5$』表示為$\{X=5\}$或$X=5$，
其機率表示為$P(\{X=5\})$或$P(X=5)$。
透過$P(X=a)$，我們區分了隨機變數$X$和$X$可以採取的值（例如$a$）。
然而，這可能會導致繁瑣的表示。
為了簡化符號，一方面，我們可以將$P(X)$表示為隨機變數$X$上的*分佈*（distribution）：
分佈告訴我們$X$獲得某一值的機率。
另一方面，我們可以簡單用$P(a)$表示隨機變數取值$a$的機率。
由於機率論中的事件是來自樣本空間的一組結果，因此我們可以為隨機變數指定值的可取範圍。
例如，$P(1 \leq X \leq 3)$表示事件$\{1 \leq X \leq 3\}$，
即$\{X = 1, 2, \text{or}, 3\}$的機率。
等價地，$P(1 \leq X \leq 3)$表示隨機變數$X$從$\{1, 2, 3\}$中取值的機率。

請注意，*離散*（discrete）隨機變數（如骰子的每一面）
和*連續*（continuous）隨機變數（如人的體重和身高）之間存在微妙的區別。
現實生活中，測量兩個人是否具有完全相同的身高沒有太大意義。
如果我們進行足夠精確的測量，最終會發現這個星球上沒有兩個人具有完全相同的身高。
在這種情況下，詢問某人的身高是否落入給定的區間，比如是否在1.79米和1.81米之間更有意義。
在這些情況下，我們將這個看到某個數值的可能性量化為*密度*（density）。
高度恰好為1.80米的機率為0，但密度不是0。
在任何兩個不同高度之間的區間，我們都有非零的機率。
在本節的其餘部分中，我們將考慮離散空間中的機率。
連續隨機變數的機率可以參考深度學習數學附錄中[隨機變數](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/random-variables.html)
的一節。

## 處理多個隨機變數

很多時候，我們會考慮多個隨機變數。
比如，我們可能需要對疾病和症狀之間的關係進行建模。
給定一個疾病和一個症狀，比如『流感』和『咳嗽』，以某個機率存在或不存在於某個患者身上。
我們需要估計這些機率以及機率之間的關係，以便我們可以運用我們的推斷來實現更好的醫療服務。

再舉一個更復雜的例子：影象包含數百萬畫素，因此有數百萬個隨機變數。
在許多情況下，影象會附帶一個*標籤*（label），標識影象中的物件。
我們也可以將標籤視為一個隨機變數。
我們甚至可以將所有元資料視為隨機變數，例如位置、時間、光圈、焦距、ISO、對焦距離和相機型別。
所有這些都是聯合發生的隨機變數。
當我們處理多個隨機變數時，會有若干個變數是我們感興趣的。

### 聯合機率

第一個被稱為*聯合機率*（joint probability）$P(A=a,B=b)$。
給定任意值$a$和$b$，聯合機率可以回答：$A=a$和$B=b$同時滿足的機率是多少？
請注意，對於任何$a$和$b$的取值，$P(A = a, B=b) \leq P(A=a)$。
這點是確定的，因為要同時發生$A=a$和$B=b$，$A=a$就必須發生，$B=b$也必須發生（反之亦然）。因此，$A=a$和$B=b$同時發生的可能性不大於$A=a$或是$B=b$單獨發生的可能性。

### 條件機率

聯合機率的不等式帶給我們一個有趣的比率：
$0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$。
我們稱這個比率為*條件機率*（conditional probability），
並用$P(B=b \mid A=a)$表示它：它是$B=b$的機率，前提是$A=a$已發生。

### 貝葉斯定理

使用條件機率的定義，我們可以得出統計學中最有用的方程之一：
*Bayes定理*（Bayes' theorem）。
根據*乘法法則*（multiplication rule ）可得到$P(A, B) = P(B \mid A) P(A)$。
根據對稱性，可得到$P(A, B) = P(A \mid B) P(B)$。
假設$P(B)>0$，求解其中一個條件變數，我們得到

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

請注意，這裡我們使用緊湊的表示法：
其中$P(A, B)$是一個*聯合分佈*（joint distribution），
$P(A \mid B)$是一個*條件分佈*（conditional distribution）。
這種分佈可以在給定值$A = a, B=b$上進行求值。

### 邊際化

為了能進行事件機率求和，我們需要*求和法則*（sum rule），
即$B$的機率相當於計算$A$的所有可能選擇，並將所有選擇的聯合機率聚合在一起：

$$P(B) = \sum_{A} P(A, B),$$

這也稱為*邊際化*（marginalization）。
邊際化結果的機率或分佈稱為*邊際機率*（marginal probability）
或*邊際分佈*（marginal distribution）。

### 獨立性

另一個有用屬性是*依賴*（dependence）與*獨立*（independence）。
如果兩個隨機變數$A$和$B$是獨立的，意味著事件$A$的發生跟$B$事件的發生無關。
在這種情況下，統計學家通常將這一點表述為$A \perp  B$。
根據貝葉斯定理，馬上就能同樣得到$P(A \mid B) = P(A)$。
在所有其他情況下，我們稱$A$和$B$依賴。
比如，兩次連續丟擲一個骰子的事件是相互獨立的。
相比之下，燈開關的位置和房間的亮度並不是（因為可能存在燈泡壞掉、電源故障，或者開關故障）。

由於$P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$等價於$P(A, B) = P(A)P(B)$，
因此兩個隨機變數是獨立的，若且唯若兩個隨機變數的聯合分佈是其各自分佈的乘積。
同樣地，給定另一個隨機變數$C$時，兩個隨機變數$A$和$B$是*條件獨立的*（conditionally independent），
若且唯若$P(A, B \mid C) = P(A \mid C)P(B \mid C)$。
這個情況表示為$A \perp B \mid C$。

### 應用
:label:`subsec_probability_hiv_app`

我們實戰演練一下！
假設一個醫生對患者進行愛滋病病毒（HIV）測試。
這個測試是相當準確的，如果患者健康但測試顯示他患病，這個機率只有1%；
如果患者真正感染HIV，它永遠不會檢測不出。
我們使用$D_1$來表示診斷結果（如果陽性，則為$1$，如果陰性，則為$0$），
$H$來表示感染愛滋病病毒的狀態（如果陽性，則為$1$，如果陰性，則為$0$）。
在 :numref:`conditional_prob_D1`中列出了這樣的條件機率。

:條件機率為$P(D_1 \mid H)$

| 條件機率 | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

請注意，每列的加和都是1（但每行的加和不是），因為條件機率需要總和為1，就像機率一樣。
讓我們計算如果測試出來呈陽性，患者感染HIV的機率，即$P(H = 1 \mid D_1 = 1)$。
顯然，這將取決於疾病有多常見，因為它會影響錯誤警報的數量。
假設人口總體是相當健康的，例如，$P(H=1) = 0.0015$。
為了應用貝葉斯定理，我們需要運用邊際化和乘法法則來確定

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$
因此，我們得到

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

換句話說，儘管使用了非常準確的測試，患者實際上患有愛滋病的機率只有13.06%。
正如我們所看到的，機率可能是違反直覺的。

患者在收到這樣可怕的訊息後應該怎麼辦？
很可能，患者會要求醫生進行另一次測試來確定病情。
第二個測試具有不同的屬性，它不如第一個測試那麼精確，
如 :numref:`conditional_prob_D2`所示。

:條件機率為$P(D_2 \mid H)$

| 條件機率 | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

不幸的是，第二次測試也顯示陽性。讓我們透過假設條件獨立性來計算出應用Bayes定理的必要機率：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$
現在我們可以應用邊際化和乘法規則：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

最後，鑑於存在兩次陽性檢測，患者患有愛滋病的機率為

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

也就是說，第二次測試使我們能夠對患病的情況獲得更高的信心。
儘管第二次檢驗比第一次檢驗的準確性要低得多，但它仍然顯著提高我們的預測機率。

## 期望和方差

為了概括機率分佈的關鍵特徵，我們需要一些測量方法。
一個隨機變數$X$的*期望*（expectation，或平均值（average））表示為

$$E[X] = \sum_{x} x P(X = x).$$

當函式$f(x)$的輸入是從分佈$P$中抽取的隨機變數時，$f(x)$的期望值為

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

在許多情況下，我們希望衡量隨機變數$X$與其期望值的偏置。這可以透過方差來量化

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

方差的平方根被稱為*標準差*（standard deviation）。
隨機變數函式的方差衡量的是：當從該隨機變數分佈中取樣不同值$x$時，
函式值偏離該函式的期望的程度：

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## 小結

* 我們可以從機率分佈中取樣。
* 我們可以使用聯合分佈、條件分佈、Bayes定理、邊緣化和獨立性假設來分析多個隨機變數。
* 期望和方差為機率分佈的關鍵特徵的概括提供了實用的度量形式。

## 練習

1. 進行$m=500$組實驗，每組抽取$n=10$個樣本。改變$m$和$n$，觀察和分析實驗結果。
2. 給定兩個機率為$P(\mathcal{A})$和$P(\mathcal{B})$的事件，計算$P(\mathcal{A} \cup \mathcal{B})$和$P(\mathcal{A} \cap \mathcal{B})$的上限和下限。（提示：使用[友元圖](https://en.wikipedia.org/wiki/Venn_diagram)來展示這些情況。)
3. 假設我們有一系列隨機變數，例如$A$、$B$和$C$，其中$B$只依賴於$A$，而$C$只依賴於$B$，能簡化聯合機率$P(A, B, C)$嗎？（提示：這是一個[馬爾可夫鏈](https://en.wikipedia.org/wiki/Markov_chain)。)
4. 在 :numref:`subsec_probability_hiv_app`中，第一個測試更準確。為什麼不執行第一個測試兩次，而是同時執行第一個和第二個測試?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1761)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1762)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1760)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11685)
:end_tab:
