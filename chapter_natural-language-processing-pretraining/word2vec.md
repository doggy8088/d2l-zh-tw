# 詞嵌入（word2vec）
:label:`sec_word2vec`

自然語言是用來表達人腦思維的複雜系統。
在這個系統中，詞是意義的基本單元。顧名思義，
*詞向量*是用於表示單詞意義的向量，
並且還可以被認為是單詞的特徵向量或表示。
將單詞對映到實向量的技術稱為*詞嵌入*。
近年來，詞嵌入逐漸成為自然語言處理的基礎知識。

## 為何獨熱向量是一個糟糕的選擇

在 :numref:`sec_rnn_scratch`中，我們使用獨熱向量來表示詞（字元就是單詞）。假設詞典中不同詞的數量（詞典大小）為$N$，每個詞對應一個從$0$到$N−1$的不同整數（索引）。為了得到索引為$i$的任意詞的獨熱向量表示，我們建立了一個全為0的長度為$N$的向量，並將位置$i$的元素設定為1。這樣，每個詞都被表示為一個長度為$N$的向量，可以直接由神經網路使用。

雖然獨熱向量很容易建立，但它們通常不是一個好的選擇。一個主要原因是獨熱向量不能準確表達不同詞之間的相似度，比如我們經常使用的『餘弦相似度』。對於向量$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，它們的餘弦相似度是它們之間角度的餘弦：

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

由於任意兩個不同詞的獨熱向量之間的餘弦相似度為0，所以獨熱向量不能編碼詞之間的相似性。

## 自監督的word2vec

[word2vec](https://code.google.com/archive/p/word2vec/)工具是為瞭解決上述問題而提出的。它將每個詞對映到一個固定長度的向量，這些向量能更好地表達不同詞之間的相似性和類比關係。word2vec工具包含兩個模型，即*跳元模型*（skip-gram） :cite:`Mikolov.Sutskever.Chen.ea.2013`和*連續詞袋*（CBOW） :cite:`Mikolov.Chen.Corrado.ea.2013`。對於在語義上有意義的表示，它們的訓練依賴於條件機率，條件機率可以被看作使用語料庫中一些詞來預測另一些單詞。由於是不帶標籤的資料，因此跳元模型和連續詞袋都是自監督模型。

下面，我們將介紹這兩種模式及其訓練方法。

## 跳元模型（Skip-Gram）
:label:`subsec_skip-gram`

跳元模型假設一個詞可以用來在文字序列中生成其周圍的單詞。以文字序列『the』『man』『loves』『his』『son』為例。假設*中心詞*選擇『loves』，並將上下文視窗設定為2，如圖 :numref:`fig_skip_gram`所示，給定中心詞『loves』，跳元模型考慮生成*上下文詞*『the』『man』『him』『son』的條件機率：

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

假設上下文詞是在給定中心詞的情況下獨立生成的（即條件獨立性）。在這種情況下，上述條件機率可以重寫為：

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![跳元模型考慮了在給定中心詞的情況下生成周圍上下文詞的條件機率](../img/skip-gram.svg)
:label:`fig_skip_gram`

在跳元模型中，每個詞都有兩個$d$維向量表示，用於計算條件機率。更具體地說，對於詞典中索引為$i$的任何詞，分別用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示其用作*中心詞*和*上下文詞*時的兩個向量。給定中心詞$w_c$（詞典中的索引$c$），生成任何上下文詞$w_o$（詞典中的索引$o$）的條件機率可以透過對向量點積的softmax操作來建模：

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

其中詞表索引集$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。給定長度為$T$的文字序列，其中時間步$t$處的詞表示為$w^{(t)}$。假設上下文詞是在給定任何中心詞的情況下獨立生成的。對於上下文視窗$m$，跳元模型的似然函式是在給定任何中心詞的情況下生成所有上下文詞的機率：

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

其中可以省略小於$1$或大於$T$的任何時間步。

### 訓練

跳元模型引數是詞表中每個詞的中心詞向量和上下文詞向量。在訓練中，我們透過最大化似然函式（即極大似然估計）來學習模型引數。這相當於最小化以下損失函式：

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

當使用隨機梯度下降來最小化損失時，在每次迭代中可以隨機抽樣一個較短的子序列來計算該子序列的（隨機）梯度，以更新模型引數。為了計算該（隨機）梯度，我們需要獲得對數條件機率關於中心詞向量和上下文詞向量的梯度。通常，根據 :eqref:`eq_skip-gram-softmax`，涉及中心詞$w_c$和上下文詞$w_o$的對數條件機率為：

$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

透過微分，我們可以獲得其相對於中心詞向量$\mathbf{v}_c$的梯度為

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`

注意， :eqref:`eq_skip-gram-grad`中的計算需要詞典中以$w_c$為中心詞的所有詞的條件機率。其他詞向量的梯度可以以相同的方式獲得。

對詞典中索引為$i$的詞進行訓練後，得到$\mathbf{v}_i$（作為中心詞）和$\mathbf{u}_i$（作為上下文詞）兩個詞向量。在自然語言處理應用中，跳元模型的中心詞向量通常用作詞表示。

## 連續詞袋（CBOW）模型

*連續詞袋*（CBOW）模型類似於跳元模型。與跳元模型的主要區別在於，連續詞袋模型假設中心詞是基於其在文字序列中的周圍上下文詞生成的。例如，在文字序列『the』『man』『loves』『his』『son』中，在『loves』為中心詞且上下文視窗為2的情況下，連續詞袋模型考慮基於上下文詞『the』『man』『him』『son』（如 :numref:`fig_cbow`所示）生成中心詞『loves』的條件機率，即：

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![連續詞袋模型考慮了給定周圍上下文詞生成中心詞條件機率](../img/cbow.svg)
:label:`fig_cbow`


由於連續詞袋模型中存在多個上下文詞，因此在計算條件機率時對這些上下文詞向量進行平均。具體地說，對於字典中索引$i$的任意詞，分別用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示用作*上下文*詞和*中心*詞的兩個向量（符號與跳元模型中相反）。給定上下文詞$w_{o_1}, \ldots, w_{o_{2m}}$（在詞表中索引是$o_1, \ldots, o_{2m}$）生成任意中心詞$w_c$（在詞表中索引是$c$）的條件機率可以由以下公式建模:

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`

為了簡潔起見，我們設為$\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$和$\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$。那麼 :eqref:`fig_cbow-full`可以簡化為：

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

給定長度為$T$的文字序列，其中時間步$t$處的詞表示為$w^{(t)}$。對於上下文視窗$m$，連續詞袋模型的似然函式是在給定其上下文詞的情況下生成所有中心詞的機率：

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### 訓練

訓練連續詞袋模型與訓練跳元模型幾乎是一樣的。連續詞袋模型的最大似然估計等價於最小化以下損失函式：

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

請注意，

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

透過微分，我們可以獲得其關於任意上下文詞向量$\mathbf{v}_{o_i}$（$i = 1, \ldots, 2m$）的梯度，如下：

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`

其他詞向量的梯度可以以相同的方式獲得。與跳元模型不同，連續詞袋模型通常使用上下文詞向量作為詞表示。

## 小結

* 詞向量是用於表示單詞意義的向量，也可以看作詞的特徵向量。將詞對映到實向量的技術稱為詞嵌入。
* word2vec工具包含跳元模型和連續詞袋模型。
* 跳元模型假設一個單詞可用於在文字序列中，生成其周圍的單詞；而連續詞袋模型假設基於上下文詞來生成中心單詞。

## 練習

1. 計算每個梯度的計算複雜度是多少？如果詞表很大，會有什麼問題呢？
1. 英語中的一些固定短語由多個單詞組成，例如『new york』。如何訓練它們的詞向量？提示:檢視word2vec論文的第四節 :cite:`Mikolov.Sutskever.Chen.ea.2013`。
1. 讓我們以跳元模型為例來思考word2vec設計。跳元模型中兩個詞向量的點積與餘弦相似度之間有什麼關係？對於語義相似的一對詞，為什麼它們的詞向量（由跳元模型訓練）的餘弦相似度可能很高？

[Discussions](https://discuss.d2l.ai/t/5744)
