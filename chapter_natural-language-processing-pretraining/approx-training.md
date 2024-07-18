# 近似訓練
:label:`sec_approx_train`

回想一下我們在 :numref:`sec_word2vec`中的討論。跳元模型的主要思想是使用softmax運算來計算基於給定的中心詞$w_c$生成上下文字$w_o$的條件機率（如 :eqref:`eq_skip-gram-softmax`），對應的對數損失在 :eqref:`eq_skip-gram-log`給出。

由於softmax操作的性質，上下文詞可以是詞表$\mathcal{V}$中的任意項， :eqref:`eq_skip-gram-log`包含與整個詞表大小一樣多的項的求和。因此， :eqref:`eq_skip-gram-grad`中跳元模型的梯度計算和 :eqref:`eq_cbow-gradient`中的連續詞袋模型的梯度計算都包含求和。不幸的是，在一個詞典上（通常有幾十萬或數百萬個單詞）求和的梯度的計算成本是巨大的！

為了降低上述計算複雜度，本節將介紹兩種近似訓練方法：*負取樣*和*分層softmax*。
由於跳元模型和連續詞袋模型的相似性，我們將以跳元模型為例來描述這兩種近似訓練方法。

## 負取樣
:label:`subsec_negative-sampling`

負取樣修改了原目標函式。給定中心詞$w_c$的上下文視窗，任意上下文詞$w_o$來自該上下文視窗的被認為是由下式建模機率的事件：

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

其中$\sigma$使用了sigmoid啟用函式的定義：

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

讓我們從最大化文字序列中所有這些事件的聯合機率開始訓練詞嵌入。具體而言，給定長度為$T$的文字序列，以$w^{(t)}$表示時間步$t$的詞，並使上下文視窗為$m$，考慮最大化聯合機率：

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

然而， :eqref:`eq-negative-sample-pos`只考慮那些正樣本的事件。僅當所有詞向量都等於無窮大時， :eqref:`eq-negative-sample-pos`中的聯合機率才最大化為1。當然，這樣的結果毫無意義。為了使目標函式更有意義，*負取樣*新增從預定義分佈中取樣的負樣本。

用$S$表示上下文詞$w_o$來自中心詞$w_c$的上下文視窗的事件。對於這個涉及$w_o$的事件，從預定義分佈$P(w)$中取樣$K$個不是來自這個上下文視窗*噪聲詞*。用$N_k$表示噪聲詞$w_k$（$k=1, \ldots, K$）不是來自$w_c$的上下文視窗的事件。假設正例和負例$S, N_1, \ldots, N_K$的這些事件是相互獨立的。負取樣將 :eqref:`eq-negative-sample-pos`中的聯合機率（僅涉及正例）重寫為

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

透過事件$S, N_1, \ldots, N_K$近似條件機率：

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

分別用$i_t$和$h_k$表示詞$w^{(t)}$和噪聲詞$w_k$在文字序列的時間步$t$處的索引。 :eqref:`eq-negative-sample-conditional-prob`中關於條件機率的對數損失為：

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

我們可以看到，現在每個訓練步的梯度計算成本與詞表大小無關，而是線性依賴於$K$。當將超引數$K$設定為較小的值時，在負取樣的每個訓練步處的梯度的計算成本較小。

## 層序Softmax

作為另一種近似訓練方法，*層序Softmax*（hierarchical softmax）使用二叉樹（ :numref:`fig_hi_softmax`中說明的資料結構），其中樹的每個葉節點表示詞表$\mathcal{V}$中的一個詞。

![用於近似訓練的分層softmax，其中樹的每個葉節點表示詞表中的一個詞](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

用$L(w)$表示二叉樹中表示字$w$的從根節點到葉節點的路徑上的節點數（包括兩端）。設$n(w,j)$為該路徑上的$j^\mathrm{th}$節點，其上下文字向量為$\mathbf{u}_{n(w, j)}$。例如， :numref:`fig_hi_softmax`中的$L(w_3) = 4$。分層softmax將 :eqref:`eq_skip-gram-softmax`中的條件機率近似為

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

其中函式$\sigma$在 :eqref:`eq_sigma-f`中定義，$\text{leftChild}(n)$是節點$n$的左子節點：如果$x$為真，$[\![x]\!] = 1$;否則$[\![x]\!] = -1$。

為了說明，讓我們計算 :numref:`fig_hi_softmax`中給定詞$w_c$生成詞$w_3$的條件機率。這需要$w_c$的詞向量$\mathbf{v}_c$和從根到$w_3$的路徑（ :numref:`fig_hi_softmax`中加粗的路徑）上的非葉節點向量之間的點積，該路徑依次向左、向右和向左遍歷：

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

由$\sigma(x)+\sigma(-x) = 1$，它認為基於任意詞$w_c$生成詞表$\mathcal{V}$中所有詞的條件機率總和為1：

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

幸運的是，由於二叉樹結構，$L(w_o)-1$大約與$\mathcal{O}(\text{log}_2|\mathcal{V}|)$是一個數量級。當詞表大小$\mathcal{V}$很大時，與沒有近似訓練的相比，使用分層softmax的每個訓練步的計算代價顯著降低。

## 小結

* 負取樣透過考慮相互獨立的事件來構造損失函式，這些事件同時涉及正例和負例。訓練的計算量與每一步的噪聲詞數成線性關係。
* 分層softmax使用二叉樹中從根節點到葉節點的路徑構造損失函式。訓練的計算成本取決於詞表大小的對數。

## 練習

1. 如何在負取樣中對噪聲詞進行取樣？
1. 驗證 :eqref:`eq_hi-softmax-sum-one`是否有效。
1. 如何分別使用負取樣和分層softmax訓練連續詞袋模型？

[Discussions](https://discuss.d2l.ai/t/5741)
