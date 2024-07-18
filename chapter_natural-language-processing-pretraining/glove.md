# 全域向量的詞嵌入（GloVe）
:label:`sec_glove`

上下文視窗內的詞共現可以攜帶豐富的語義訊息。例如，在一個大型語料庫中，『固體』比『氣體』更有可能與『冰』共現，但『氣體』一詞與『蒸汽』的共現頻率可能比與『冰』的共現頻率更高。此外，可以預先計算此類共現的全域語料庫統計資料：這可以提高訓練效率。為了利用整個語料庫中的統計訊息進行詞嵌入，讓我們首先回顧 :numref:`subsec_skip-gram`中的跳元模型，但是使用全域語料庫統計（如共現計數）來解釋它。

## 帶全域語料統計的跳元模型
:label:`subsec_skipgram-global`

用$q_{ij}$表示詞$w_j$的條件機率$P(w_j\mid w_i)$，在跳元模型中給定詞$w_i$，我們有：

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

其中，對於任意索引$i$，向量$\mathbf{v}_i$和$\mathbf{u}_i$分別表示詞$w_i$作為中心詞和上下文詞，且$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$是詞表的索引集。

考慮詞$w_i$可能在語料庫中出現多次。在整個語料庫中，所有以$w_i$為中心詞的上下文詞形成一個詞索引的*多重集*$\mathcal{C}_i$，該索引允許同一元素的多個實例。對於任何元素，其實例數稱為其*重數*。舉例說明，假設詞$w_i$在語料庫中出現兩次，並且在兩個上下文視窗中以$w_i$為其中心詞的上下文詞索引是$k, j, m, k$和$k, l, k, j$。因此，多重集$\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$，其中元素$j, k, l, m$的重數分別為2、4、1、1。

現在，讓我們將多重集$\mathcal{C}_i$中的元素$j$的重數表示為$x_{ij}$。這是詞$w_j$（作為上下文詞）和詞$w_i$（作為中心詞）在整個語料庫的同一上下文視窗中的全域共現計數。使用這樣的全域語料庫統計，跳元模型的損失函式等價於：

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

我們用$x_i$表示上下文視窗中的所有上下文詞的數量，其中$w_i$作為它們的中心詞出現，這相當於$|\mathcal{C}_i|$。設$p_{ij}$為用於生成上下文詞$w_j$的條件機率$x_{ij}/x_i$。給定中心詞$w_i$， :eqref:`eq_skipgram-x_ij`可以重寫為：

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

在 :eqref:`eq_skipgram-p_ij`中，$-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$計算全域語料統計的條件分佈$p_{ij}$和模型預測的條件分佈$q_{ij}$的交叉熵。如上所述，這一損失也按$x_i$加權。在 :eqref:`eq_skipgram-p_ij`中最小化損失函式將使預測的條件分佈接近全域語料庫統計中的條件分佈。

雖然交叉熵損失函式通常用於測量機率分佈之間的距離，但在這裡可能不是一個好的選擇。一方面，正如我們在 :numref:`sec_approx_train`中提到的，規範化$q_{ij}$的代價在於整個詞表的求和，這在計算上可能非常昂貴。另一方面，來自大型語料庫的大量罕見事件往往被交叉熵損失建模，從而賦予過多的權重。

## GloVe模型

有鑑於此，*GloVe*模型基於平方損失 :cite:`Pennington.Socher.Manning.2014`對跳元模型做了三個修改：

1. 使用變數$p'_{ij}=x_{ij}$和$q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$
而非機率分佈，並取兩者的對數。所以平方損失項是$\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$。
2. 為每個詞$w_i$新增兩個標量模型引數：中心詞偏置$b_i$和上下文詞偏置$c_i$。
3. 用權重函式$h(x_{ij})$替換每個損失項的權重，其中$h(x)$在$[0, 1]$的間隔內遞增。

整合程式碼，訓練GloVe是為了儘量降低以下損失函式：

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

對於權重函式，建議的選擇是：當$x < c$（例如，$c = 100$）時，$h(x) = (x/c) ^\alpha$（例如$\alpha = 0.75$）；否則$h(x) = 1$。在這種情況下，由於$h(0)=0$，為了提高計算效率，可以省略任意$x_{ij}=0$的平方損失項。例如，當使用小批次隨機梯度下降進行訓練時，在每次迭代中，我們隨機抽樣一小批次*非零*的$x_{ij}$來計算梯度並更新模型引數。注意，這些非零的$x_{ij}$是預先計算的全域語料庫統計資料；因此，該模型GloVe被稱為*全域向量*。

應該強調的是，當詞$w_i$出現在詞$w_j$的上下文視窗時，詞$w_j$也出現在詞$w_i$的上下文視窗。因此，$x_{ij}=x_{ji}$。與擬合非對稱條件機率$p_{ij}$的word2vec不同，GloVe擬合對稱機率$\log \, x_{ij}$。因此，在GloVe模型中，任意詞的中心詞向量和上下文詞向量在數學上是等價的。但在實際應用中，由於初始值不同，同一個詞經過訓練後，在這兩個向量中可能得到不同的值：GloVe將它們相加作為輸出向量。

## 從條件機率比值理解GloVe模型

我們也可以從另一個角度來理解GloVe模型。使用 :numref:`subsec_skipgram-global`中的相同符號，設$p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$為生成上下文詞$w_j$的條件機率，給定$w_i$作為語料庫中的中心詞。 :numref:`tab_glove`根據大量語料庫的統計資料，列出了給定單詞『ice』和『steam』的共現機率及其比值。

大型語料庫中的詞-詞共現機率及其比值（根據 :cite:`Pennington.Socher.Manning.2014`中的表1改編）

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

從 :numref:`tab_glove`中，我們可以觀察到以下幾點：

* 對於與『ice』相關但與『steam』無關的單詞$w_k$，例如$w_k=\text{solid}$，我們預計會有更大的共現機率比值，例如8.9。
* 對於與『steam』相關但與『ice』無關的單詞$w_k$，例如$w_k=\text{gas}$，我們預計較小的共現機率比值，例如0.085。
* 對於同時與『ice』和『steam』相關的單詞$w_k$，例如$w_k=\text{water}$，我們預計其共現機率的比值接近1，例如1.36.
* 對於與『ice』和『steam』都不相關的單詞$w_k$，例如$w_k=\text{fashion}$，我們預計共現機率的比值接近1，例如0.96.

由此可見，共現機率的比值能夠直觀地表達詞與詞之間的關係。因此，我們可以設計三個詞向量的函式來擬合這個比值。對於共現機率${p_{ij}}/{p_{ik}}$的比值，其中$w_i$是中心詞，$w_j$和$w_k$是上下文詞，我們希望使用某個函式$f$來擬合該比值：

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

在$f$的許多可能的設計中，我們只在以下幾點中選擇了一個合理的選擇。因為共現機率的比值是標量，所以我們要求$f$是標量函式，例如$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$。在 :eqref:`eq_glove-f`中交換詞索引$j$和$k$，它必須保持$f(x)f(-x)=1$，所以一種可能性是$f(x)=\exp(x)$，即：

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

現在讓我們選擇$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$，其中$\alpha$是常數。從$p_{ij}=x_{ij}/x_i$開始，取兩邊的對數得到$\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$。我們可以使用附加的偏置項來擬合$- \log\, \alpha + \log\, x_i$，如中心詞偏置$b_i$和上下文詞偏置$c_j$：

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

透過對 :eqref:`eq_glove-square`的加權平方誤差的度量，得到了 :eqref:`eq_glove-loss`的GloVe損失函式。

## 小結

* 諸如詞-詞共現計數的全域語料庫統計可以來解釋跳元模型。
* 交叉熵損失可能不是衡量兩種機率分佈差異的好選擇，特別是對於大型語料庫。GloVe使用平方損失來擬合預先計算的全域語料庫統計資料。
* 對於GloVe中的任意詞，中心詞向量和上下文詞向量在數學上是等價的。
* GloVe可以從詞-詞共現機率的比率來解釋。

## 練習

1. 如果詞$w_i$和$w_j$在同一上下文視窗中同時出現，我們如何使用它們在文字序列中的距離來重新設計計算條件機率$p_{ij}$的方法？提示：參見GloVe論文 :cite:`Pennington.Socher.Manning.2014`的第4.2節。
1. 對於任何一個詞，它的中心詞偏置和上下文偏置在數學上是等價的嗎？為什麼？

[Discussions](https://discuss.d2l.ai/t/5736)
