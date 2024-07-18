# 透過時間反向傳播
:label:`sec_bptt`

到目前為止，我們已經反覆提到像*梯度爆炸*或*梯度消失*，
以及需要對迴圈神經網路*分離梯度*。
例如，在 :numref:`sec_rnn_scratch`中，
我們在序列上呼叫了`detach`函式。
為了能夠快速建立模型並瞭解其工作原理，
上面所說的這些概念都沒有得到充分的解釋。
本節將更深入地探討序列模型反向傳播的細節，
以及相關的數學原理。

當我們首次實現迴圈神經網路（ :numref:`sec_rnn_scratch`）時，
遇到了梯度爆炸的問題。
如果做了練習題，就會發現梯度截斷對於確保模型收斂至關重要。
為了更好地理解此問題，本節將回顧序列模型梯度的計算方式，
它的工作原理沒有什麼新概念，畢竟我們使用的仍然是鏈式法則來計算梯度。

我們在 :numref:`sec_backprop`中描述了多層感知機中的
前向與反向傳播及相關的計算圖。
迴圈神經網路中的前向傳播相對簡單。
*透過時間反向傳播*（backpropagation through time，BPTT）
 :cite:`Werbos.1990`實際上是迴圈神經網路中反向傳播技術的一個特定應用。
它要求我們將迴圈神經網路的計算圖一次展開一個時間步，
以獲得模型變數和引數之間的依賴關係。
然後，基於鏈式法則，應用反向傳播來計算和儲存梯度。
由於序列可能相當長，因此依賴關係也可能相當長。
例如，某個1000個字元的序列，
其第一個詞元可能會對最後位置的詞元產生重大影響。
這在計算上是不可行的（它需要的時間和記憶體都太多了），
並且還需要超過1000個矩陣的乘積才能得到非常難以捉摸的梯度。
這個過程充滿了計算與統計的不確定性。
在下文中，我們將闡明會發生什麼以及如何在實踐中解決它們。

## 迴圈神經網路的梯度分析
:label:`subsec_bptt_analysis`

我們從一個描述迴圈神經網路工作原理的簡化模型開始，
此模型忽略了隱狀態的屬性及其更新方式的細節。
這裡的數學表示沒有像過去那樣明確地區分標量、向量和矩陣，
因為這些細節對於分析並不重要，
反而只會使本小節中的符號變得混亂。

在這個簡化模型中，我們將時間步$t$的隱狀態表示為$h_t$，
輸入表示為$x_t$，輸出表示為$o_t$。
回想一下我們在 :numref:`subsec_rnn_w_hidden_states`中的討論，
輸入和隱狀態可以拼接後與隱藏層中的一個權重變數相乘。
因此，我們分別使用$w_h$和$w_o$來表示隱藏層和輸出層的權重。
每個時間步的隱狀態和輸出可以寫為：

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

其中$f$和$g$分別是隱藏層和輸出層的變換。
因此，我們有一個鏈
$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$，
它們透過迴圈計算彼此依賴。
前向傳播相當簡單，一次一個時間步的遍歷三元組$(x_t, h_t, o_t)$，
然後透過一個目標函式在所有$T$個時間步內
評估輸出$o_t$和對應的標籤$y_t$之間的差異：

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

對於反向傳播，問題則有點棘手，
特別是當我們計算目標函式$L$關於引數$w_h$的梯度時。
具體來說，按照鏈式法則：

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

在 :eqref:`eq_bptt_partial_L_wh`中乘積的第一項和第二項很容易計算，
而第三項$\partial h_t/\partial w_h$是使事情變得棘手的地方，
因為我們需要迴圈地計算引數$w_h$對$h_t$的影響。
根據 :eqref:`eq_bptt_ht_ot`中的遞迴計算，
$h_t$既依賴於$h_{t-1}$又依賴於$w_h$，
其中$h_{t-1}$的計算也依賴於$w_h$。
因此，使用鏈式法則產生：

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

為了匯出上述梯度，假設我們有三個序列$\{a_{t}\},\{b_{t}\},\{c_{t}\}$，
當$t=1,2,\ldots$時，序列滿足$a_{0}=0$且$a_{t}=b_{t}+c_{t}a_{t-1}$。
對於$t\geq 1$，就很容易得出：

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

基於下列公式替換$a_t$、$b_t$和$c_t$：

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

公式 :eqref:`eq_bptt_partial_ht_wh_recur`中的梯度計算
滿足$a_{t}=b_{t}+c_{t}a_{t-1}$。
因此，對於每個 :eqref:`eq_bptt_at`，
我們可以使用下面的公式移除 :eqref:`eq_bptt_partial_ht_wh_recur`中的迴圈計算

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

雖然我們可以使用鏈式法則遞迴地計算$\partial h_t/\partial w_h$，
但當$t$很大時這個鏈就會變得很長。
我們需要想想辦法來處理這一問題.

### 完全計算 ###

顯然，我們可以僅僅計算 :eqref:`eq_bptt_partial_ht_wh_gen`中的全部總和，
然而，這樣的計算非常緩慢，並且可能會發生梯度爆炸，
因為初始條件的微小變化就可能會對結果產生巨大的影響。
也就是說，我們可以觀察到類似於蝴蝶效應的現象，
即初始條件的很小變化就會導致結果發生不成比例的變化。
這對於我們想要估計的模型而言是非常不可取的。
畢竟，我們正在尋找的是能夠很好地泛化高穩定性模型的估計器。
因此，在實踐中，這種方法幾乎從未使用過。

### 截斷時間步 ###

或者，我們可以在$\tau$步後截斷
 :eqref:`eq_bptt_partial_ht_wh_gen`中的求和計算。
這是我們到目前為止一直在討論的內容，
例如在 :numref:`sec_rnn_scratch`中分離梯度時。
這會帶來真實梯度的*近似*，
只需將求和終止為$\partial h_{t-\tau}/\partial w_h$。
在實踐中，這種方式工作得很好。
它通常被稱為截斷的透過時間反向傳播 :cite:`Jaeger.2002`。
這樣做導致該模型主要側重於短期影響，而不是長期影響。
這在現實中是可取的，因為它會將估計值偏向更簡單和更穩定的模型。

### 隨機截斷 ###

最後，我們可以用一個隨機變數替換$\partial h_t/\partial w_h$，
該隨機變數在預期中是正確的，但是會截斷序列。
這個隨機變數是透過使用序列$\xi_t$來實現的，
序列預定義了$0 \leq \pi_t \leq 1$，
其中$P(\xi_t = 0) = 1-\pi_t$且$P(\xi_t = \pi_t^{-1}) = \pi_t$，
因此$E[\xi_t] = 1$。
我們使用它來替換 :eqref:`eq_bptt_partial_ht_wh_recur`中的
梯度$\partial h_t/\partial w_h$得到：

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

從$\xi_t$的定義中推導出來$E[z_t] = \partial h_t/\partial w_h$。
每當$\xi_t = 0$時，遞迴計算終止在這個$t$時間步。
這導致了不同長度序列的加權和，其中長序列出現的很少，
所以將適當地加大權重。
這個想法是由塔萊克和奧利維爾 :cite:`Tallec.Ollivier.2017`提出的。

### 比較策略

![比較RNN中計算梯度的策略，3行自上而下分別為：隨機截斷、常規截斷、完整計算](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

 :numref:`fig_truncated_bptt`說明了
當基於迴圈神經網路使用透過時間反向傳播
分析《時間機器》書中前幾個字元的三種策略：

* 第一行採用隨機截斷，方法是將文字劃分為不同長度的片斷；
* 第二行採用常規截斷，方法是將文字分解為相同長度的子序列。
  這也是我們在迴圈神經網路實驗中一直在做的；
* 第三行採用透過時間的完全反向傳播，結果是產生了在計算上不可行的表示式。

遺憾的是，雖然隨機截斷在理論上具有吸引力，
但很可能是由於多種因素在實踐中並不比常規截斷更好。
首先，在對過去若干個時間步經過反向傳播後，
觀測結果足以捕獲實際的依賴關係。
其次，增加的方差抵消了時間步數越多梯度越精確的事實。
第三，我們真正想要的是只有短範圍互動的模型。
因此，模型需要的正是截斷的透過時間反向傳播方法所具備的輕度正則化效果。

## 透過時間反向傳播的細節

在討論一般性原則之後，我們看一下透過時間反向傳播問題的細節。
與 :numref:`subsec_bptt_analysis`中的分析不同，
下面我們將展示如何計算目標函式相對於所有分解模型引數的梯度。
為了保持簡單，我們考慮一個沒有偏置引數的迴圈神經網路，
其在隱藏層中的啟用函式使用恆等對映（$\phi(x)=x$）。
對於時間步$t$，設單個樣本的輸入及其對應的標籤分別為
$\mathbf{x}_t \in \mathbb{R}^d$和$y_t$。
計算隱狀態$\mathbf{h}_t \in \mathbb{R}^h$和
輸出$\mathbf{o}_t \in \mathbb{R}^q$的方式為：

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

其中權重引數為$\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$、
$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和
$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。
用$l(\mathbf{o}_t, y_t)$表示時間步$t$處
（即從序列開始起的超過$T$個時間步）的損失函式，
則我們的目標函式的總體損失是：

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

為了在迴圈神經網路的計算過程中視覺化模型變數和引數之間的依賴關係，
我們可以為模型繪製一個計算圖，
如 :numref:`fig_rnn_bptt`所示。
例如，時間步3的隱狀態$\mathbf{h}_3$的計算
取決於模型引數$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$，
以及最終時間步的隱狀態$\mathbf{h}_2$
以及當前時間步的輸入$\mathbf{x}_3$。

![上圖表示具有三個時間步的迴圈神經網路模型依賴關係的計算圖。未著色的方框表示變數，著色的方框表示引數，圓表示運算符](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

正如剛才所說， :numref:`fig_rnn_bptt`中的模型引數是
$\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$和$\mathbf{W}_{qh}$。
通常，訓練該模型需要對這些引數進行梯度計算：
$\partial L/\partial \mathbf{W}_{hx}$、
$\partial L/\partial \mathbf{W}_{hh}$和
$\partial L/\partial \mathbf{W}_{qh}$。
根據 :numref:`fig_rnn_bptt`中的依賴關係，
我們可以沿箭頭的相反方向遍歷計算圖，依次計算和儲存梯度。
為了靈活地表示鏈式法則中不同形狀的矩陣、向量和標量的乘法，
我們繼續使用如 :numref:`sec_backprop`中
所述的$\text{prod}$運算符。

首先，在任意時間步$t$，
目標函式關於模型輸出的微分計算是相當簡單的：

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

現在，我們可以計算目標函式關於輸出層中引數$\mathbf{W}_{qh}$的梯度：
$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。
基於 :numref:`fig_rnn_bptt`，
目標函式$L$透過$\mathbf{o}_1, \ldots, \mathbf{o}_T$
依賴於$\mathbf{W}_{qh}$。
依據鏈式法則，得到

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

其中$\partial L/\partial \mathbf{o}_t$是
由 :eqref:`eq_bptt_partial_L_ot`給出的。

接下來，如 :numref:`fig_rnn_bptt`所示，
在最後的時間步$T$，目標函式$L$僅透過$\mathbf{o}_T$
依賴於隱狀態$\mathbf{h}_T$。
因此，我們透過使用鏈式法可以很容易地得到梯度
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$：

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

當目標函式$L$透過$\mathbf{h}_{t+1}$和$\mathbf{o}_t$
依賴$\mathbf{h}_t$時，
對任意時間步$t < T$來說都變得更加棘手。
根據鏈式法則，隱狀態的梯度
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$
在任何時間步驟$t < T$時都可以遞迴地計算為：

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

為了進行分析，對於任何時間步$1 \leq t \leq T$展開遞迴計算得

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

我們可以從 :eqref:`eq_bptt_partial_L_ht`中看到，
這個簡單的線性例子已經展現了長序列模型的一些關鍵問題：
它陷入到$\mathbf{W}_{hh}^\top$的潛在的非常大的冪。
在這個冪中，小於1的特徵值將會消失，大於1的特徵值將會發散。
這在數值上是不穩定的，表現形式為梯度消失或梯度爆炸。
解決此問題的一種方法是按照計算方便的需要截斷時間步長的尺寸
如 :numref:`subsec_bptt_analysis`中所述。
實際上，這種截斷是透過在給定數量的時間步之後分離梯度來實現的。
稍後，我們將學習更復雜的序列模型（如長短期記憶模型）
是如何進一步緩解這一問題的。

最後， :numref:`fig_rnn_bptt`表明：
目標函式$L$透過隱狀態$\mathbf{h}_1, \ldots, \mathbf{h}_T$
依賴於隱藏層中的模型引數$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$。
為了計算有關這些引數的梯度
$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$和$\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$，
我們應用鏈式規則得：

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

其中$\partial L/\partial \mathbf{h}_t$
是由 :eqref:`eq_bptt_partial_L_hT_final_step`和
 :eqref:`eq_bptt_partial_L_ht_recur`遞迴計算得到的，
是影響數值穩定性的關鍵量。

正如我們在 :numref:`sec_backprop`中所解釋的那樣，
由於透過時間反向傳播是反向傳播在迴圈神經網路中的應用方式，
所以訓練迴圈神經網路交替使用前向傳播和透過時間反向傳播。
透過時間反向傳播依次計算並存儲上述梯度。
具體而言，儲存的中間值會被重複使用，以避免重複計算，
例如儲存$\partial L/\partial \mathbf{h}_t$，
以便在計算$\partial L / \partial \mathbf{W}_{hx}$和
$\partial L / \partial \mathbf{W}_{hh}$時使用。

## 小結

* 『透過時間反向傳播』僅僅適用於反向傳播在具有隱狀態的序列模型。
* 截斷是計算方便性和數值穩定性的需要。截斷包括：規則截斷和隨機截斷。
* 矩陣的高次冪可能導致神經網路特徵值的發散或消失，將以梯度爆炸或梯度消失的形式表現。
* 為了計算的效率，『透過時間反向傳播』在計算期間會快取中間值。

## 練習

1. 假設我們擁有一個對稱矩陣$\mathbf{M} \in \mathbb{R}^{n \times n}$，其特徵值為$\lambda_i$，對應的特徵向量是$\mathbf{v}_i$（$i = 1, \ldots, n$）。通常情況下，假設特徵值的序列順序為$|\lambda_i| \geq |\lambda_{i+1}|$。
   1. 證明$\mathbf{M}^k$擁有特徵值$\lambda_i^k$。
   1. 證明對於一個隨機向量$\mathbf{x} \in \mathbb{R}^n$，$\mathbf{M}^k \mathbf{x}$將有較高機率與$\mathbf{M}$的特徵向量$\mathbf{v}_1$在一條直線上。形式化這個證明過程。
   1. 上述結果對於迴圈神經網路中的梯度意味著什麼？
1. 除了梯度截斷，還有其他方法來應對迴圈神經網路中的梯度爆炸嗎？

[Discussions](https://discuss.d2l.ai/t/2107)
