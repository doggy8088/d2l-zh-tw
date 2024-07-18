# 前向傳播、反向傳播和計算圖
:label:`sec_backprop`

我們已經學習瞭如何用小批次隨機梯度下降訓練模型。
然而當實現該演算法時，我們只考慮了透過*前向傳播*（forward propagation）所涉及的計算。
在計算梯度時，我們只調用了深度學習框架提供的反向傳播函式，而不知其所以然。

梯度的自動計算（自動微分）大大簡化了深度學習演算法的實現。
在自動微分之前，即使是對複雜模型的微小調整也需要手工重新計算複雜的導數，
學術論文也不得不分配大量頁面來推導更新規則。
本節將透過一些基本的數學和計算圖，
深入探討*反向傳播*的細節。
首先，我們將重點放在帶權重衰減（$L_2$正則化）的單隱藏層多層感知機上。

## 前向傳播

*前向傳播*（forward propagation或forward pass）
指的是：按順序（從輸入層到輸出層）計算和儲存神經網路中每層的結果。

我們將一步步研究單隱藏層神經網路的機制，
為了簡單起見，我們假設輸入樣本是 $\mathbf{x}\in \mathbb{R}^d$，
並且我們的隱藏層不包括偏置項。
這裡的中間變數是：

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

其中$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
是隱藏層的權重引數。
將中間變數$\mathbf{z}\in \mathbb{R}^h$透過啟用函式$\phi$後，
我們得到長度為$h$的隱藏啟用向量：

$$\mathbf{h}= \phi (\mathbf{z}).$$

隱藏變數$\mathbf{h}$也是一個中間變數。
假設輸出層的引數只有權重$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$，
我們可以得到輸出層變數，它是一個長度為$q$的向量：

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

假設損失函式為$l$，樣本標籤為$y$，我們可以計算單個資料樣本的損失項，

$$L = l(\mathbf{o}, y).$$

根據$L_2$正則化的定義，給定超引數$\lambda$，正則化項為

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

其中矩陣的Frobenius範數是將矩陣展平為向量後應用的$L_2$範數。
最後，模型在給定資料樣本上的正則化損失為：

$$J = L + s.$$

在下面的討論中，我們將$J$稱為*目標函式*（objective function）。

## 前向傳播計算圖

繪製*計算圖*有助於我們視覺化計算中運算子和變數的依賴關係。
 :numref:`fig_forward` 是與上述簡單網路相對應的計算圖，
 其中正方形表示變數，圓圈表示運算子。
 左下角表示輸入，右上角表示輸出。
 注意顯示資料流的箭頭方向主要是向右和向上的。

![前向傳播的計算圖](../img/forward.svg)
:label:`fig_forward`

## 反向傳播

*反向傳播*（backward propagation或backpropagation）指的是計算神經網路引數梯度的方法。
簡言之，該方法根據微積分中的*鏈式規則*，按相反的順序從輸出層到輸入層遍歷網路。
該演算法儲存了計算某些引數梯度時所需的任何中間變數（偏導數）。
假設我們有函式$\mathsf{Y}=f(\mathsf{X})$和$\mathsf{Z}=g(\mathsf{Y})$，
其中輸入和輸出$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$是任意形狀的張量。
利用鏈式法則，我們可以計算$\mathsf{Z}$關於$\mathsf{X}$的導數

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

在這裡，我們使用$\text{prod}$運算符在執行必要的操作（如換位和交換輸入位置）後將其引數相乘。
對於向量，這很簡單，它只是矩陣-矩陣乘法。
對於高維張量，我們使用適當的對應項。
運算符$\text{prod}$指代了所有的這些符號。

回想一下，在計算圖 :numref:`fig_forward`中的單隱藏層簡單網路的引數是
$\mathbf{W}^{(1)}$和$\mathbf{W}^{(2)}$。
反向傳播的目的是計算梯度$\partial J/\partial \mathbf{W}^{(1)}$和
$\partial J/\partial \mathbf{W}^{(2)}$。
為此，我們應用鏈式法則，依次計算每個中間變數和引數的梯度。
計算的順序與前向傳播中執行的順序相反，因為我們需要從計算圖的結果開始，並朝著引數的方向努力。第一步是計算目標函式$J=L+s$相對於損失項$L$和正則項$s$的梯度。

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

接下來，我們根據鏈式法則計算目標函式關於輸出層變數$\mathbf{o}$的梯度：

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

接下來，我們計算正則化項相對於兩個引數的梯度：

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

現在我們可以計算最接近輸出層的模型引數的梯度
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$。
使用鏈式法則得出：

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

為了獲得關於$\mathbf{W}^{(1)}$的梯度，我們需要繼續沿著輸出層到隱藏層反向傳播。
關於隱藏層輸出的梯度$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$由下式給出：

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

由於啟用函式$\phi$是按元素計算的，
計算中間變數$\mathbf{z}$的梯度$\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
需要使用按元素乘法運算符，我們用$\odot$表示：

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

最後，我們可以得到最接近輸入層的模型引數的梯度
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$。
根據鏈式法則，我們得到：

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## 訓練神經網路

在訓練神經網路時，前向傳播和反向傳播相互依賴。
對於前向傳播，我們沿著依賴的方向遍歷計算圖並計算其路徑上的所有變數。
然後將這些用於反向傳播，其中計算順序與計算圖的相反。

以上述簡單網路為例：一方面，在前向傳播期間計算正則項
 :eqref:`eq_forward-s`取決於模型引數$\mathbf{W}^{(1)}$和
$\mathbf{W}^{(2)}$的當前值。
它們是由最佳化演算法根據最近迭代的反向傳播給出的。
另一方面，反向傳播期間引數 :eqref:`eq_backprop-J-h`的梯度計算，
取決於由前向傳播給出的隱藏變數$\mathbf{h}$的當前值。

因此，在訓練神經網路時，在初始化模型引數後，
我們交替使用前向傳播和反向傳播，利用反向傳播給出的梯度來更新模型引數。
注意，反向傳播重複利用前向傳播中儲存的中間值，以避免重複計算。
帶來的影響之一是我們需要保留中間值，直到反向傳播完成。
這也是訓練比單純的預測需要更多的記憶體（顯存）的原因之一。
此外，這些中間值的大小與網路層的數量和批次的大小大致成正比。
因此，使用更大的批次來訓練更深層次的網路更容易導致*記憶體不足*（out of memory）錯誤。

## 小結

* 前向傳播在神經網路定義的計算圖中按順序計算和儲存中間變數，它的順序是從輸入層到輸出層。
* 反向傳播按相反的順序（從輸出層到輸入層）計算和儲存神經網路的中間變數和引數的梯度。
* 在訓練深度學習模型時，前向傳播和反向傳播是相互依賴的。
* 訓練比預測需要更多的記憶體。

## 練習

1. 假設一些標量函式$\mathbf{X}$的輸入$\mathbf{X}$是$n \times m$矩陣。$f$相對於$\mathbf{X}$的梯度維數是多少？
1. 向本節中描述的模型的隱藏層新增偏置項（不需要在正則化項中包含偏置項）。
    1. 畫出相應的計算圖。
    1. 推導正向和反向傳播方程。
1. 計算本節所描述的模型，用於訓練和預測的記憶體佔用。
1. 假設想計算二階導數。計算圖發生了什麼？預計計算需要多長時間？
1. 假設計算圖對當前擁有的GPU來說太大了。
    1. 請試著把它劃分到多個GPU上。
    1. 與小批次訓練相比，有哪些優點和缺點？

[Discussions](https://discuss.d2l.ai/t/5769)
