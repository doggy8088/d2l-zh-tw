# 多層感知機
:label:`chap_perceptrons`

在本章中，我們將第一次介紹真正的*深度*網路。
最簡單的深度網路稱為*多層感知機*。多層感知機由多層神經元組成，
每一層與它的上一層相連，從中接收輸入；
同時每一層也與它的下一層相連，影響當前層的神經元。
當我們訓練容量較大的模型時，我們面臨著*過擬合*的風險。
因此，本章將從基本的概念介紹開始講起，包括*過擬合*、*欠擬合*和模型選擇。
為了解決這些問題，本章將介紹*權重衰減*和*暫退法*等正則化技術。
我們還將討論數值穩定性和引數初始化相關的問題，
這些問題是成功訓練深度網路的關鍵。
在本章的最後，我們將把所介紹的內容應用到一個真實的案例：房價預測。
關於模型計算效能、可伸縮性和效率相關的問題，我們將放在後面的章節中討論。

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```
