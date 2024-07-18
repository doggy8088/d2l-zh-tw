# 計算效能
:label:`chap_performance`

在深度學習中，資料集和模型通常都很大，導致計算量也會很大。
因此，計算的效能非常重要。
本章將集中討論影響計算效能的主要因素：指令式程式設計、符號程式設計、
非同步計算、自動並行和多GPU計算。
透過學習本章，對於前幾章中實現的那些模型，可以進一步提高它們的計算效能。
例如，我們可以在不影響準確性的前提下，大大減少訓練時間。

```toc
:maxdepth: 2

hybridize
async-computation
auto-parallelism
hardware
multiple-gpus
multiple-gpus-concise
parameterserver
```
