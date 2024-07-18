# 注意力機制
:label:`chap_attention`

靈長類動物的視覺系統接受了大量的感官輸入，
這些感官輸入遠遠超過了大腦能夠完全處理的程度。
然而，並非所有刺激的影響都是相等的。
意識的聚集和專注使靈長類動物能夠在複雜的視覺環境中將注意力引向感興趣的物體，例如獵物和天敵。
只關注一小部分訊息的能力對進化更加有意義，使人類得以生存和成功。

自19世紀以來，科學家們一直致力於研究認知神經科學領域的注意力。
本章的很多章節將涉及到一些研究。

首先回顧一個經典注意力框架，解釋如何在視覺場景中展開注意力。
受此框架中的*注意力提示*（attention cues）的啟發，
我們將設計能夠利用這些注意力提示的模型。
1964年的Nadaraya-Waston核迴歸（kernel regression）正是具有
*注意力機制*（attention mechanism）的機器學習的簡單示範。

然後繼續介紹的是注意力函式，它們在深度學習的注意力模型設計中被廣泛使用。
具體來說，我們將展示如何使用這些函式來設計*Bahdanau注意力*。
Bahdanau注意力是深度學習中的具有突破性價值的注意力模型，它雙向對齊並且可以微分。

最後將描述僅僅基於注意力機制的*Transformer*架構，
該架構中使用了*多頭注意力*（multi-head attention）
和*自注意力*（self-attention）。
自2017年橫空出世，Transformer一直都普遍存在於現代的深度學習應用中，
例如語言、視覺、語音和強化學習領域。

```toc
:maxdepth: 2

attention-cues
nadaraya-waston
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
```
