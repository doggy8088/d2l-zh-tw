# 自然語言處理：應用
:label:`chap_nlp_app`

前面我們學習瞭如何在文字序列中表示詞元，
並在 :numref:`chap_nlp_pretrain`中訓練了詞元的表示。
這樣的預訓練文字表示可以透過不同模型架構，放入不同的下游自然語言處理任務。

前一章我們提及到一些自然語言處理應用，這些應用沒有預訓練，只是為了解釋深度學習架構。
例如，在 :numref:`chap_rnn`中，
我們依賴迴圈神經網路設計語言模型來生成類似中篇小說的文字。
在 :numref:`chap_modern_rnn`和 :numref:`chap_attention`中，
我們還設計了基於迴圈神經網路和注意力機制的機器翻譯模型。

然而，本書並不打算全面涵蓋所有此類應用。
相反，我們的重點是*如何應用深度語言表徵學習來解決自然語言處理問題*。
在給定預訓練的文字表示的情況下，
本章將探討兩種流行且具有代表性的下游自然語言處理任務：
情感分析和自然語言推斷，它們分別分析單個文字和文字對之間的關係。

![預訓練文字表示可以透過不同模型架構，放入不同的下游自然語言處理應用（本章重點介紹如何為不同的下游應用設計模型）](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

如 :numref:`fig_nlp-map-app`所述，
本章將重點描述然後使用不同型別的深度學習架構
（如多層感知機、卷積神經網路、迴圈神經網路和注意力）
設計自然語言處理模型。
儘管在 :numref:`fig_nlp-map-app`中，
可以將任何預訓練的文字表示與任何應用的架構相結合，
但我們選擇了一些具有代表性的組合。
具體來說，我們將探索基於迴圈神經網路和卷積神經網路的流行架構進行情感分析。
對於自然語言推斷，我們選擇注意力和多層感知機來示範如何分析文字對。
最後，我們介紹瞭如何為廣泛的自然語言處理應用，
如在序列級（單文字分類和文字對分類）和詞元級（文字標註和問答）上
對預訓練BERT模型進行微調。
作為一個具體的經驗案例，我們將針對自然語言推斷對BERT進行微調。

正如我們在 :numref:`sec_bert`中介紹的那樣，
對於廣泛的自然語言處理應用，BERT只需要最少的架構更改。
然而，這一好處是以微調下游應用的大量BERT引數為代價的。
當空間或時間有限時，基於多層感知機、卷積神經網路、迴圈神經網路
和注意力的精心建立的模型更具可行性。
下面，我們從情感分析應用開始，分別解讀基於迴圈神經網路和卷積神經網路的模型設計。

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
