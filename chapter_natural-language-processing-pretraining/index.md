# 自然語言處理：預訓練
:label:`chap_nlp_pretrain`

人與人之間需要交流。
出於人類這種基本需要，每天都有大量的書面文字產生。
比如，社交媒體、聊天應用、電子郵件、產品評論、新聞文章、
研究論文和書籍中的豐富文字，
使電腦能夠理解它們以提供幫助或基於人類語言做出決策變得至關重要。

*自然語言處理*是指研究使用自然語言的電腦和人類之間的互動。
在實踐中，使用自然語言處理技術來處理和分析文字資料是非常常見的，
例如 :numref:`sec_language_model`的語言模型
和 :numref:`sec_machine_translation`的機器翻譯模型。

要理解文字，我們可以從學習它的表示開始。
利用來自大型語料庫的現有文字序列，
*自監督學習*（self-supervised learning）
已被廣泛用於預訓練文字表示，
例如透過使用周圍文字的其它部分來預測文字的隱藏部分。
透過這種方式，模型可以透過有監督地從*海量*文字資料中學習，而不需要*昂貴*的標籤標註！

本章我們將看到：當將每個單詞或子詞視為單個詞元時，
可以在大型語料庫上使用word2vec、GloVe或子詞嵌入模型預先訓練每個詞元的詞元。
經過預訓練後，每個詞元的表示可以是一個向量。
但是，無論上下文是什麼，它都保持不變。
例如，『bank』（可以譯作銀行或者河岸）的向量表示在
『go to the bank to deposit some money』（去銀行存點錢）
和『go to the bank to sit down』（去河岸坐下來）中是相同的。
因此，許多較新的預訓練模型使相同詞元的表示適應於不同的上下文，
其中包括基於Transformer編碼器的更深的自監督模型BERT。
在本章中，我們將重點討論如何預訓練文字的這種表示，
如 :numref:`fig_nlp-map-pretrain`中所強調的那樣。

![預訓練好的文字表示可以放入各種深度學習架構，應用於不同自然語言處理任務（本章主要研究上游文字的預訓練）](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`

 :numref:`fig_nlp-map-pretrain`顯示了
預訓練好的文字表示可以放入各種深度學習架構，應用於不同自然語言處理任務。
我們將在 :numref:`chap_nlp_app`中介紹它們。


```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining
```
