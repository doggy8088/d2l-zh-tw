# 前言

幾年前，在大公司和初創公司中，並沒有大量的深度學習科學家開發智慧產品和服務。我們中年輕人（作者）進入這個領域時，機器學習並沒有在報紙上獲得頭條新聞。我們的父母根本不知道什麼是機器學習，更不用說為什麼我們可能更喜歡機器學習，而不是從事醫學或法律職業。機器學習是一門具有前瞻性的學科，在現實世界的應用範圍很窄。而那些應用，例如語音識別和電腦視覺，需要大量的領域知識，以至於它們通常被認為是完全獨立的領域，而機器學習對這些領域來說只是一個小組件。因此，神經網路——我們在本書中關注的深度學習模型的前身，被認為是過時的工具。

就在過去的五年裡，深度學習給世界帶來了驚喜，推動了電腦視覺、自然語言處理、自動語音識別、強化學習和統計建模等領域的快速發展。有了這些進步，我們現在可以製造比以往任何時候都更自主的汽車（不過可能沒有一些公司試圖讓大家相信的那麼自主），可以自動起草普通郵件的智慧回覆系統，幫助人們從令人壓抑的大收件箱中解放出來。在圍棋等棋類遊戲中，軟體超越了世界上最優秀的人，這曾被認為是幾十年後的事。這些工具已經對工業和社會產生了越來越廣泛的影響，改變了電影的製作方式、疾病的診斷方式，並在基礎科學中扮演著越來越重要的角色——從天體物理學到生物學。

## 關於本書

這本書代表了我們的嘗試——讓深度學習可平易近人，教會人們*概念*、*背景*和*程式碼*。

### 一種結合了程式碼、數學和HTML的媒介

任何一種計算技術要想發揮其全部影響力，都必須得到充分的理解、充分的文件記錄，並得到成熟的、維護良好的工具的支援。關鍵思想應該被清楚地提煉出來，儘可能減少需要讓新的從業者跟上時代的入門時間。成熟的函式庫應該自動化常見的任務，範例程式碼應該使從業者可以輕鬆地修改、應用和擴充套件常見的應用程式，以滿足他們的需求。以動態網頁應用為例。儘管許多公司，如亞馬遜，在20世紀90年代開發了成功的資料庫驅動網頁應用程式。但在過去的10年裡，這項技術在幫助創造性企業家方面的潛力已經得到了更大程度的發揮，部分原因是開發了功能強大、文件完整的框架。

測試深度學習的潛力帶來了獨特的挑戰，因為任何一個應用都會將不同的學科結合在一起。應用深度學習需要同時瞭解（1）以特定方式提出問題的動機；（2）給定建模方法的數學；（3）將模型擬合數據的最佳化演算法；（4）能夠有效訓練模型、克服數值計算缺陷並最大限度地利用現有硬體的工程方法。同時教授表述問題所需的批判性思維技能、解決問題所需的數學知識，以及實現這些解決方案所需的軟體工具，這是一個巨大的挑戰。

在我們開始寫這本書的時候，沒有資源能夠同時滿足一些條件：（1）是最新的；（2）涵蓋了現代機器學習的所有領域，技術深度豐富；（3）在一本引人入勝的教科書中，人們可以在實踐教程中找到乾淨的可執行程式碼，並從中穿插高質量的闡述。我們發現了大量關於如何使用給定的深度學習框架（例如，如何對TensorFlow中的矩陣進行基本的數值計算)或實現特定技術的程式碼範例（例如，LeNet、AlexNet、ResNet的程式碼片段），這些程式碼範例分散在各種部落格文章和GitHub庫中。但是，這些範例通常關注如何實現給定的方法，但忽略了為什麼做出某些演算法決策的討論。雖然一些互動資源已經零星地出現以解決特定主題。例如，在網站[Distill](http://distill.pub)上發布的引人入勝的部落格文章或個人部落格，但它們僅覆蓋深度學習中的選定主題，並且通常缺乏相關程式碼。另一方面，雖然已經出現了幾本教科書，其中最著名的是 :cite:`Goodfellow.Bengio.Courville.2016`（中文名《深度學習》），它對深度學習背後的概念進行了全面的調查，但這些資源並沒有將這些概念的描述與這些概念的程式碼實現結合起來。有時會讓讀者對如何實現它們一無所知。此外，太多的資源隱藏在商業課程提供商的付費壁壘後面。

我們著手建立的資源可以：（1）每個人都可以免費獲得；（2）提供足夠的技術深度，為真正成為一名應用機器學習科學家提供起步；（3）包括可執行的程式碼，向讀者展示如何解決實踐中的問題；（4）允許我們和社群的快速更新;（5）由一個[論壇](http://discuss.d2l.ai)作為補充，用於技術細節的互動討論和回答問題。

這些目標經常是相互衝突的。公式、定理和引用最好用LaTeX來管理和佈局。程式碼最好用Python描述。網頁原生是HTML和JavaScript的。此外，我們希望內容既可以作為可執行程式碼訪問、作為紙質書訪問，作為可下載的PDF訪問，也可以作為網站在網際網路上訪問。目前還沒有完全適合這些需求的工具和工作流程，所以我們不得不自行組裝。我們在 :numref:`sec_how_to_contribute` 中詳細描述了我們的方法。我們選擇GitHub來共享原始碼並允許編輯，選擇Jupyter記事本來混合程式碼、公式和文字，選擇Sphinx作為渲染引擎來生成多個輸出，併為論壇提供討論。雖然我們的體系尚不完善，但這些選擇在相互衝突的問題之間提供了一個很好的妥協。我們相信，這可能是第一本使用這種整合工作流程出版的書。

### 在實踐中學習

許多教科書教授一系列的主題，每一個都非常詳細。例如，Chris Bishop的優秀教科書 :cite:`Bishop.2006` ，對每個主題都教得很透徹，以至於要讀到線性迴歸這一章需要大量的工作。雖然專家們喜歡這本書正是因為它的透徹性，但對初學者來說，這一屬性限制了它作為介紹性文字的實用性。

在這本書中，我們將適時教授大部分概念。換句話說，你將在實現某些實際目的所需的非常時刻學習概念。雖然我們在開始時花了一些時間來教授基礎的背景知識，如線性代數和機率，但我們希望你在思考更深奧的機率分佈之前，先體會一下訓練模型的滿足感。

除了提供基本數學背景速成課程的幾節初步課程外，後續的每一章都介紹了適量的新概念，並提供可獨立工作的例子——使用真實的資料集。這帶來了組織上的挑戰。某些模型可能在邏輯上組合在單節中。而一些想法可能最好是透過連續允許幾個模型來傳授。另一方面，堅持『一個工作例子一節』的策略有一個很大的好處：這使你可以透過利用我們的程式碼儘可能輕鬆地啟動你自己的研究專案。只需複製這一節的內容並開始修改即可。

我們將根據需要將可執行程式碼與背景材料交錯。通常，在充分解釋工具之前，我們常常會在提供工具這一方面犯錯誤（我們將在稍後解釋背景）。例如，在充分解釋*隨機梯度下降*為什麼有用或為什麼有效之前，我們可以使用它。這有助於給從業者提供快速解決問題所需的彈藥，同時需要讀者相信我們的一些決定。

這本書將從頭開始教授深度學習的概念。有時，我們想深入研究模型的細節，這些的細節通常會被深度學習框架的高階抽象隱藏起來。特別是在基礎教程中，我們希望讀者瞭解在給定層或最佳化器中發生的一切。在這些情況下，我們通常會提供兩個版本的範例：一個是我們從零開始實現一切，僅依賴張量操作和自動微分；另一個是更實際的範例，我們使用深度學習框架的高階API編寫簡潔的程式碼。一旦我們教了您一些元件是如何工作的，我們就可以在隨後的教程中使用高階API了。

### 內容和結構

全書大致可分為三個部分，在 :numref:`fig_book_org` 中用不同的顏色呈現：

![全書結構](../img/book-org.svg)
:label:`fig_book_org`

* 第一部分包括基礎知識和預備知識。
:numref:`chap_introduction` 提供深度學習的入門課程。然後在 :numref:`chap_preliminaries` 中，我們將快速介紹實踐深度學習所需的前提條件，例如如何儲存和處理資料，以及如何應用基於線性代數、微積分和機率基本概念的各種數值運算。 :numref:`chap_linear` 和 :numref:`chap_perceptrons` 涵蓋了深度學習的最基本概念和技術，例如線性迴歸、多層感知機和正則化。

* 接下來的五章集中討論現代深度學習技術。
:numref:`chap_computation` 描述了深度學習計算的各種關鍵元件，併為我們隨後實現更復雜的模型奠定了基礎。接下來，在 :numref:`chap_cnn` 和 :numref:`chap_modern_cnn` 中，我們介紹了卷積神經網路（convolutional neural network，CNN），這是構成大多數現代電腦視覺系統骨幹的強大工具。隨後，在 :numref:`chap_rnn` 和 :numref:`chap_modern_rnn` 中，我們引入了迴圈神經網路(recurrent neural network，RNN)，這是一種利用資料中的時間或序列結構的模型，通常用於自然語言處理和時間序列預測。在 :numref:`chap_attention` 中，我們介紹了一類新的模型，它採用了一種稱為注意力機制的技術，最近它們已經開始在自然語言處理中取代迴圈神經網路。這一部分將幫助讀者快速瞭解大多數現代深度學習應用背後的基本工具。

* 第三部分討論可伸縮性、效率和應用程式。
首先，在 :numref:`chap_optimization` 中，我們討論了用於訓練深度學習模型的幾種常用最佳化演算法。下一章 :numref:`chap_performance` 將探討影響深度學習程式碼計算效能的幾個關鍵因素。在 :numref:`chap_cv` 中，我們展示了深度學習在電腦視覺中的主要應用。在 :numref:`chap_nlp_pretrain` 和 :numref:`chap_nlp_app` 中，我們展示了如何預訓練語言表示模型並將其應用於自然語言處理任務。

### 程式碼
:label:`sec_code`

本書的大部分章節都以可執行程式碼為特色，因為我們相信互動式學習體驗在深度學習中的重要性。目前，某些直覺只能透過試錯、小幅調整程式碼並觀察結果來發展。理想情況下，一個優雅的數學理論可能會精確地告訴我們如何調整程式碼以達到期望的結果。不幸的是，這種優雅的理論目前還沒有出現。儘管我們盡了最大努力，但仍然缺乏對各種技術的正式解釋，這既是因為描述這些模型的數學可能非常困難，也是因為對這些主題的認真研究最近才進入高潮。我們希望隨著深度學習理論的發展，這本書的未來版本將能夠在當前版本無法提供的地方提供見解。

有時，為了避免不必要的重複，我們將本書中經常匯入和引用的函式、類等封裝在`d2l`套件中。對於要儲存到套件中的任何程式碼區塊，比如一個函式、一個類或者多個匯入，我們都會標記為`#@save`。我們在 :numref:`sec_d2l` 中提供了這些函式和類別的詳細描述。`d2l`軟體套件是輕量級的，僅需要以下軟體套件和模組作為依賴項：

```{.python .input}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
本書中的大部分程式碼都是基於Apache MXNet的。MXNet是深度學習的開源框架，是亞馬遜以及許多大學和公司的首選。本書中的所有程式碼都透過了最新MXNet版本的測試。但是，由於深度學習的快速發展，一些在印刷版中程式碼
可能在MXNet的未來版本無法正常工作。
但是，我們計劃使線上版本保持最新。如果讀者遇到任何此類問題，請檢視 :ref:`chap_installation` 以更新程式碼和執行時環境。

下面是我們如何從MXNet匯入模組。
:end_tab:

:begin_tab:`pytorch`
本書中的大部分程式碼都是基於PyTorch的。PyTorch是一個開源的深度學習框架，在研究界非常受歡迎。本書中的所有程式碼都在最新版本的PyTorch下透過了測試。但是，由於深度學習的快速發展，一些在印刷版中程式碼可能在PyTorch的未來版本無法正常工作。
但是，我們計劃使線上版本保持最新。如果讀者遇到任何此類問題，請檢視 :ref:`chap_installation` 以更新程式碼和執行時環境。

下面是我們如何從PyTorch匯入模組。
:end_tab:

:begin_tab:`tensorflow`
本書中的大部分程式碼都是基於TensorFlow的。TensorFlow是一個開源的深度學習框架，在研究界和產業界都非常受歡迎。本書中的所有程式碼都在最新版本的TensorFlow下透過了測試。但是，由於深度學習的快速發展，一些在印刷版中程式碼可能在TensorFlow的未來版本無法正常工作。
但是，我們計劃使線上版本保持最新。如果讀者遇到任何此類問題，請檢視 :ref:`chap_installation` 以更新程式碼和執行時環境。

下面是我們如何從TensorFlow匯入模組。
:end_tab:

```{.python .input}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab paddle
#@save
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.vision import transforms
import paddle.vision as paddlevision
from PIL import Image
paddle.disable_signal_handler()
```

### 目標受眾

本書面向學生（本科生或研究生）、工程師和研究人員，他們希望紮實掌握深度學習的實用技術。因為我們從頭開始解釋每個概念，所以不需要過往的深度學習或機器學習背景。全面解釋深度學習的方法需要一些數學和程式設計，但我們只假設讀者瞭解一些基礎知識，包括線性代數、微積分、機率和非常基礎的Python程式設計。此外，在附錄中，我們提供了本書所涵蓋的大多數數學知識的複習。大多數時候，我們會優先考慮直覺和想法，而不是數學的嚴謹性。有許多很棒的書可以引導感興趣的讀者走得更遠。Bela Bollobas的《線性分析》 :cite:`Bollobas.1999` 對線性代數和函式分析進行了深入的研究。 :cite:`Wasserman.2013` 是一本很好的統計學指南。如果讀者以前沒有使用過Python語言，那麼可以仔細閱讀這個[Python教程](http://learnpython.org/)。

### 論壇

與本書相關，我們已經啟動了一個論壇，在[discuss.d2l.ai](https://discuss.d2l.ai/)。當對本書的任何一節有疑問時，請在每一節的末尾找到相關的討論頁連結。

## 致謝

感謝中英文草稿的數百位撰稿人。他們幫助改進了內容並提供了寶貴的反饋。
感謝Anirudh Dagar和唐源將部分較早版本的MXNet實現分別改編為PyTorch和TensorFlow實現。
感謝百度團隊將較新的PyTorch實現改編為PaddlePaddle實現。
感謝張帥將更新的LaTeX樣式整合進PDF檔案的編譯。

特別地，我們要感謝這份中文稿的每一位撰稿人，是他們的無私奉獻讓這本書變得更好。他們的GitHub ID或姓名是(沒有特定順序)：alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, 315930399, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Zhu Yuanxiang,
thebesttv, Quanshangze Du, Yanbo Chen。

我們感謝Amazon Web Services，特別是Swami Sivasubramanian、Peter DeSantis、Adam Selipsky和Andrew Jassy對撰寫本書的慷慨支援。如果沒有可用的時間、資源、與同事的討論和不斷的鼓勵，這本書就不會出版。

## 小結

* 深度學習已經徹底改變了模式識別，引入了一系列技術，包括電腦視覺、自然語言處理、自動語音識別。
* 要成功地應用深度學習，必須知道如何丟擲一個問題、建模的數學方法、將模型與資料擬合的演算法，以及實現所有這些的工程技術。
* 這本書提供了一個全面的資源，包括文字、圖表、數學和程式碼，都集中在一個地方。
* 要回答與本書相關的問題，請訪問我們的論壇[discuss.d2l.ai](https://discuss.d2l.ai/).
* 所有Jupyter記事本都可以在GitHub上下載。

## 練習

1. 在本書[discuss.d2l.ai](https://discuss.d2l.ai/)的論壇上註冊帳戶。
1. 在電腦上安裝Python。
1. 沿著本節底部的連結進入論壇，在那裡可以尋求幫助、討論這本書，並透過與作者和社群接觸來找到問題的答案。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2085)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2086)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2087)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11678)
:end_tab:
