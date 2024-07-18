# 特徵豐富的推薦系統

互動資料是表明使用者偏好和興趣的最基本指標。它在之前介紹的模型中起到了關鍵性作用。但是，互動資料通常比較稀疏，有時也會包含一些噪音。為了解決這一問題，我們可以將物品特徵、使用者資料和互動發生時的上下文等輔助訊息整合到推薦模型中。利用這些特徵可以幫助做出更好的推薦，這是因為，當互動資料較為匱乏時，這些特徵能夠很好地預測使用者的興趣。因此，推薦模型需要具備處理這類特徵的能力，並且能夠捕捉到內容和上下文中的訊息。為了示範這一模型的原理，我們將介紹一個線上廣告點選率（click-through rate，CTR）預測任務:cite:`McMahan.Holt.Sculley.ea.2013`，並給出一個匿名的廣告資料集。定向廣告服務已在業界引起了廣泛關注，它通常被設計成推薦引擎的形式。對於提高點選率來說，推薦對應使用者品味和興趣的廣告非常重要。

數字營銷人員利用線上廣告向客戶展示廣告訊息。點選率是一種測量指標，它用於衡量客戶的廣告被點選的比例大小。點選率由以下公式計算得到：

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

點選率是預示演算法有效性的重要指標，而點選率預測則是計算網站上的某些內容的點選機率的任務。點選率預測模型不僅能夠用在定向廣告系統中，它也能用在常規物品（電影、新聞和產品等）推薦系統、電子郵件廣告系統和搜尋引擎中。它還和使用者滿意度以及轉化率有著緊密的關係。在設定營銷目標時它也能有所幫助，因為它可以讓廣告商的預期切合實際。

```python
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## 線上廣告資料集

由於網際網路和移動計算技術的巨大進步，線上廣告已經成為網際網路行業的一項重要收入來源，併為其帶來了絕大部分營收。所以，展示相關廣告或者激發使用者興趣的廣告，進而將普通使用者轉化為付費使用者就變得非常重要。接下來我們將介紹一個線上廣告資料集。它由34個欄位組成，其中第一清單示廣告是否被點選（1表示點選，0表示未點選）的目標變數。其他列都是標籤化特徵。這些列可能表示廣告ID、站點ID、應用ID、裝置ID、時間戳和使用者訊息等等。出於匿名和隱私保護的目的，這些列的真實語義並未公佈。

下面的程式碼將從我們的伺服器上下載該資料集，然後儲存到本地資料夾中。

```python
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

訓練集和測試集中分別包含了15000條和3000條資料。

## 資料集套件裝器

為了方便地從csv檔案中載入廣告資料，我們實現了一個名為`CTRDataset`的類，它可以被`DataLoader`呼叫。

```python
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1 # feature_cnts[feature_dim]->{value1:cnts1,...}
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()} # feat_mapper[feature_dim]-> set(v1,...)
            self.feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} # feat_map[feature_dim]-> {v1:idx1,...}
                                for i, cnt in feat_mapper.items()}
            self.defaults = {i: len(cnt) for i, cnt in feat_mapper.items()} # default index for feature[dim][value]
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy() # offset for feature in dim with value X
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

下面的例子將會載入訓練資料，然後輸出第一條記錄。

```python
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

如你所見，這34個欄位都是分類特徵。每個數值都代表了對應條目的獨熱索引，標籤$0$表示沒有被點選。這裡的`CTRDataset`也可以用來載入其他的資料集，例如Criteo展示廣告挑戰賽[資料集](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)和Avazu點選率預測[資料集](https://www.kaggle.com/c/avazu-ctr-prediction) 。

## 小結

* 點選率是一項很重要的指標，它能用於評估廣告系統和推薦系統的效能。
* 點選率預測經常被轉化為二分類問題。該問題的目標是，在給定特徵後，預測廣告或物品是否會被點選。

## 練習

* 你能使用`CTRDataset`載入Criteo和Avazu資料集嗎？需要注意的是，Criteo包含了實數值特徵，因此你可能需要稍微修改一下程式碼。

[討論](https://discuss.d2l.ai/t/405)
