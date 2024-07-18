# MovieLens資料集

用於推薦系統研究的資料集有很多，而其中[MovieLens](https://movielens.org/)資料集可能是最受歡迎的一個。1997年，為了收集評分資料用於研究目的，明尼蘇達大學的GroupLens研究實驗室建立了本資料集。MovieLens資料集在包括個性化推薦和社會心理學在內的多個研究領域起到了關鍵作用。

## 獲取資料

MovieLens資料集託管在了[GroupLens](https://grouplens.org/datasets/movielens/)網站上。它包括多個可用版本。此處我們將使用其中的100K版本:cite:`Herlocker.Konstan.Borchers.ea.1999`。該資料集的10萬條評分（從一星到五星）來自於943名使用者對於1682部電影的評價。該資料集已經經過清洗處理，每個使用者都至少有二十條評分資料。該資料集還提供了簡單的人口統計訊息，例如年齡、性別、風格和物品等。下載壓縮封裝[ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)後解壓得到`u.data`檔案，其中包含了csv格式的10萬條評分。資料夾中還有許多其他的檔案，關於這些檔案的詳細說明可以在資料集的[README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt)中找到。

在開始之前，讓我們先導入執行本節試驗所必須的模組。

```python
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

之後，我們下載MovieLens-100k資料集，以`DataFrame`格式載入互動資料。

```python
#@save
d2l.DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    """讀取MovieLens-100k資料集"""
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## 資料集統計

我們載入一下資料，然後手動檢查一下前五條記錄。如此一來，我們可以有效地瞭解資料結構並確保它們已經正確載入。

```python
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

每行數據由四列組成，其中包括使用者id（1-943）、物品id（1-1682）、評分（1-5）和時間戳。我們可以據此構造一個大小為$n \times m$的矩陣，$n$和$m$分別代表使用者和物品的數量。該資料集僅記錄了已有的評分，因此我們可以把它叫作評分矩陣。由於該矩陣的數值可能用於表示精確的評分，因此我們將會互換地使用互動矩陣和評分矩陣。因為使用者尚未評價大部分電影，因此評分矩陣中的大部分數值都是未知的。我們將會展示該矩陣的稀疏性。此處稀疏度的定義為`1 - 非零實體的數量 / ( 使用者數量 * 物品數量)`。顯然，該矩陣非常稀疏（稀疏度為93.695%）。現實世界中的係數矩陣可能會面臨更嚴重的稀疏問題，該問題也一直是建立推薦系統所面臨的長期挑戰。一個可行的解決方案是，使用額外的輔助訊息，例如使用者和物品特徵，來消除這種稀疏性。

接下來，我們繪製評分計數的分佈情況。正如預期的一樣，該分佈看起來像是一個常態分佈，大部分評分資料都集中在3-4之間。

```python
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## 分割資料集

我們將資料集切分為訓練集和測試集兩部分。下面的函式提供了`隨機`和`序列感知`兩種分割模式。在`隨機`模式下，該函式將忽略時間戳，然後隨機切分100k的互動資料。在預設情況下，其中90%的資料將作為訓練樣本，剩餘的10%用作測試樣本。在`序列感知`模式下，我們利用時間戳排序使用者的歷史評分，然後將使用者的最新評分用於測試，將其餘的評分用作訓練集。這一模式會用在序列感知推薦的小節中。

```python
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """以隨機模式或者序列感知模式分割資料集"""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time) # 最新的評分
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()] 
        train_data = [item for item in train_list if item not in test_data] # 移除測試資料集中已有的評分
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio] # np.random.uniform(0,1,len(data)<1-test_ratio).tolist()
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

請注意，在日常實踐中，除了測試集最好還要有驗證集。但是簡潔起見，我們在這裡忽略了驗證集。在這種情況下，我們的測試集可以視作保留的驗證集。

## 載入資料

分割資料集後，為了方面使用，我們將訓練集和測試集轉化為了清單和字典（或者矩陣）。下面的函式按行讀取dataframe中資料，並且從0開始列舉使用者和物品的索引。該函式的回傳值為使用者、物品和評分清單，以及一個記錄了互動資料的字典或者矩陣。我們可以將回傳的型別指定為`顯式`或者`隱式`。

```python
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    """載入MovieLens-100k資料集"""
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

接下來，為了在之後的章節使用資料集，我們整合上述步驟。這裡得到的結果將會封裝到`Dataset`和`DataLoader`之中。請注意，訓練資料的`DataLoader`的`last_batch`選項被設定為了`rollover`（剩餘樣本將捲動到下一週期），而且資料的順序也是打亂的。

```python
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    """分割並載入MovieLens-100k資料集"""
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## 小結

* MovieLens廣泛用於推薦系統研究。它是免費且公開可用的。
* 為了能在後續章節中使用，我們定義了一些函式用來下載和預處理MovieLens-100k資料集。

## 練習

* 你可以找到其他類似的推薦資料集嗎？
* 你可以在[https://movielens.org/](https://movielens.org/)上瀏覽到關於MovieLens資料集更多的訊息。

[Discussions](https://discuss.d2l.ai/t/399)

