# 微調
:label:`sec_fine_tuning`

前面的一些章節介紹瞭如何在只有6萬張影象的Fashion-MNIST訓練資料集上訓練模型。
我們還描述了學術界當下使用最廣泛的大規模影象資料集ImageNet，它有超過1000萬的影象和1000類別的物體。
然而，我們平常接觸到的資料集的規模通常在這兩者之間。

假如我們想識別圖片中不同型別的椅子，然後向用戶推薦購買連結。
一種可能的方法是首先識別100把普通椅子，為每把椅子拍攝1000張不同角度的影象，然後在收集的影象資料集上訓練一個分類模型。
儘管這個椅子資料集可能大於Fashion-MNIST資料集，但實例數量仍然不到ImageNet中的十分之一。
適合ImageNet的複雜模型可能會在這個椅子資料集上過擬合。
此外，由於訓練樣本數量有限，訓練模型的準確性可能無法滿足實際要求。

為了解決上述問題，一個顯而易見的解決方案是收集更多的資料。
但是，收集和標記資料可能需要大量的時間和金錢。
例如，為了收集ImageNet資料集，研究人員花費了數百萬美元的研究資金。
儘管目前的資料收集成本已大幅降低，但這一成本仍不能忽視。

另一種解決方案是應用*遷移學習*（transfer learning）將從*源資料集*學到的知識遷移到*目標資料集*。
例如，儘管ImageNet資料集中的大多數影象與椅子無關，但在此資料集上訓練的模型可能會提取更通用的影象特徵，這有助於識別邊緣、紋理、形狀和物件組合。
這些類似的特徵也可能有效地識別椅子。

## 步驟

本節將介紹遷移學習中的常見技巧:*微調*（fine-tuning）。如 :numref:`fig_finetune`所示，微調包括以下四個步驟。

1. 在源資料集（例如ImageNet資料集）上預訓練神經網路模型，即*源模型*。
1. 建立一個新的神經網路模型，即*目標模型*。這將複製源模型上的所有模型設計及其引數（輸出層除外）。我們假定這些模型引數包含從源資料集中學到的知識，這些知識也將適用於目標資料集。我們還假設源模型的輸出層與源資料集的標籤密切相關；因此不在目標模型中使用該層。
1. 向目標模型新增輸出層，其輸出數是目標資料集中的類別數。然後隨機初始化該層的模型引數。
1. 在目標資料集（如椅子資料集）上訓練目標模型。輸出層將從頭開始進行訓練，而所有其他層的引數將根據源模型的引數進行微調。

![微調。](../img/finetune.svg)
:label:`fig_finetune`

當目標資料集比源資料集小得多時，微調有助於提高模型的泛化能力。

## 熱狗識別

讓我們透過具體案例示範微調：熱狗識別。
我們將在一個小型資料集上微調ResNet模型。該模型已在ImageNet資料集上進行了預訓練。
這個小型資料集包含數千張包含熱狗和不包含熱狗的影象，我們將使用微調模型來識別影象中是否包含熱狗。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
from paddle import nn
import paddle
import paddle.vision as paddlevision
import os
```

### 獲取資料集

我們使用的[**熱狗資料集來源於網路**]。
該資料集包含1400張熱狗的『正類』影象，以及包含儘可能多的其他食物的『負類』影象。
含著兩個類別的1000張圖片用於訓練，其餘的則用於測試。

解壓下載的資料集，我們獲得了兩個資料夾`hotdog/train`和`hotdog/test`。
這兩個資料夾都有`hotdog`（有熱狗）和`not-hotdog`（無熱狗）兩個子資料夾，
子資料夾內都包含相應類別的影象。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

我們建立兩個實例來分別讀取訓練和測試資料集中的所有影象檔案。

```{.python .input}
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab paddle
train_imgs = paddlevision.datasets.DatasetFolder(os.path.join(data_dir, 'train'))
test_imgs = paddlevision.datasets.DatasetFolder(os.path.join(data_dir, 'test'))
```

下面顯示了前8個正類樣本圖片和最後8張負類樣本圖片。正如所看到的，[**影象的大小和縱橫比各有不同**]。

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

在訓練期間，我們首先從影象中裁切隨機大小和隨機長寬比的區域，然後將該區域縮放為$224 \times 224$輸入影象。
在測試過程中，我們將影象的高度和寬度都縮放到256畫素，然後裁剪中央$224 \times 224$區域作為輸入。
此外，對於RGB（紅、綠和藍）顏色通道，我們分別*標準化*每個通道。
具體而言，該通道的每個值減去該通道的平均值，然後將結果除以該通道的標準差。

[~~資料增廣~~]

```{.python .input}
# 使用RGB通道的均值和標準差，以標準化每個通道
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# 使用RGB通道的均值和標準差，以標準化每個通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab paddle
# 使用RGB通道的均值和標準差，以標準化每個通道
normalize = paddle.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = paddlevision.transforms.Compose([
    paddlevision.transforms.RandomResizedCrop(224),
    paddlevision.transforms.RandomHorizontalFlip(),
    paddlevision.transforms.ToTensor(),
    normalize])

test_augs = paddlevision.transforms.Compose([
    paddlevision.transforms.Resize(256),
    paddlevision.transforms.CenterCrop(224),
    paddlevision.transforms.ToTensor(),
    normalize])
```

### [**定義和初始化模型**]

我們使用在ImageNet資料集上預訓練的ResNet-18作為源模型。
在這裡，我們指定`pretrained=True`以自動下載預訓練的模型引數。
如果首次使用此模型，則需要連線網際網路才能下載。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

```{.python .input}
#@tab paddle
pretrained_net = paddlevision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
預訓練的源模型實例包含兩個成員變數：`features`和`output`。
前者包含除輸出層以外的模型的所有層，後者是模型的輸出層。
此劃分的主要目的是促進對除輸出層以外所有層的模型引數進行微調。
源模型的成員變數`output`如下所示。
:end_tab:

:begin_tab:`pytorch`
預訓練的源模型實例包含許多特徵層和一個輸出層`fc`。
此劃分的主要目的是促進對除輸出層以外所有層的模型引數進行微調。
下面給出了源模型的成員變數`fc`。
:end_tab:

:begin_tab:`paddle`
預訓練的源模型實例包含許多特徵層和一個輸出層`fc`。
此劃分的主要目的是促進對除輸出層以外所有層的模型引數進行微調。
下面給出了源模型的成員變數`fc`。
:end_tab:

```{.python .input}
pretrained_net.output
```

```{.python .input}
#@tab pytorch, paddle
pretrained_net.fc
```

在ResNet的全域平均彙聚層後，全連線層轉換為ImageNet資料集的1000個類輸出。
之後，我們建立一個新的神經網路作為目標模型。
它的定義方式與預訓練源模型的定義方式相同，只是最終層中的輸出數量被設定為目標資料集中的類數（而不是1000個）。

在下面的程式碼中，目標模型`finetune_net`中成員變數`features`的引數被初始化為源模型相應層的模型引數。
由於模型引數是在ImageNet資料集上預訓練的，並且足夠好，因此通常只需要較小的學習率即可微調這些引數。

成員變數`output`的引數是隨機初始化的，通常需要更高的學習率才能從頭開始訓練。
假設`Trainer`實例中的學習率為$\eta$，我們將成員變數`output`中引數的學習率設定為$10\eta$。

```{.python .input}
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# 輸出層中的學習率比其他層的學習率大十倍
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

```{.python .input}
#@tab paddle
finetune_net = paddlevision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(pretrained_net.fc.state_dict()['weight'].shape[0], 2)
nn.initializer.XavierUniform(pretrained_net.fc.state_dict()['weight']);
```

### [**微調模型**]

首先，我們定義了一個訓練函式`train_fine_tuning`，該函式使用微調，因此可以多次呼叫。

```{.python .input}
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# 如果param_group=True，輸出層中的模型引數將使用十倍的學習率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab paddle
# 如果param_group=True，輸出層中的模型引數將使用十倍的學習率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = paddle.io.DataLoader(paddle.vision.datasets.DatasetFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = paddle.io.DataLoader(paddle.vision.datasets.DatasetFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = paddle.optimizer.SGD(learning_rate=learning_rate, parameters=[{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'learning_rate': learning_rate * 10}],
                                    weight_decay=0.001)
    else:
        trainer = paddle.optimizer.SGD(learning_rate=learning_rate, parameters=net.parameters(), 
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

我們[**使用較小的學習率**]，透過*微調*預訓練獲得的模型引數。

```{.python .input}
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch, paddle
train_fine_tuning(finetune_net, 5e-5)
```

[**為了進行比較，**]我們定義了一個相同的模型，但是將其(**所有模型引數初始化為隨機值**)。
由於整個模型需要從頭開始訓練，因此我們需要使用更大的學習率。

```{.python .input}
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

```{.python .input}
#@tab paddle
scratch_net = paddlevision.models.resnet18()
scratch_net.fc = nn.Linear(pretrained_net.fc.state_dict()['weight'].shape[0], 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

意料之中，微調模型往往表現更好，因為它的初始引數值更有效。

## 小結

* 遷移學習將從源資料集中學到的知識*遷移*到目標資料集，微調是遷移學習的常見技巧。
* 除輸出層外，目標模型從源模型中複製所有模型設計及其引數，並根據目標資料集對這些引數進行微調。但是，目標模型的輸出層需要從頭開始訓練。
* 通常，微調引數使用較小的學習率，而從頭開始訓練輸出層可以使用更大的學習率。

## 練習

1. 繼續提高`finetune_net`的學習率，模型的準確性如何變化？
2. 在比較實驗中進一步調整`finetune_net`和`scratch_net`的超引數。它們的準確性還有不同嗎？
3. 將輸出層`finetune_net`之前的引數設定為源模型的引數，在訓練期間不要更新它們。模型的準確性如何變化？提示：可以使用以下程式碼。

```{.python .input}
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

```{.python .input}
#@tab paddle
for param in finetune_net.parameters():
    param.stop_gradient = True
```

4. 事實上，`ImageNet`資料集中有一個『熱狗』類別。我們可以透過以下程式碼獲取其輸出層中的相應權重引數，但是我們怎樣才能利用這個權重引數？

```{.python .input}
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

```{.python .input}
#@tab paddle
weight = pretrained_net.fc.weight
hotdog_w = paddle.split(weight.T, 1000, axis=0)[713]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2893)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2894)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11802)
:end_tab:
