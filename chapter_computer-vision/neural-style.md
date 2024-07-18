# 風格遷移

攝影愛好者也許接觸過濾波器。它能改變照片的顏色風格，從而使風景照更加銳利或者令人像更加美白。但一個濾波器通常只能改變照片的某個方面。如果要照片達到理想中的風格，可能需要嘗試大量不同的組合。這個過程的複雜程度不亞於模型調參。

本節將介紹如何使用卷積神經網路，自動將一個影象中的風格應用在另一影象之上，即*風格遷移*（style transfer） :cite:`Gatys.Ecker.Bethge.2016`。
這裡我們需要兩張輸入影象：一張是*內容影象*，另一張是*風格影象*。
我們將使用神經網路修改內容影象，使其在風格上接近風格影象。
例如， :numref:`fig_style_transfer`中的內容影象為本書作者在西雅圖郊區的雷尼爾山國家公園拍攝的風景照，而風格影象則是一幅主題為秋天橡樹的油畫。
最終輸出的合成影象應用了風格影象的油畫筆觸讓整體顏色更加鮮豔，同時保留了內容影象中物體主體的形狀。

![輸入內容影象和風格影象，輸出風格遷移後的合成影象](../img/style-transfer.svg)
:label:`fig_style_transfer`

## 方法

 :numref:`fig_style_transfer_model`用簡單的例子闡述了基於卷積神經網路的風格遷移方法。
首先，我們初始化合成影象，例如將其初始化為內容影象。
該合成影象是風格遷移過程中唯一需要更新的變數，即風格遷移所需迭代的模型引數。
然後，我們選擇一個預訓練的卷積神經網路來抽取影象的特徵，其中的模型引數在訓練中無須更新。
這個深度卷積神經網路憑藉多個層逐級抽取影象的特徵，我們可以選擇其中某些層的輸出作為內容特徵或風格特徵。
以 :numref:`fig_style_transfer_model`為例，這裡選取的預訓練的神經網路含有3個卷積層，其中第二層輸出內容特徵，第一層和第三層輸出風格特徵。

![基於卷積神經網路的風格遷移。實線箭頭和虛線箭頭分別表示前向傳播和反向傳播](../img/neural-style.svg)
:label:`fig_style_transfer_model`

接下來，我們透過前向傳播（實線箭頭方向）計算風格遷移的損失函式，並透過反向傳播（虛線箭頭方向）迭代模型引數，即不斷更新合成影象。
風格遷移常用的損失函式由3部分組成：

1. *內容損失*使合成影象與內容影象在內容特徵上接近；
1. *風格損失*使合成影象與風格影象在風格特徵上接近；
1. *全變分損失*則有助於減少合成影象中的噪點。

最後，當模型訓練結束時，我們輸出風格遷移的模型引數，即得到最終的合成影象。

在下面，我們將透過程式碼來進一步瞭解風格遷移的技術細節。

## [**閱讀內容和風格影象**]

首先，我們讀取內容和風格影象。
從列印出的影象坐標軸可以看出，它們的尺寸並不一樣。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import paddle
import paddle.vision as paddlevision
import paddle.nn as nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch, paddle
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**預處理和後處理**]

下面，定義影象的預處理函式和後處理函式。
預處理函式`preprocess`對輸入影象在RGB三個通道分別做標準化，並將結果變換成卷積神經網路接受的輸入格式。
後處理函式`postprocess`則將輸出影象中的畫素值還原回標準化之前的值。
由於影象列印函式要求每個畫素的浮點數值在0～1之間，我們對小於0和大於1的值分別取0和1。

```{.python .input}
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

```{.python .input}
#@tab paddle
rgb_mean = paddle.to_tensor([0.485, 0.456, 0.406])
rgb_std = paddle.to_tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = paddlevision.transforms.Compose([
        paddlevision.transforms.Resize(image_shape),
        paddlevision.transforms.ToTensor(),
        paddlevision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0]
    img = paddle.clip(img.transpose((1, 2, 0)) * rgb_std + rgb_mean, 0, 1)
    return img
```

## [**抽取影象特徵**]

我們使用基於ImageNet資料集預訓練的VGG-19模型來抽取影象特徵 :cite:`Gatys.Ecker.Bethge.2016`。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

```{.python .input}
#@tab paddle
pretrained_net = paddlevision.models.vgg19(pretrained=True)
```

為了抽取影象的內容特徵和風格特徵，我們可以選擇VGG網路中某些層的輸出。
一般來說，越靠近輸入層，越容易抽取影象的細節訊息；反之，則越容易抽取影象的全域訊息。
為了避免合成影象過多保留內容影象的細節，我們選擇VGG較靠近輸出的層，即*內容層*，來輸出影象的內容特徵。
我們還從VGG中選擇不同層的輸出來對應區域性和全域的風格，這些圖層也稱為*風格層*。
正如 :numref:`sec_vgg`中所介紹的，VGG網路使用了5個卷積區塊。
實驗中，我們選擇第四卷積區塊的最後一個卷積層作為內容層，選擇每個卷積區塊的第一個卷積層作為風格層。
這些層的索引可以透過列印`pretrained_net`實例獲取。

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

使用VGG層抽取特徵時，我們只需要用到從輸入層到最靠近輸出層的內容層或風格層之間的所有層。
下面建立一個新的網路`net`，它只保留需要用到的VGG的所有層。

```{.python .input}
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch, paddle
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

給定輸入`X`，如果我們簡單地呼叫前向傳播`net(X)`，只能獲得最後一層的輸出。
由於我們還需要中間層的輸出，因此這裡我們逐層計算，並保留內容層和風格層的輸出。

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

下面定義兩個函式：`get_contents`函式對內容影象抽取內容特徵；
`get_styles`函式對風格影象抽取風格特徵。
因為在訓練時無須改變預訓練的VGG的模型引數，所以我們可以在訓練開始之前就提取出內容特徵和風格特徵。
由於合成影象是風格遷移所需迭代的模型引數，我們只能在訓練過程中透過呼叫`extract_features`函式來抽取合成影象的內容特徵和風格特徵。

```{.python .input}
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab paddle
def get_contents(image_shape):
    content_X = preprocess(content_img, image_shape)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape):
    style_X = preprocess(style_img, image_shape)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**定義損失函式**]

下面我們來描述風格遷移的損失函式。
它由內容損失、風格損失和全變分損失3部分組成。

### 內容損失

與線性迴歸中的損失函式類似，內容損失透過平方誤差函式衡量合成影象與內容影象在內容特徵上的差異。
平方誤差函式的兩個輸入均為`extract_features`函式計算所得到的內容層的輸出。

```{.python .input}
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # 我們從動態計算梯度的樹中分離目標：
    # 這是一個規定的值，而不是一個變數。
    return torch.square(Y_hat - Y.detach()).mean()
```

```{.python .input}
#@tab paddle
def content_loss(Y_hat, Y):
    # 我們從動態計算梯度的樹中分離目標：
    # 這是一個規定的值，而不是一個變數。
    return paddle.square(Y_hat - Y.detach()).mean()
```

### 風格損失

風格損失與內容損失類似，也透過平方誤差函式衡量合成影象與風格影象在風格上的差異。
為了表達風格層輸出的風格，我們先透過`extract_features`函式計算風格層的輸出。
假設該輸出的樣本數為1，通道數為$c$，高和寬分別為$h$和$w$，我們可以將此輸出轉換為矩陣$\mathbf{X}$，其有$c$行和$hw$列。
這個矩陣可以被看作由$c$個長度為$hw$的向量$\mathbf{x}_1, \ldots, \mathbf{x}_c$組合而成的。其中向量$\mathbf{x}_i$代表了通道$i$上的風格特徵。

在這些向量的*格拉姆矩陣*$\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$中，$i$行$j$列的元素$x_{ij}$即向量$\mathbf{x}_i$和$\mathbf{x}_j$的內積。它表達了通道$i$和通道$j$上風格特徵的相關性。我們用這樣的格拉姆矩陣來表達風格層輸出的風格。
需要注意的是，當$hw$的值較大時，格拉姆矩陣中的元素容易出現較大的值。
此外，格拉姆矩陣的高和寬皆為通道數$c$。
為了讓風格損失不受這些值的大小影響，下面定義的`gram`函式將格拉姆矩陣除以了矩陣中元素的個數，即$chw$。

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

自然地，風格損失的平方誤差函式的兩個格拉姆矩陣輸入分別基於合成影象與風格影象的風格層輸出。這裡假設基於風格影象的格拉姆矩陣`gram_Y`已經預先計算好了。

```{.python .input}
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

```{.python .input}
#@tab paddle
def style_loss(Y_hat, gram_Y):
    return paddle.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### 全變分損失

有時候，我們學到的合成影象裡面有大量高頻噪點，即有特別亮或者特別暗的顆粒畫素。
一種常見的去噪方法是*全變分去噪*（total variation denoising）：
假設$x_{i, j}$表示坐標$(i, j)$處的畫素值，降低全變分損失

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

能夠儘可能使鄰近的畫素值相似。

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### 損失函式

[**風格轉移的損失函式是內容損失、風格損失和總變化損失的加權和**]。
透過調節這些權重超引數，我們可以權衡合成影象在保留內容、遷移風格以及去噪三方面的相對重要性。

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分別計算內容損失、風格損失和全變分損失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 對所有損失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**初始化合成影象**]

在風格遷移中，合成的影象是訓練期間唯一需要更新的變數。因此，我們可以定義一個簡單的模型`SynthesizedImage`，並將合成的影象視為模型引數。模型的前向傳播只需回傳模型引數即可。

```{.python .input}
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

```{.python .input}
#@tab paddle
class SynthesizedImage(nn.Layer):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = paddle.create_parameter(shape=img_shape,
                                            dtype="float32")

    def forward(self):
        return self.weight
```

下面，我們定義`get_inits`函式。該函式建立了合成影象的模型實例，並將其初始化為影象`X`。風格影象在各個風格層的格拉姆矩陣`styles_Y_gram`將在訓練前預先計算好。

```{.python .input}
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab paddle
def get_inits(X, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.weight.set_value(X)
    trainer = paddle.optimizer.Adam(parameters = gen_img.parameters(), learning_rate=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**訓練模型**]

在訓練模型進行風格遷移時，我們不斷抽取合成影象的內容特徵和風格特徵，然後計算損失函式。下面定義了訓練迴圈。

```{.python .input}
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab paddle
def train(X, contents_Y, styles_Y, lr, num_epochs, step_size):
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=lr, gamma=0.8, step_size=step_size)
    X, styles_Y_gram, trainer = get_inits(X, scheduler, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.clear_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

現在我們[**訓練模型**]：
首先將內容影象和風格影象的高和寬分別調整為300和450畫素，用內容影象來初始化合成影象。

```{.python .input}
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

```{.python .input}
#@tab paddle
device, image_shape = d2l.try_gpu(),(300, 450)
content_X, contents_Y = get_contents(image_shape)
_, styles_Y = get_styles(image_shape)
output = train(content_X, contents_Y, styles_Y, 0.3, 500, 50)
```

我們可以看到，合成影象保留了內容影象的風景和物體，並同時遷移了風格影象的色彩。例如，合成影象具有與風格影象中一樣的色彩區塊，其中一些甚至具有畫筆筆觸的細微紋理。

## 小結

* 風格遷移常用的損失函式由3部分組成：（1）內容損失使合成影象與內容影象在內容特徵上接近；（2）風格損失令合成影象與風格影象在風格特徵上接近；（3）全變分損失則有助於減少合成影象中的噪點。
* 我們可以透過預訓練的卷積神經網路來抽取影象的特徵，並透過最小化損失函式來不斷更新合成影象來作為模型引數。
* 我們使用格拉姆矩陣表達風格層輸出的風格。

## 練習

1. 選擇不同的內容和風格層，輸出有什麼變化？
1. 調整損失函式中的權重超引數。輸出是否保留更多內容或減少更多噪點？
1. 替換實驗中的內容影象和風格影象，能創作出更有趣的合成影象嗎？
1. 我們可以對文字使用風格遷移嗎？提示:可以參閱調查報告 :cite:`Hu.Lee.Aggarwal.ea.2020`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3299)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3300)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11813)
:end_tab:
