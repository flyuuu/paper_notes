---
title: 用于改善质量、稳定性和多样性的可增长式GAN
tags: GANs,NVIDIA, Fly
grammar_cjkRuby: true
---
![real or fake ?][1]

![real or fake ?][2]

![1024 x 1024 images generated using the CELEBA-HQ dataset][1]

## 来源

- [x]  论文：Progressive Growing of GANs for Improved Quality, Stability, and Variation
- [x]  链接：http://research.nvidia.com/publication/2017-10_Progressive-Growing-of
- [x]  推荐理由： Under review as a conference paper at ICLR 2018


## 摘要
- 描述了生成对抗网络的==新的训练方法==

- 关键思想是通过渐进的方式训练生成器和鉴别器：从低分辨率开始，逐步添加新的层次，从而在训练进展中增加更精细的细节

- 还提出了一种增加生成图像变化的简单方法，并且在无监督的CIFAR10中实现了创记录的8.80的初始分数。

- 此外，描述了几个实现细节，这些细节对于抑制生成器和鉴别器之间的不健康竞争非常重要。 

- 提出了一个新的衡量GAN结果的指标，无论是在图像质量和变化方面。 

- 作为额外的贡献，构建了更高质量的CelebA数据集。

## 提出模型
### 模型图
![逐层递增的网络][3]


以往的 GAN 生成低分辨率图片稳定迅速，但生成高分辨率图片困难重重。这篇文章从简单的低分辨率图片开始同时训练生成器和判别器，然后逐层增加分辨率，让训练的难度每层只增加一点点。感觉就像是算法里面的暴力搜索到二分法搜索，大大提高了高分辨率图片的生成速度及质量。

### 其他的一些改进
- 以往没有好的办法去判断 GAN  生成的图片是好是坏，很多时候需要肉眼来看，有很大的主观性，而且人能检查的样本空间不够大。文章的第 5 节介绍了如何用统计的方法来直观的判断生成样本的好坏，采用的思路是在各个尺度上抽取 7x7 个像素的局域碎片，比较生成图片与训练图片的局域结构相似性。 

- GAN 生成图像的多样性不好量化，当判别器过强时生成器可能会塌缩到单个类。这篇文章不添加任何超参数，只是将所有属性在所有空间位置的统计标准差求平均，作为卷积神经网络 Feature Map 的一个常量通道，就得到了更好的多样性 。 

- 使用了一种“local response normalization”方法来对 FeatureMap 做归一化，不清楚与 BatchNormalization 的效果相比有没有提升。 

- 在 WGAN-GP 的正规化项中使用非常大的 gamma 因子，从公式上看当生成图片与训练图片相差过大时，大的 gamma 因子可以让生成分布快速漂移到训练图像分布。

## 算法结果

[六分钟的视频](http://v.youku.com/v_show/id_XMzEyODU1MjE5Mg==.html?tpa=dW5pb25faWQ9MTAzMjUyXzEwMDAwMV8wMV8wMQ+)

## 缺点以及不足
论文最后说了一下还面临的一些情况：

与真实的写实主义相比，还有一段路要走：
- 图片语义敏感性和理解数据集的相关结束还有很大进步空间
- 图像的微观结构也有改进的余地

## reference
http://mp.weixin.qq.com/s/1XkOEIlTD4Igr_Ws2sJvoQ
https://www.leiphone.com/news/201710/tPXkf1dcoGDqv5HD.html
http://research.nvidia.com/publication/2017-10_Progressive-Growing-of


  [1]: ./images/1509608164594.jpg
  [2]: ./images/1509626274201.jpg
  [3]: ./images/1509608211934.jpg