---
layout: post
title: "谈谈upsampling"
subtitle: 'Talking about upsampling'
author: "cfanyyx"
header-style: text
tags:
  - Deep learning
---

## 一. 背景

超分，GAN，这些神经网络或多或少都会涉及到上采样的步骤，也就是upsampling，最近也在看相关的资料，发现有一个总结得还比较好，就是[Super-Resolution](https://wiki.tum.de/display/lfdv/Super-Resolution)这篇文章，其主要概括了当前在做超分的一些方法，包括传统和深度的方法，以及它们的对比。这其中就会涉及到upsampling的过程。传统的方法往往是在opencv，PIL的API中可以看到的什么NearestNeighbor以及各种线性插值，深度的方法就会涉及到反卷积，反池化这些操作。文中也提到了一些衡量超分结果的指标，但是这些指标也就是个数值，真正看图片的质量感觉还是要用人肉眼来看比较靠谱。最后逛知乎，发现一篇宝藏文章[从SRCNN到EDSR，总结深度学习端到端超分辨率方法发展历程](https://zhuanlan.zhihu.com/p/31664818)，这篇文章全面总结了超分深度做法的发展历程，还给出了所有的文章及代码的链接。

## 二. 各种upsampling

看了这么多超分的文章，总结下来upsampling无非就是以下几种吧：

### 1.首先就是unpooling
通常用unpooling来逆向一些pooling的操作，比如记住max pooling时取max的位置（switch variables），然后用unpooling进行还原。

### 2.upsampling+convolution
比较有代表性就是这篇文章->[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)，这篇文章主要意思就是说，如果使用deconvolution layer去做upsampling的话，当设置的kernal size不能被stride整除的时候，反卷积出来的feature map就可能在交界处产生重叠，于是会出现文中所说的checkerboard现象，而使用upsampling+convolution替换deconvolution就能避免这种现象的发生。但事实上，本人在做超分+风格迁移的实验过程中使用这个方式进行upsampling并没有抑制checkerboard现象的出现，反而在生成的图像上还带来了无法消除的网格状的“斑块”现象。所以现在严重怀疑upsampling+convolution的能力稍弱于deconvolution。

### 3.deconvolution
其实就是一种卷积的方式，有专门的论文来论证这一操作（Is the deconvolution layer the same as a convolutional layer?），通过可学习的方式来做上采样的工作，其能力要强于upsampling+convoliton这种结构。

### 4.sub-pixel convolution
这篇文章主要涉及到的就是一个超分精细化的操作，感觉它既不能算作convolution，又不能算作单纯的upsamling，但是它出来的结果感觉会很exquisite。这大概得益于它对upsampling过程的一个改造设计。

![20190318_sub-pixel-convolution](/img/in-post/20190318/20190318_sub-pixel-convolution.jpg)

其实上图已经把它的思想具体展示出来了，放上代码的话大概是这个样子的（参考了[【超分辨率】Efficient Sub-Pixel Convolutional Neural Network](https://blog.csdn.net/shwan_ma/article/details/78440394)）：

```python
def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(3, 3, X)
        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X
```

其实在pytorch的代码里面已经有现成的API可以调用了，具体可参考[这个链接](https://blog.csdn.net/gdymind/article/details/82388068)。

### 5.Convolutional Sparse Coding
其实严格意义上来说Sparse Coding是它的前辈，Sparse Coding的思想是说通过学习一个字典matrix D，将一幅图片能够通过D*a来进行最大程度的还原，而通过这一思想，可以将任何低清与高清的图片对转化为低清与高清的两个字典matrix D，并认为任何一个低清与高清的图片组都可以用同一个a来进行表示。

而由其衍生出来的Convolutional Sparse Coding，分为training phase和testing phase，其中training phase需要学习到3组参数，一是低清filters，二是低清feature maps到高清feature maps的映射关系，三是高清的filters。然后在testing phase，低清的图片首先会被低清filters解构成Sparse低清feature maps，之后通过低清feature maps到高清feature maps的映射关系将低清feature maps转换成高清feature maps，最后通过高清的filters重构高清图。然后具体的过程比较复杂，可以参考这篇论文->[Convolutional Sparse Coding for Image Super-resolution](http://openaccess.thecvf.com/content_iccv_2015/papers/Gu_Convolutional_Sparse_Coding_ICCV_2015_paper.pdf)。


### 6.Total Variation Loss
这个tv loss并不能算作是upsampling的方法，而只能算作upsampling过程中的一个小trick，这个loss可以让生成的结果更加平滑，其实就像是加了一个滤波，把图磨平了吧就这样理解，它的代码也是到处都能找到了。TF里面已经有人提了request，所以已经有官方[Total Variation代码](https://www.tensorflow.org/api_docs/python/tf/image/total_variation)了。而pytorch的话，贴一个github吧->[这里](https://github.com/jxgu1016/Total_Variation_Loss.pytorch)。


*-The End-*
