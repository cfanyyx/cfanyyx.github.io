---
layout: post
title: "谈谈Android里的getpixels"
subtitle: 'Talking about getPixels in Android'
author: "cfanyyx"
header-style: text
tags:
  - Android
---

## 一. 事故

最近搞了一下Android的工程方面的东西，然后遇到一个坑，就是在用getPixels方法的时候，拿到的图片像素值中alpha通道的值不正确，全是255，后来翻了一下源码，
发现会调一个native的getPixels方法，然后就没有深入调研了，等有时间再跟。大致就是本来应该是alpha为0的值，但是使用getPixels之后，该点的像素值为-16777216，
而-16777216的补码十六进制表示是#FF000000，也就是argb下不透明的黑色，但是但是FF本应该是00，难道是因为位数不够？（我是不是应该试一下用long来存？）

## 二. 解决

换ByteBuffer来做这个事情：

```java
ByteBuffer gtByteBuffer = ByteBuffer.allocate(gtBitmap.getWidth() * gtBitmap.getHeight() * 4);
gtBitmap.copyPixelsToBuffer(gtByteBuffer);
gtByteBuffer.position(0);
byte gtByteArray[] = new byte[gtBitmap.getWidth() * gtBitmap.getHeight() * 4];
gtByteBuffer.get(gtByteArray);

byte gtR = 0, gtG = 0, gtB = 0, gtA = 0;
int unionPixels = 0, interPixels = 0;
for(int i = 0; i < gtByteArray.length; i+=4){
    gtR = gtByteArray[i + 0];
    gtG = gtByteArray[i + 1];
    gtB = gtByteArray[i + 2];
    gtA = gtByteArray[i + 3];
}
```

等后续调研清楚了再来填坑......

*-The End-*
