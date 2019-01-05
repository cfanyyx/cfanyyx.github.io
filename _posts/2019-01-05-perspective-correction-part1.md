---
layout: post
title: "在Android设备上使用OpenCV做透视矫正（一）"
subtitle: 'Perspective Correction on Android Part1'
author: "cfanyyx"
header-style: text
tags:
  - Android
---

## 一. 整体框架

跨年回来，终于感觉可以有点时间写点什么东西，于是把折腾了一阵子的一个用OpenCV搞的东西拿出来记录一下，搞这个的目的一是发现有的P图应用里面连个透视矫正都没有还P个什么图，看着不别扭么，另外一个原因就是对Adobe家的自动矫正技术还蛮好奇是怎样实现的，于是整个工程的整体框架基于github项目[OpenCV_native](https://github.com/xiaoxiaoqingyi/NDKDemos) ，该项目主要介绍了在Android中接入使用OpenCV的三种方式，但是感觉都不是很友好，可能是用惯了python这种语言，然后发现在Android里面用个OpenCV如此麻烦，也是感觉心累。

## 二. 界面的设计

由于主要介绍的是怎样实现，于是界面什么的就一切从简了，顺便能够节能减排。原型的话（呸，什么原型，分明就是实物图）大致就如下图所示了：

![20190105_ui](/img/in-post/20190105/20190105_ui.jpg)

接下来复习一下控件的操作然后一些踩的坑。

## 三.控件的操作

这里用这些控件元素要注意的点有如下一些，好久没写Android界面的我也是遇坑无数：

#### 1. 屏幕旋转之后，要防止Activity的资源被销毁

> 这里使用的是将公共元素放到Application类中以防止丢失的办法解决的

#### 2. “选择图像”调用的逻辑，这里包括了存储权限的获取、获得回调返回的图像数据、获得图像数据后图像发生旋转的问题

> 下面来逐一说明一下以上几个要点，首先关于存储权限的获取，说到权限自然要在manifest里面加上以下二货：
> ```xml
>     <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
>     <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
> ```
> 然后要在Activity里面加上判断是否已经获取到了存储权限的代码，如果没有获得存储权限那就要向用户要权限，不然怎么拿到图片？
> ```java
>    public static boolean isGrantExternalRW(Activity activity) {
>        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && activity.checkSelfPermission(
>                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
>            activity.requestPermissions(new String[]{
>                    Manifest.permission.READ_EXTERNAL_STORAGE,
>                    Manifest.permission.WRITE_EXTERNAL_STORAGE
>            }, PERMISSIONS_CODE);
>            return false;
>        }
>        return true;
>    }
> ```
> 拿到权限之后要进行选择图片的操作：
> ```java
>     public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
>         super.onRequestPermissionsResult(requestCode, permissions, grantResults);
> 
>         if (requestCode == PERMISSIONS_CODE) {
>             for (int i = 0; i < permissions.length; i++) {
>                 String permission = permissions[i];
>                 int grantResult = grantResults[i];
> 
>                 if (permission.equals(Manifest.permission.READ_EXTERNAL_STORAGE)) {
>                     if (grantResult == PackageManager.PERMISSION_GRANTED) {
>                         Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
>                         startActivityForResult(intent, RESULT_LOAD_IMAGE);
>                     } else {
>                         Toast.makeText(context, "You need to grant external rw permission first...", Toast.LENGTH_SHORT).show();
>                     }
>                 }
>             }
>         }
>     }
> ```
> 接下来就是要获得回调返回的图像数据，在重载的onActivityResult方法里面要通过URL拿到图像数据，然后进行图像的解码，需要说明的是，如果直接解码可能会发生图像旋转的现象，这里我们用图像的EXIF信息来对原图的角度进行重置，具体代码如下：
> ```java
>     try {
> 		ExifInterface exif = new ExifInterface(picturePath);
> 		int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1);
> 		Log.d("EXIF", "Exif: " + orientation);
> 		Matrix matrix = new Matrix();
> 		if (orientation == 6) {
> 			matrix.postRotate(90);
> 		}
> 		else if (orientation == 3) {
> 			matrix.postRotate(180);
> 		}
> 		else if (orientation == 8) {
> 			matrix.postRotate(270);
> 		}
> 		myBitmap = Bitmap.createBitmap(myBitmap, 0, 0, myBitmap.getWidth(), myBitmap.getHeight(), matrix, true); // rotating bitmap
> 	}
> 	catch (Exception e) {
> 		Log.d(TAG, "something was wrong when dealing with exif orientation...");
> 	}
> ```

#### 3. “选择图像”，“对比”，以及图像处理的相关操作如果使用不当会导致ImageView显示的图像错乱，所以要准备好几个图像的对象分别进行各个状态下图像的存储

> 这里我使用了三个Bitmap对象对图像进行分状态的存取，它们分别是：
> result保存最终的结果；
> loadedPic保存原始图像用以“对比”；
> processingPic保存处理中的图像，用以在不同的图像处理状态间切换；

#### 4. 之所以使用两个ImageView是因为另外一个ImageView需要显示透视矫正过程中的半透明参考线

#### 5. SeekBar的UI配置也是错综复杂，很麻烦，我用了一个Animation来做刻度的显示

> 但是还有个Bug没有解决，就是我的刻度显示动画效果是用透明度实现的，现在的问题是如果在上一个动画没有执行完的情况下，马上调整SeekBar就会导致刻度的那个控件透明度直接干到0，无法显示了，只有当动画的时间过掉之后才能恢复正常，在setOnSeekBarChangeListener的三个重载方法里面周旋了好久还是没有解决这个问题，闹心，回头再看吧

## 四. 透视的算法

其实这里没有什么特别的，主要是放在JNI里面很蛋疼，那代码写的既不像python，又不像c++，首先碰到的就是JNI头文件的生成问题，就是如果java代码里面涉及到了Android的一些类，会提示找不到相关类的报错，于是需要手动指定一下Android SDK相关类文件的位置

```shell
    javah -class Android_sdk_path/sdk/platforms/(version)/android.jar -d output_h_file_path java_class_with_package_name
```

然后要注意的就是关于透视的实现，透视的实现，之前是使用了传图像数组的方式，后来使用OpenCV透视变换出来的图会呈现雪花状，这里怀疑是图像通道的问题，后来改成传图像内存地址的方式进行操作，就正常了，从内存读图像的代码如下：

```c
    # get image content from mem
    AndroidBitmapInfo inBmpInfo, outBmpInfo;
    void* inPixelsAddress;
    void* outPixelsAddress;
    int ret;
    if ((ret = AndroidBitmap_getInfo(env, bmpIn, &inBmpInfo)) < 0) {
        LOGE("AndroidBitmap_getInfo() bmpIn failed ! error=%d", ret);
        return 0;
    }
    if ((ret = AndroidBitmap_getInfo(env, bmpOut, &outBmpInfo)) < 0) {
        LOGE("AndroidBitmap_getInfo() bmpOut failed ! error=%d", ret);
        return 0;
    }
    LOGI("original image :: width is %d; height is %d; stride is %d; format is %d;flags is %d", inBmpInfo.width, inBmpInfo.height, inBmpInfo.stride, inBmpInfo.format, inBmpInfo.flags, inBmpInfo.stride);
    if ((ret = AndroidBitmap_lockPixels(env, bmpIn, &inPixelsAddress)) < 0) {
        LOGE("AndroidBitmap_lockPixels() bmpIn failed ! error=%d", ret);
    }
    if ((ret = AndroidBitmap_lockPixels(env, bmpOut, &outPixelsAddress)) < 0) {
        LOGE("AndroidBitmap_lockPixels() bmpOut failed ! error=%d", ret);
    }
    Mat inMat(inBmpInfo.height, inBmpInfo.width, CV_8UC4, inPixelsAddress);
    Mat outMat(outBmpInfo.height, outBmpInfo.width, CV_8UC4, outPixelsAddress);

    int w = inBmpInfo.width;
    int h = inBmpInfo.height;

	# after processing, you need unlock the mem content
	AndroidBitmap_unlockPixels(env, bmpIn);
    AndroidBitmap_unlockPixels(env, bmpOut);
```

具体的透视变换代码如下：

```c
    // max angle == 20, 50 / 20 == 2.5
    float angle = abs(progress) / 2.5;

    cv::Point2f src_points[] = {
        cv::Point2f(0, 0),
        cv::Point2f(w, 0),
        cv::Point2f(0, h),
        cv::Point2f(w, h)
    };
    cv::Point2f dst_points[4];
    if (direction == 0) {
        // horizontal
        float dy = w * tan(angle * PI / 180) / (tan(30 * PI / 180) + tan(angle * PI / 180));
        float dx = dy * tan(30 * PI / 180);
        if (progress > 0) {
            // right side
            dst_points[0] = cv::Point2f(0, 0);
            dst_points[1] = cv::Point2f(w + dy, 0 - dx);
            dst_points[2] = cv::Point2f(0, h);
            dst_points[3] = cv::Point2f(w + dy, h + dx);
        } else {
            // left side
            dst_points[0] = cv::Point2f(0 - dy, 0 - dx);
            dst_points[1] = cv::Point2f(w, 0);
            dst_points[2] = cv::Point2f(0 - dy, h + dx);
            dst_points[3] = cv::Point2f(w, h);
        }
        LOGE("h: %d\n", h);
        LOGE("w: %d\n", w);
        LOGE("dy: %f\n", dy);
        LOGE("dx: %f\n", dx);
    } else {
        // vertical
        float dy = h * tan(angle * PI / 180) / (tan(30 * PI / 180) + tan(angle * PI / 180));
        float dx = dy * tan(30 * PI / 180);
        if (progress > 0) {
            // up side
            dst_points[0] = cv::Point2f(0 - dx, 0 - dy);
            dst_points[1] = cv::Point2f(w + dx, 0 - dy);
            dst_points[2] = cv::Point2f(0, h);
            dst_points[3] = cv::Point2f(w, h);
        } else {
            // down side
            dst_points[0] = cv::Point2f(0, 0);
            dst_points[1] = cv::Point2f(w, 0);
            dst_points[2] = cv::Point2f(0 - dx, h + dy);
            dst_points[3] = cv::Point2f(w + dx, h + dy);
        }
    }
    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            LOGE("m[%d][%d] = %d", i, j, M.data[3 * i + j] - '0');
        }
    }

    cv::warpPerspective(inMat, outMat, M, outMat.size(), cv::INTER_LINEAR);
···

关于基本的透视变换就是这么多了，貌似在load OpenCV库的时候还有时延问题，因为如果进入app操作透视变换的命令太快app就会crash，后续再跟一下看具体是什么问题导致的，下次写写自动透视矫正的内容。

*The End*
