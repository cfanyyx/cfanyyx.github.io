---
layout: post
title: "TensorRT就是个深坑"
subtitle: 'TensorRT means trouble'
author: "cfanyyx"
header-style: text
tags:
  - Tensorrt
  - CUDA
---

## 一. 坑

trt如果只是用于普通的模型转换那么文档中的API说明应当是足够用的，给的示例也足够丰富，但是trt并不是任何操作都支持的啊，那不支持的操作你要么就要用已经有的方法绕过，要么就是自己去实现，而这个就会涉及到trt的插件编写，那么trt的官方文档在插件这块描述的就不足够，给的例子也比较偏，偏主要是偏在插件的例子都是基于模型转换的，没有自己手动写网络的例子，于是自己写网络该怎么样用插件真是一头雾水，还好有两个github可以供参考，减少了很多自己手动去探索发现的时间。


## 二. C

涉及到自己去写plugin这个事情就是很麻烦的，这里我要实现的是instance normalization和reflect padding，分析了一下两个东西之后发现还算是比较好实现的，instance normalization主要是计算每个通道的均值和方差，然后整个通道用求得的mean和var做normalization即可；而reflect padding可以先使用自带的zero padding把padding先加上（这里这样做主要是不太明白怎么样加padding会更快，毕竟加padding会更改特征图的大小，复制原特征图内容怎样更快就丢给tensorrt本身去做了），然后在根据reflect的规则修改每个padding的值。

方案确定后就开始写吧，怎么写呢？没有一个参考的文档真的是痛苦，于是几经辗转挖掘到了LitLeo的[TensorRT_Tutorial项目](https://github.com/LitLeo/TensorRT_Tutorial)，主要就是阐述怎么在c环境下编写plugins加入自定义的trt网络中。以下摘自上述git。


> ......  
> 首先来简单介绍IPlugin接口类的成员函数，详细见TensorRT-3.0.0\include\NvInfer.h文件中的类定义。  
> ......  
> 根据类成员函数和leaky relu层的原理，设计LeakyReluPlugin类，可以很容易计算出的成员变量和各个成员函数的返回值。LeakyReluPlugin类实现代码如下。  
> ......  
> 然后插入到网络中即可，代码如下。  
> LeakyReluPlugin *lr = new LeakyReluPlugin();  
> auto plugin_lr = network->addPlugin(&inputs_v[0], 1, *lr);  
> plugin_lr->setName(PLUGIN_LEAKY_RELU_NAME);  

于是我这边的实现方式放在了我自己的git项目[tensorrt_plugins](https://github.com/cfanyyx/tensorrt_plugins)中，c的实现在c_plugins文件夹下。


## 三. Python

写完c之后，本来就结束了，但是整体inference的耗时比较长，于是想尝试一下python实现会不会好一些，问题又来了，python实现也没有文档，于是又是一番搜索，发现了souseiki的[UpsamplingPlugin](https://github.com/souseiki/UpsamplingPlugin)这个git项目，以下摘自上述git。


> ......  
> 如何让Tensorrt感知新的Layer Plugin？    
> 在Plugin实现文件中调用REGISTER_TENSORRT_PLUGIN这个宏，用于注册一个Plugin Creator。例如：  
> REGISTER_TENSORRT_PLUGIN(UpsamplePluginCreator);  
> 有了这个宏，当在.py文件中调用xxplugin.so的时候就会自动执行这个语句，然后就会在tensorrt中注册UpsamplePluginCreator的信息，可以用于创建新的Plugin,实际的效果就是在  
> trt.get_plugin_registry().plugin_creator_list添加了一个UpsamplePluginCreator  
> ......  
> python的调用方式：  
> 1.获取tensorrt中的creator列表。代码如下：  
> trt.init_libnvinfer_plugins(TRT_LOGGER, '')  
> PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list  
> 2.有了上面的列表，就可以根据名字匹配相应的 Plugin Creator，并且传入相应的参数，构建对应的plugin。代码如下：  
> def get_upsample_plugin(plugin_name, sacle_factor=2, align_corners=False):  
>     plugin = None  
>     for plugin_creator in PLUGIN_CREATORS:  
>       if plugin_creator.name == plugin_name:  
>           scale_factor_field = trt.PluginField("scaleFactor", np.array([sacle_factor], dtype=np.int8), trt.PluginFieldType.INT8)  
>           align_corners_field = trt.PluginField("alignCorners", np.array([int(align_corners)], dtype=np.int8), trt.PluginFieldType.INT8)  
>           field_collection = trt.PluginFieldCollection([align_corners_field, scale_factor_field])  
>           plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)  
>   return plugin  
> Note: 参数的载入tensorrt使用的是trt.PluginField，第一个参数是名字，第二个是参数的内存地址（buffer类型， 一般用numpy来实现），第三个是类型。名字和类型必须跟你在Creator中使用的一样，不然报错  
> 3.创建好了Plugin，就可以用network.add_plugin_v2调用了。代码如下：  
> upsample_layer = network.add_plugin_v2(inputs=[inputs], plugin=get_upsample_plugin("UpsamplePlugin", sacle_factor, align_corners))  

我自己的实现放在了[tensorrt_plugins](https://github.com/cfanyyx/tensorrt_plugins)的python_plugins文件夹下。


## 四. 后续坑

1.我的instance normalization的实现方式是各个通道分开单独计算各个通道的mean和var，后面测试时间的时候发现这样的做法比较耗时，但是一时半会儿还想不到更快的实现方式，后续如果有更好的方法再做更新；

2.device变量会在整个程序没有结束的情况下永不清零，所以在需要的时候需要在合适的地方手动清零；

3.设置gridDim的时候有两种方式：①(n+blockSize-1)/blockSize以及②(n/blockSize)+1两种，但是②方案在边界值（n=blockSize或者blockSize-1)的时候是有问题的，所以应该使用①方案来设置gridDim

*-The End-*
