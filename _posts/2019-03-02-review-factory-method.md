---
layout: post
title: "跟着Distiller复习工厂模式"
subtitle: 'Fellow Distiller, and you will konw what is factory-method'
author: "cfanyyx"
header-style: text
tags:
  - Python
  - Model compress
  - Design pattern
  - YAML
---

## 一. 前奏

3月了，最近跟“前奏”这个词打交道很多，所以来说说最近，最近在看python的distiller框架，这个框架是做模型压缩的，所以同样我们不看模型压缩而是来看看里面涉及到的工厂模式，我觉得还蛮有意思的，主要好久不看设计模式了，所以也顺便复习一下，好的，快点开始快点结束吧。

## 二. YAML

在distiller中，各种模型压缩方法都通过配置文件来方便用户编写，这里使用的是YAML，全称也是相当别致，叫做Yet Another Markup Language，它有点像XML，但是好像又没有XML复杂，因为它不包含标签什么的东西。这里举一个栗子吧，distiller里面的配置文件大概长这个样子：

```xml
version: 1
pruners:
  filter_pruner_1:
    class: 'L1RankedStructureParameterPruner'
    group_type: Filters
    desired_sparsity: 0.6
    weights: [
      netG.model.13.weight]

  filter_pruner_2:
    class: 'L1RankedStructureParameterPruner'
    group_type: Filters
    desired_sparsity: 0.5
    weights: [
      netG.model.23.conv_block.5.weight]

  filter_pruner_3:
    class: 'L1RankedStructureParameterPruner'
    group_type: Filters
    desired_sparsity: 0.1
    weights: [netG.model.25.weight]


extensions:
  net_thinner:
      class: 'FilterRemover'
      thinning_func_str: remove_filters
      arch: 'resnet56_cifar'
      dataset: 'cifar10'

policies:
  - pruner:
      instance_name: filter_pruner_1
    epochs: [2]

  - pruner:
      instance_name: filter_pruner_2
    epochs: [2]

  - pruner:
      instance_name: filter_pruner_3
    epochs: [2]

  - extension:
      instance_name: net_thinner
    epochs: [2]
```

然后在代码里面，使用了OrderedDict去load这个YAML文件，详细方法参考了[stackoverflow上的一个question](https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts) 。于是我们就可以拿到一个OrderedDict对象。后面就会通过工厂模式解析这个OrderedDict对象。

## 三.工厂模式

工厂分为简单工厂和抽象工厂，抽象工厂就是把工厂抽象出来，方便后续添加新的“产品”对象，而不用每添加一个“产品”就要去修改工厂类，做一个抽象的工厂类，要添加新“产品”，就可以通过先添加新工厂，然后在新工厂里面造新“产品”。

然后在distiller中，其具体解析OrderedDict的过程大致是：
- 首先通过global()方法，获得当前全局变量的字典，包括你导入的对象等等，然后会根据配置文件中配置的class str，在全局变量字典里面找到相应的类，然后将YAML配置文件中使用到的参数传给这个class的init方法从而构建该class，这里有一个很神奇的操作是我之前不太懂的，就是我们可以通过**方法将形参转成dict字典对象，或者是将dict字典对象转换为方法的形参们，举栗如下：

```python
def greet_me(name, n_type):
    print(name)
    print(n_type)

test_dict = {'name':'cfanyyx', 'n_type':1}

>>> greet_me(**test_dict)
cfanyyx
1
```

```python
def greet_me(**kwargs):
    if kwargs is not None:
        for key, value in kwargs.items():
            print("%s == %s" %(key,value))

>>> greet_me(name="cfanyyx", n_type=1)
name == cfanyyx
n_type == 1
```

这里还想分享一个最近用到的，把dict对象转换成类对象的方法，真是超级好用：

```python
def obj_dic(d):  
    top = type('new', (object,), d)  
    seqs = tuple, list, set, frozenset  
    for i, j in d.items():  
        if isinstance(j, dict):  
            setattr(top, i, obj_dic(j))  
        elif isinstance(j, seqs):  
            setattr(top, i,   
                type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))  
        else:  
            setattr(top, i, j)  
    return top

# and then you can change every dict object to an class object
>>> opt_class = obj_dic(dict_obj)
```

然后在python中使用工厂模式的好处是，我们根本不用去手动写构造工厂和“产品”的方法，甚至抽象工厂的代码也不用写了，一样可以很轻松添加新的“产品”，你只要事先实现好相应的工厂以及“产品”类文件，确定好它们的层次（fu zi）关系就可以直接用上述的global()方法去造“产品”了，其实上述过程很像反射有木有，或者说它就是反射吧。

举栗如下，比如一个YAML文件中会有很多模型压缩操作，那么我们会通过反射的方法构建这些“产品”，构建过程用到的参数会从YAML文件的前半段对不同“产品”的描述中获得，这里的“产品”就是pruner啊，regularizer啊，quantizer啊这种东西：

```python
pruners = __factory('pruners', model, sched_dict)
regularizers = __factory('regularizers', model, sched_dict)
quantizers = __factory('quantizers', model, sched_dict, optimizer=optimizer)
if len(quantizers) > 1:
	raise ValueError("\nError: Multiple Quantizers not supported")
extensions = __factory('extensions', model, sched_dict)

def __factory(container_type, model, sched_dict, **kwargs):
    container = {}
    if container_type in sched_dict:
        try:
            for name, cfg_kwargs in sched_dict[container_type].items():
                try:
                    cfg_kwargs.update(kwargs)
                    # Instantiate pruners using the 'class' argument
                    cfg_kwargs['model'] = model
                    cfg_kwargs['name'] = name
                    class_ = globals()[cfg_kwargs['class']]
                    container[name] = class_(**__filter_kwargs(cfg_kwargs, class_.__init__))
                except NameError as error:
                    print("\nFatal error while parsing [section:%s] [item:%s]" % (container_type, name))
                    raise
                except Exception as exception:
                    print("\nFatal error while parsing [section:%s] [item:%s]" % (container_type, name))
                    print("Exception: %s %s" % (type(exception), exception))
                    raise
        except Exception as exception:
            print("\nFatal while creating %s" % container_type)
            print("Exception: %s %s" % (type(exception), exception))
            raise

    return container
```

构建完模型压缩的“产品”之后，会继续通过YAML配置文件中的“policies”部分，构建相应的policy，在这个过程中，会check这里所使用的policy是否在模型前半段列举的模型压缩“产品”中出现过。如果出现过那么配置相应的policy对象，并将其添加到scheduler中，完成所有的构建过程。大致过程如下：

```python
if scheduler is None:
    scheduler = distiller.CompressionScheduler(model)
try:
	lr_policies = []
	for policy_def in sched_dict['policies']:
		policy = None
		if 'pruner' in policy_def:
			try:
				instance_name, args = __policy_params(policy_def, 'pruner')
			except TypeError as e:
				print('\n\nFatal Error: a policy is defined with a null pruner')
				print('Here\'s the policy definition for your reference:\n{}'.format(json.dumps(policy_def, indent=1)))
				raise
			assert instance_name in pruners, "Pruner {} was not defined in the list of pruners".format(instance_name)
			pruner = pruners[instance_name]
			policy = distiller.PruningPolicy(pruner, args)

		elif 'regularizer' in policy_def:
			instance_name, args = __policy_params(policy_def, 'regularizer')
			assert instance_name in regularizers, "Regularizer {} was not defined in the list of regularizers".format(instance_name)
			regularizer = regularizers[instance_name]
			if args is None:
				policy = distiller.RegularizationPolicy(regularizer)
			else:
				policy = distiller.RegularizationPolicy(regularizer, **args)

		elif 'quantizer' in policy_def:
			instance_name, args = __policy_params(policy_def, 'quantizer')
			assert instance_name in quantizers, "Quantizer {} was not defined in the list of quantizers".format(instance_name)
			quantizer = quantizers[instance_name]
			policy = distiller.QuantizationPolicy(quantizer)

		elif 'lr_scheduler' in policy_def:
			# LR schedulers take an optimizer in their CTOR, so postpone handling until we're certain
			# a quantization policy was initialized (if exists)
			lr_policies.append(policy_def)
			continue

		elif 'extension' in policy_def:
			instance_name, args = __policy_params(policy_def, 'extension')
			assert instance_name in extensions, "Extension {} was not defined in the list of extensions".format(instance_name)
			extension = extensions[instance_name]
			policy = extension

		else:
			raise ValueError("\nFATAL Parsing error while parsing the pruning schedule - unknown policy [%s]".format(policy_def))

		add_policy_to_scheduler(policy, policy_def, scheduler)

	# Any changes to the optmizer caused by a quantizer have occured by now, so safe to create LR schedulers
	lr_schedulers = __factory('lr_schedulers', model, sched_dict, optimizer=optimizer)
	for policy_def in lr_policies:
		instance_name, args = __policy_params(policy_def, 'lr_scheduler')
		assert instance_name in lr_schedulers, "LR-scheduler {} was not defined in the list of lr-schedulers".format(
			instance_name)
		lr_scheduler = lr_schedulers[instance_name]
		policy = distiller.LRPolicy(lr_scheduler)
		add_policy_to_scheduler(policy, policy_def, scheduler)

except AssertionError:
	# propagate the assertion information
	raise
except Exception as exception:
	print("\nFATAL Parsing error!\n%s" % json.dumps(policy_def, indent=1))
	print("Exception: %s %s" % (type(exception), exception))
	raise
return scheduler
```

之后在distiller做模型压缩的时候就可以在训练的各个阶段通过sceduler调用相应的policy对象，再由policy对象去实际操作模型压缩“产品”从而实现真正的模型压缩过程了。

好了，这篇文章就到这里，一堆事啊一堆事，接踵而至。

*-The End-*
