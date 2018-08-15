---
layout: post
title: AI模型互通的新开放生态系统ONNX介绍与实践
date: 2017-12-29 
cover: /img/ML/mobile_ML_ONNX/onnx.png
categories: 机器学习
author: 张永超
--- 

### 导读

不久前，Facebook与微软联合推出了一种开放式的神经网络切换格式——ONNX，它是一种表征深度学习模型的标准，可以实现模型在不同框架之间进行迁移。

ONNX 的全称为“Open Neural Network Exchange”，即“开放的神经网络切换”。顾名思义，该项目的目的是让不同的神经网络开发框架做到互通互用。目前，ONNX 已经得到 PyTorch、Caffe2、CNTK、MXNet 以及包括 Intel、ARM、Huawei、高通、AMD、IBM 等芯片商的支持。

按照该项目的设想，ONNX 的推出主要是为了解决当下AI生态系统的关键问题之一：开发框架的碎片化。现在有大量的不同框架用来构建和执行神经网络，还有其他的机器学习系统，但是它们之间不尽相同，而且无法协同运行。

<img src="/img/ML/mobile_ML_ONNX/fragmentation.png" />

AI的开发技术主要是深度学习神经网络，而神经网络训练和推理通常采用一种主流的深度学习框架。目前，主流的框架有如下这些：

- TensorFlow（Google）
- Caffe/Caffe2（Facebook）
- CNTK（Microsoft）
- MXNet（Amazon主导）
- PyTorch（Facebook主导）

不同的框架有不同的优势，但是框架之间的互通性并不好，甚至没有互通性。开发AI模型时，工程师和研究人员有很多的AI框架可以选择，但是并不能“一次开发，多处直接使用”。面对不同的服务平台，往往需要耗费大量的时间把模型从一个开发平台移植到另一个，让开发者苦不堪言。因此增强框架之间的互通性变的非常重要，例如：

- 情况1：某个框架的某个模型不管是准确度还是性能都非常好，但是和软件整体的架构不相符；
- 情况2：由于框架A表现很好，用它训练了一个神经网络，结果生产环境中使用的是框架B，这就意味着你可能需要将框架A训练的神经网络移植到框架B支持的格式上。


但是如果使用ONNX，就可以消除这种尴尬，例如ONNX可以导出用PyTorch构建的训练模型，并将它与Caffe2结合起来用于推理（详细教程）。这对于研究阶段使用PyTorch构建模型，正式环境使用Caffe2的结构非常合适，且省去了模型移植的时间成本和人力成本等。

目前支持ONNX的框架如下：

<img src="/img/ML/mobile_ML_ONNX/supported_frameworks.png" />

详细支持情况，可查看[http://onnx.ai/getting-started](http://onnx.ai/getting-started "Importing and Exporting from Frameworks")中各项详细说明。

### ONNX 技术概括

根据官方介绍中的内容，ONNX 为可扩展的计算图模型、内部运算器（Operator）以及标准数据类型提供了定义。每个计算图以节点列表的形式组织起来，构成一个非循环的图。节点有一个或多个的输入与输出。每个节点都是对一个运算器的调用。图还会包含协助记录其目的、作者等元数据信息。运算器在图的外部实现，但那些内置的运算器可移植到不同的框架上。每个支持 ONNX 的框架将在匹配的数据类型上提供这些运算器的实现。


概括的说就是，ONNX是在跟踪分析AI框架所生成的神经网络在运行时是如何执行的，然后会利用分析的信息，创建一张可以传输的通用计算图，即符合ONNX标准的计算图。虽然各个框架中的计算表示法不同，但是生成的结果非常相似，因此这样的方式也是行得通的。


那么ONNX标准的推出，为研究者、给开发者带来的意义是什么呢？

### ONNX标准的意义

- 首当其冲就是框架之间的互通性

	开发者能够方便的在不同框架之间切换，为不同的任务选择最优的工具。往往在研发阶段需要的模型属性和产品阶段是不同的，而且不同的框架也会在特定的某个属性上有所优化，例如训练速度、对网络架构的支持、是否支持移动设备等等。如果不在研发阶段更换框架或者将研发阶段的模型进行移植，可能造成项目延迟等。ONNX解决了这个难题，使用ONNX支持的AI框架，则可以灵活选择AI框架，方便的进行模型切换。

- 共享优化


	芯片制造商不断地推出针对神经网络性能有所优化的硬件，如果这个优化频繁发生，那么把优化整合到各个框架是非常耗时的事。但是使用ONNX标准，开发者就可以直接使用优化的框架了。

### 这次没有Google

<img src="/img/ML/mobile_ML_ONNX/no_TensorFlow.png" />

Google 是 TensorFlow 框架的核心主导者，而 TensorFlow 目前是业界的主流，是 GitHub 最受欢迎、生态体系健全度较高的框架。但是目前看来，TensorFlow 官方并没有支持ONNX。但是社区已经有了非官方的ONNX版TensorFlow，详情可参考 [https://github.com/tjingrant/onnx-tf](https://github.com/tjingrant/onnx-tf "onnx-tf")。

### 如何使用

以下是根据ONNX官方教程，进行的实践和遇到的具体问题。Getting Started。
ONNX 的使用总体分为两步：

1. 模型的导出：将使用支持ONNX标准的AI框架训练的模型导出为ONNX格式；

2. 模型的导入：将ONNX格式的模型导入到支持ONNX的另一个AI框架中。

这里我选择模型导出的AI框架是：PyTorch，导出的模型是Apple Core ML所支持的mlmodel格式。

#### 1. ONNX 安装

本地环境：

- macOS High Sierra v10.13.1
- Python 2.7
- Xcode 9.1 command line tools
- conda 4.3.30

ONNX的安装支持Binaries、Docker、Source三种方式，由于我本地常使用Anaconda，因此这里我使用Binaries方式 进行安装：

	conda install -c conda-forge onnx
	
安装完成后如下显示：

<img src="/img/ML/mobile_ML_ONNX/install_success.png" />

#### 2. coremltools 安装

coremltools 的安装较为简单，如果本地环境无误，运行如下指令：

	pip install coremltools

即可完成安装。

#### 3. PyTorch 安装

[PyTorch](http://pytorch.org/ "http://pytorch.org/") ONNX支持的版本目前仅支持从源码Master分支安装，因此如果你本机的环境中已经安装了PyTorch，可能需要重新安装，安装方法可见：[From Source](https://github.com/pytorch/pytorch "https://github.com/pytorch/pytorch")。

需要注意的是，在进行安装时，需要将指令 `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install` 中的MACOSX_DEPLOYMENT_TARGET部分修改为当前你设备安装的macOS版本号。

截止发稿时，PyTorch已经更新了官网版本，最新版本为 0.3.0，已经支持了ONNX，因此你可以直接使用官网的方式安装了。【注意：如果本地已经安装过了，使用此方法安装后，需要重新安装TorchVision (conda install torchvision -c pytorch)，否则部分功能可能不能使用。】

<img src="/img/ML/mobile_ML_ONNX/get_started.png" />

#### 4. 模型算法选择

这里选择的是PyTorch中的一个示例模型算法[Super-resolution](https://github.com/pytorch/examples/tree/master/super_resolution "https://github.com/pytorch/examples/tree/master/super_resolution") 进行实践。

Super-resolution（超分辨率）技术是指由低分辨率（LR, low-resolution）的图像或图像序列恢复出高分辨率（HR, high-resolution）的技术，其原理可参考[图像超分辨率技术（Image Super Resolution）](https://blog.csdn.net/geekmanong/article/details/46368509 "https://blog.csdn.net/geekmanong/article/details/46368509")中的介绍。 Super-resolution（超分辨率）核心思想是用时间分辨率（同一场景的多帧图像序列）换成更高的空间分辨率，实现时间分辨率向空间分辨率的转换，[论文地址](https://arxiv.org/abs/1609.05158 "https://arxiv.org/abs/1609.05158")。

在本文中，将使用一个具有虚拟输入的小型超分辨率模型。

本文主要是讲解如何使用ONNX进行模型转换，对于具体的技术原理可参考文中链接。

#### 5. 模型构建

了解了相关的软件安装和模型之后，我们开始构建Super-resolution模型网络，此部分内容借鉴[PyTorch examples](https://github.com/pytorch/examples/tree/master/super_resolution "https://github.com/pytorch/examples/tree/master/super_resolution")中的实现。


	# Super Resolution model definition in PyTorch
	import torch.nn as nn
	import torch.nn.init as init

	class SuperResolutionNet(nn.Module):
    	def __init__(self, upscale_factor, inplace=False):
        	super(SuperResolutionNet, self).__init__()

        	self.relu = nn.ReLU(inplace=inplace)
        	self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        	self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        	self.conv3 = nn.Conv2d(64, 32, (3, 3,), (1, 1), (1, 1))
        	self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        	self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        	self._initialize_weights()

    	def _initialize_weights(self):
        	init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        	init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        	init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        	init.orthogonal(self.conv4.weight)


    	def forward(self, x):
        	x = self.relu(self.conv1(x))
        	x = self.relu(self.conv2(x))
        	x = self.relu(self.conv3(x))
        	x = self.pixel_shuffle(self.conv4(x))
        	return x

以上模型网络构件完成后，我们就可以创建一个模型了，例如：

	torch_model = SuperResolutionNet(upscale_factor=3)

此模型的结构如下：

	SuperResolutionNet(
  		(relu): ReLU()
  		(conv1): Conv2d (1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  		(conv2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 		(conv3): Conv2d (64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  		(conv4): Conv2d (32, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  		(pixel_shuffle): PixelShuffle(upscale_factor=3)
	)

如果你不想自己再去训练模型，可以下载已经构建好的预训练模型权重文件[pretrained model weights](https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth")，进行模型的构建，例如：

	import torch.utils.model_zoo as model_zoo

	# 加载预训练模型权重文件
	model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
	batch_size = 1    # 这里是一个随机数（这了仅仅是为了演示使用，实际项目中需要调整）

	# 使用权重文件初始化模型
	torch_model.load_state_dict(model_zoo.load_url(model_url))

	# 设置训练模式为False，因为我们仅使用正向传播的方式
	torch_model.train(False)


训练后，得到的模型结构同上。

#### 6. 模型导出为ONNX格式

上面通过构建模型的过程，我们已经得到了模型，为了能够将此模型使用在Core ML框架中，首先需要将此模型导出为ONNX标准格式的模型文件。关于PyTorch中如何导出ONNX格式的模型文件，可以参考[torch.onnx](http://pytorch.org/docs/0.3.0/onnx.html "http://pytorch.org/docs/0.3.0/onnx.html")中的说明和示例。

简单的说，在PyTorch中要导出模型是通过跟踪其工作的方式进行的。要导出模型需要调用 torch.onnx._export() 函数，此函数将会记录模型中的运算符用于计算输出的轨迹，另外此函数需要一个输入张量 x，可以是一个图像或一个随机张量，只要它是正确的大小即可。

	import torch.onnx
	from torch.autograd import Variable

	# 张量 `x` 大小确定
	x = Variable(torch.randn(batch_size, 1, 224, 224), requires_grad=True)

	# 导出模型
	torch_out = torch.onnx._export(
    	torch_model,   # 模型对象
    	x,     # 模型输入（或多元输入的元组） 
    	"onnx-model/super_resolution.onnx",  # 保存文件的路径（或文件对象） 
    	export_params=True)  # 是否存储参数（True：将训练过的参数权重存储在模型文件中；False：不存储）

torch_out 是执行模型后的输出。通常情况下可以忽略这个输出，但是在这里我们将使用它来验证在Core ML中运行导出的模型是否和此值具有计算相同的值。


#### 7. 转换ONNX格式到Core ML支持的mlmodel格式

##### 7.1 安装转换工具onnx-coreml

首先我们要安装能够将ONNX标准格式的模型转换为Apple Core ML格式的工具[onnx-coreml](https://github.com/onnx/onnx-coreml "https://github.com/onnx/onnx-coreml")。安装指令如下：

	pip install  onnx-coreml

安装完成后，就可以使用此工具将ONNX标准格式的模型转换为Apple Core ML格式了。

##### 7.2 加载ONNX格式模型文件

在开始之前，需要先使用ONNX加载之前导出的ONNX格式模型文件到对象中：

	import onnx

	model = onnx.load('super_resolution.onnx')

##### 7.3 转换模型到CoreML

由于我们安装了ONNX 到Core ML的转换工具，因此我们可以直接使用其中的 convert() 函数转换模型：

	import onnx_coreml

	cml = onnx_coreml.convert(model)
	print type(cml)
	cml.save('coreml-model/super_resolution.mlmodel')

这里的cml是格式转换后的coreml model对象，调用其save()方法即可将转换后的对象存储文件，即得到Core ML支持的模型文件。

有了这个mlmodel格式的模型文件后，我们就可以将其应用在 Core ML中进行推理了，这里不再进行此部分的描述，具体使用方法可参考Apple官方给出的mlmodel模型文件相关的使用示例，[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app "https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app")。

### 总结

目前，ONNX所支持的AI框架和工具还较少，整个项目还处于初级阶段，但是已经能够看出，ONNX的推出无疑是对整个AI研究和开发领域极大的福祉。不远的将来，当大多数的AI框架和工具都支持ONNX标准的时候，就不会发生为了选择生产环境的AI框架而重新学习、为了能够生产环境应用模型而耗时移植模型等等类似的事件了。减少了开发人员消耗在移植过程中的时间，增加了钻研算法和开发更令人兴奋的AI应用的时间，相信每个开发者和研究者都盼望着这个时期的到来。

### 参考资料

- [Facebook and Microsoft introduce new open ecosystem for interchangeable AI frameworks](https://research.fb.com/facebook-and-microsoft-introduce-new-open-ecosystem-for-interchangeable-ai-frameworks/ "https://research.fb.com/facebook-and-microsoft-introduce-new-open-ecosystem-for-interchangeable-ai-frameworks/")
- [Microsoft and Facebook create open ecosystem for AI model interoperability](https://www.microsoft.com/en-us/cognitive-toolkit/blog/2017/09/microsoft-facebook-create-open-ecosystem-ai-model-interoperability/ "https://www.microsoft.com/en-us/cognitive-toolkit/blog/2017/09/microsoft-facebook-create-open-ecosystem-ai-model-interoperability/")
- [ONNX](https://onnx.ai/ "https://onnx.ai/")
- [Core ML](https://developer.apple.com/documentation/coreml "https://developer.apple.com/documentation/coreml")
- [PyTorch](http://pytorch.org/docs/0.3.0/ "http://pytorch.org/docs/0.3.0/")
- [Announcing ONNX Support for Apache MXNet](https://aws.amazon.com/cn/blogs/ai/announcing-onnx-support-for-apache-mxnet/ "https://aws.amazon.com/cn/blogs/ai/announcing-onnx-support-for-apache-mxnet/")