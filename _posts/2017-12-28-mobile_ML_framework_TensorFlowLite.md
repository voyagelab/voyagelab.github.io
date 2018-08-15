---
layout: post
title: 移动端机器学习框架TensorFlow Lite简介与实践
date: 2017-12-28
categories: 机器学习
author: 张自玉
--- 

### TensorFlow Lite简介

TensorFlowLite是Google在2017年5月推出的轻量级机器学习解决方案，主要针对移动端设备和嵌入式设备。针对移动端设备特点，TensorFlow Lite使用了诸多技术对内核进行了定制优化，预熔激活，量子化内核。TensorFlow Lite具有以下特征：

1. 跨平台，核心代码由C++编写，可以运行在不同平台上；
2. 轻量级，代码体积小，模型文件小，运行内存低；
3. 支持硬件加速。


### TensorFlow Lite架构


<img src="/img/ML/mobile_ML_framework_TensorFlowLite/TensorFlow_Lite_Architecture.png" />

在server端完成模型的训练。通过TensorFlow Lite Converter转化成可以在手机端运行的.tflite格式的模型。TensorFlow Lite可以接入Android底层的神经网络的API从而进行硬件加速。另外TensorFlow团队宣布，已经与Apple达成合作，可以将TensorFlow平台训练出来的模型转化成可被Apple深度学习框架CoreML所解释运行的格式.mlmodel。这样一来开发者就有了更多的选择。他们既可以选择直接使用Google原生的tflite模型，也可以通过转化工具对接Apple的CoreML框架，更高效发挥Apple的硬件性能。 CoreML的转化工具可以从https://github.com/tf-coreml/tf-corem去下载。


### TensorFlow Lite Demo

TensorFlow Lite提供了iOS和Android两个平台的Demo App。 这里以iOS平台为例，按照TensorFlow Lite提供的Tutorial完成配置。

首先安装xcode的工具链：

	xcode-select --install

安装automake和libtool：

	brew install automake
	brew install libtool

运行脚本。脚本的作用是下载相关依赖。主要下载TensorFlowLite模型：

	tensorflow/contrib/lite/download_dependencies.sh

运行编译脚本，编译TensorFlowLite通用库：

	tensorflow/contrib/lite/build_ios_universal_lib.sh

文件说明：

|文件|类型|大小|
|--|--|--|
|mobilenet_quant_v1_224.tflite|模型文件|4.3M|
|libtensorflow-lite.a|静态库(包含多种架构 所以体积较大)|28.7M|
|Demo.ipa|DemoApp的安装包|9.4M|

Demo截图：

<figure class="third">
	<img src="/img/ML/mobile_ML_framework_TensorFlowLite/demo1.png" width="200" />
	<img src="/img/ML/mobile_ML_framework_TensorFlowLite/demo2.png" width="200" />
	<img src="/img/ML/mobile_ML_framework_TensorFlowLite/demo3.png" width="200" />
</figure>


Demo里提供了一个由MobileNets训练出来的模型。Android端的代码可以从https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk下载。iOS端的Demo需要自己编译。

Demo会把相机捕获到的每一帧转化成一个224×224像素的图像，这是因为训练Mobilenet的模型的输入都是224×224像素的。处理后的图像会被进一步的转化成一个1×224×224×3的ByteBuffer。1代表数量是1，224×224像素的数量，3代表每一个像素点有3个通道。Demo接受单输入，输出是一个二维数组，第一维是分类索引，第二维是索引的可信程度（概率）。如上图所示。

由上表可以得知，模型文件大小为4.3M，静态库的大小为28.7M，这是因为静态库是包含了所有架构的通用包，所以体积稍大。而最后打出来的安装包体积仅仅有9.4M。在iPhone 6上执行实时推断。摄像头捕获帧的时间间隔大约为4ms，说明实时推断的执行时间小于4ms，性能非常好。


### TensorFlow Lite的模型

目前TensorFlowLite在移动端只能解释执行.tflite格式的模型文件。但是一般模型训练都是在server端完成的。所以，想要把server端训练好的模型转化成可以在移动端运行的tflite格式，就需要对训练中的各种文件格式有一定了解。

- GraphDef (.pb)：一组代表Tensorflow训练或者计算图的protobuf数据。数据里包含了运算符、张量和变量定义。
- CheckPoint (.ckpt)：从TensorFlow图中序列化出来的变量。checkpoint里是不包含图的结构的。单独的checkpoint无法被解释执行。
- FrozenGraphDef(固化权重图) ：GraphDef的子类。可以通过把checkpoint里的数值带入GraphDef中得到。
- SavedModel：保存的模型。是GraphDef和CheckPoint的集合。
- TensorFlow lite model (.tflite)：一组被序列化的flatbuffer，其中包含了为例TensorFlow lite操作符和可以被TensorflowLite 解释器解释的张量。跟固化权重图很相似。

其中.pb文件和.ckpt文件都可以在server端生成。但是如果想server端训练出来的模型可以被移动端所用，就需要将二者转化成.tflite格式的模型。

用bazel可以轻易的完成转化工作。

	bazel build tensorflow/contrib/lite/toco:toco

	bazel-bin/tensorflow/contrib/lite/toco/toco -- \
  		--input_file=$(pwd)/mobilenet_v1_1.0_224/frozen_graph.pb \
  		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
  		--output_file=/tmp/mobilenet_v1_1.0_224.lite --inference_type=FLOAT \
  		--input_type=FLOAT --input_arrays=input \
  		--output_arrays=MobilenetV1/Predictions/Reshape_1 --input_shapes=1,224,224,3

#### 预训模型(pre-trained model)

TensorFlow官方提供了一些专门用于TensorFlowLite的模型。模型可以从`https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md`下载。模型由以下训练算法得出：

|模型|类别|特性|
|--|--|--|
|MobileNets|计算机视觉 图像识别|低延迟，低功耗，体积轻巧|
|Inception-v3|计算机视觉 图像识别|高精度|
|On Device Smart Reply|文本处理|快速回复，目前已经在Android Wear中有应用|

#### 迁移学习（ transfer learning ）

一般来讲，计算机视觉类的模型训练通常耗时数日。但是迁移学习会把这个时间缩短到只有几个小时甚至更短。比如现在训练了一个识别猫的模型，现在需求变更为识别狗。利用迁移学习只需要把狗的训练图片作为输入，输入到已经训练好的猫类识别器里。模型会在很短的时间内完成迁移学习。

#### 迁移学习的模型选择和首选精度

如上表格所示，Inception-v3模型的首选精度为78%，但是训练出的模型大小有85M。同样的数据集在MobileNet上，模型大小只有19M。下图展示了模型和首选精度的关系。

<img src="/img/ML/mobile_ML_framework_TensorFlowLite/model_and_preferred_precision.png" />

- Y轴代表的是首选精度
- X轴代表的是对应精度所代表的计算次数
- 圆形面积大小代表的是完成训练后模型的体积
- 颜色作为不同模型的区分，特别之处是这里的紫色GoogleNet其实就是Inception-v3

由此得出，想要在移动端获取实时快速响应，并且缩小模型体积的话，就需要牺牲一定的精度，来获取更好的性能。因此MobileNet是移动端的首选。

### 从TensorFlow Lite看AI-on-Device

从芯片到框架，各大软硬件厂商都在积极地布局移动端AI。

从AI应用的现状来看，模型训练这些工作大多是在云端完成的，因为云端有着强大的GPU/CPU和内存。各种机器学习框架会充分的利用分布式、硬件加速等技术快速的完成模型的训练。 从TensorFlow Lite可以看出Google在移动端AI的野心。目前TensorFlow Lite已经具备了为移动终端提供智能服务的能力。云端训练完成的模型在移动终端上可以被快速的解释执行。但是目前移动端AI也面临一些挑战，如：

- 移动端单机无法进行分布式训练；
- 移动端设备内存有限，部署模型受到内存限制。

但是移动端的AI也有很多优势：

- 移动端是数据采集的第一现场，移动端集成了多种传感器，所以传感器产生的各种数据可以第一时间被设备获取到，节省传输时间；
- 图像、语音、文本等数据大多是由手机的使用者产生。如果能在device上部署训练，数据的隐私会得到更好的保护；
- 移动端产生的都是个性化数据，例如行为数据，对这些数据进行“过拟合”的训练，可以产生个性化的定制服务。

正如TensorFlowLite官网所说的：

“As we continue development, we hope that TensorFlow Lite will greatly simplify the developer experience of targeting a model for small devices.”

我们也希望这天早点到来。