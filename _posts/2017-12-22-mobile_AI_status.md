---
layout: post
title: 移动端AI现状——正在发生的移动平台的 AI 革命
date: 2017-12-22 
categories: 机器学习
author: 王小辉
--- 

很多年以后，当我们回望 2017 年，会意识到对于移动互联网的发展来说，这一年是一个重要的里程碑。芯片制造商、移动操作系统提供商、深度学习框架社区以及移动应用开发者都开始转向 On Device AI，同时，这个趋势同样惠及于 IoT 产业的 Edge 端设备。本文就从这几个方面来解读一下这个趋势。


首先，为什么我们需要 On Device AI 能力呢？在 Edge 端设备上的 AI 能力可以带来这几个好处：

- 降低网络依赖，使得终端设备具备识别和决策的能力；
- 降低服务器端带宽和计算成本，因为可以将计算前置到 Edge 端设备；
- 降低延时，提高 AI 能力响应速度；
- 降低用户的移动流量成本，因为不需要上传大量原始数据到服务器端处理。

那么，Edge 端设备的硬件计算能力毕竟有限，能否支持 AI 模型的 Inference 呢？

### 芯片制造商

- 2017 年 3 月 ARM 提出了面向 AI 的新架构 DynamlQ 技术。

	<img src="/img/ML/mobile_AI_status/DynamIQ.png"  />

“Cortex-A CPUs that are designed based on DynamIQ technology can carry out advanced compute capabilities in Machine Learning and Artificial Intelligence. Over the next three to five years, DynamIQ-based systems will deliver up to a 50x* boost in AI performance. This is achieved through an aggressive roadmap of future DynamIQ IP, integrated with new Arm architectural instructions, microarchitectural improvements, and further software optimizations to the Arm Compute Libraries.”

ARM 表示未来 3 到 5 年内实现比基于 Cortex-A73 的设备高 50 倍的人工智能性能，最多可将 CPU 和 SoC 上特定硬件加速器的反应速度提升 10 倍。
紧接着，ARM 在 4 月份开源了支持 Context-A 系列 CPU 和 Mali 系列 GPU 的 Compute Library，让机器学习和深度学习算法在 ARM 平台更高效地运行。

- 2017 年 8 月高通对外发布了支持 Caffe/Caffe2 和 TensorFlow 的 Neural Processing Engine SDK，让深度学习可以利用 GPU 和 DSP 的计算能力。实际上，在 2017 年 4 月，Facebook 发布了针对 Edge 端设备的深度学习框架 Caffe2，其主要作者贾扬清在 F8 大会上演讲时就提到：“Android 系统上的 GPU 也类似，我们与高通合作开发了‘骁龙神经处理引擎’（SNPE），如今高通骁龙 SoC 芯片为大量的手机服务，现在是 Caffe2 的首要概念（ first class concept），我们可以利用这些 CPU 和 DSP，从本质上提升能效和性能。”

	<img src="/img/ML/mobile_AI_status/Qualcomm.png"  />

高通在 2017 年 12 月发布最新芯片骁龙 845 时提到：“In addition to the existing support for Google’s TensorFlow and Facebook’s Caffe/Caffe2 frameworks, the Snapdragon Neural Processing Engine (NPE) SDK now supports TensorFlow Lite and the new Open Neural Network Exchange (ONNX), making it easy for developers to use their framework of choice, including Caffe2, CNTK and MxNet. Snapdragon 845 also supports Google’s Android NN API.”

也就是说，SNPE 不但支持 Caffe、Caffe2、TensorFlow 和 Edge 设备端专用的 TensorFlow Lite 等深度学习框架，也支持由 Facebook 和微软发起的 ONNX（Open Neural Network Exchange），一个 Intermediate Representation，让 Caffe2、CNTK、PyTorch 以及 MXNet 这些框架可以实现模型互通，方便芯片厂商可以针对一个标准进行硬件级别优化。

- 2017 年 9 月，华为发布了麒麟 970，内置中科院寒武纪-1A NPU，可以加速神经网络在手机端的运行。

	<img src="/img/ML/mobile_AI_status/Huawei_HiAI.png"  />

在发布旗舰设备 Mate 10 之后，华为推出了 HiAI 移动计算平台业务，践行在麒麟 970 发布会上的承诺，打造一个开放式的 AI 生态系统，让开发者可以通过这个平台为华为 Mate 10 这样搭载了最新麒麟芯片的设备上提供具有 AI 能力的应用。

- 2017 年 9 月，Apple 在年度 产品发布会上，发布了将会搭载在 iPhone 8、iPhone 8 Plus 和 iPhone X 上的 A 11 Bionic SoC，用于支持 iOS 系统机器学习和深度学习模型的框架 CoreML 和增强现实框架 ARKit，并且升级了图形处理芯片 Metal 2，用于加速深度学习模型的 Inference。

	<img src="/img/ML/mobile_AI_status/A11.png"  />

- 2017 年 11 月，Google 发布了 Android 8.1 Beta 最终版本，正式支持在今年发布的新款旗舰 Pixel 2 中搭载的 IPU（Image Processing Unit），用于加速图像处理和机器学习的 Inference：“Also, for Pixel 2 users, the Android 8.1 update on these devices enables Pixel Visual Core -- Google's first custom-designed co-processor for image processing and ML -- through a new developer option. Once enabled, apps using Android Camera API can capture HDR+ shots through Pixel Visual Core.”

	<img src="/img/ML/mobile_AI_status/Google_IPU.png"  />

### 移动操作系统提供商

- 2017 年 5 月，Google 在年度 I/O 大会上，宣布会推出 TensorFlow Lite，运行在下一代 Android 系统将会新增的 Neural Network API 之上，使得开发者可以将 TensorFlow 深度学习框架创建的模型移植到移动端运行。10 月 25 日，Google 发布了 Android 8.1 Beta 开发者预览版，正式推出了 NN API：

	<img src="/img/ML/mobile_AI_status/TensorFlow_Lite.png"  />

11 月 14 日，Google 发布的 TensorFlow Lite 已经支持 NN API 了，而 DNNLibrary 是一个 GitHub 上开源的库，支持运行通过 DNN Convert Tool 转换的 Caffe 模型。

- 2017 WWDC 上，Apple 发布了支持 Caffe、TensorFlow 等深度学习框架，以及SVM、XGBoost 、 sklearn 等机器学习模型的 CoreML。12 月 5 日，Google 发布文章 Announcing Core ML support in TensorFlow Lite，宣布已经和 Apple 合作使得 CoreML 支持 TensorFlow Lite。

	<img src="/img/ML/mobile_AI_status/Core_ML.png"  />

### 深度学习框架

- 2017 年 4 月，Facebook 宣布开源面向 Edge 端设备的深度学习框架 Caffe2，其主要作者 贾扬清在 F8 上说：“在移动端进行机器学习任务有以下好处：保护隐私，因为数据不会离开你的移动端；避免网络延迟和带宽问题；提升用户体验，比如优化 feed ranking 功能。所以我们开始着手，从底层建造一个专门为移动端优化的机器学习框架。”

	<img src="/img/ML/mobile_AI_status/Caffe2.png"  />

- 2017 年 9 月，百度发布了 Mobile Deep Learning library，腾讯早些时候也开源了类似的 ncnn 框架。

- 2017 年以来，Google 开源了专为移动端优化的 MobileNets 模型；Face++ 提出了适合移动端的 ShuffleNet 模型。OpenCV 3.3 内置了支持多种深度学习框架的 DNN。


### 应用开发者

从上面的内容可以看到，从芯片制造商到移动操作系统提供商，再到深度学习框架社区，都为 On Device AI 做了很多准备，而且，从 GitHub 和国内外的开发者博客上，我们看到了非常多基于 CoreML、基于 TensorFlow 等深度学习框架的移动端应用案例，体现了开发者对这个趋势的极大热情。无论我们是哪个移动平台的开发者，都应该清晰认识到这个趋势，及时点亮自己的技能点，为用户提供更智能、更人性化的应用。