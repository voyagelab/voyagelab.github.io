---
layout: post
title: Core ML vs ML Kit：哪一个移动端机器学习框架更适合你？
date: 2018-08-20 
categories: 机器学习
author: 张永超
--- 

<img src="/img/ML/CoreML_vs_MLKit/cover.jpeg" />

截止2018年举行的Apple全球开发者大会（WWDC2018），Apple公司的用于iOS设备的机器学习框架CoreML走过了一年的更新迭代，迎来了首次较大规模的版本更新。在同一时期，Google也发布了其一款面向iOS和安卓设备的跨平台人工智能开发框架。这两类工具的目的均是为了优化大型人工智能模型和数据集开发的负担，使得开发者能够以轻量化的实现方式，增加移动应用程序的智能化等。但是值得思考的一点是，为什么在这个时期，Google和Apple会相继推出自家的移动端机器学习框架呢？

## 移动端机器学习的重要性

机器学习无疑是一项非常实用的数据科学技术，其应用的程度正在以指数速度在增长，我们几乎每天都会看到一些发展，但是假如普通大众无法获得机器学习带来的有点，无法改善人类的生活方式，那么发展再好的技术也是不会持续下去的。面对这一情况，结合移动设备的空前普及，在移动设备端使用机器学习是最快速让机器学习应用普惠大众的方式。但是机器学习本身是一项复杂而且专业性很高的任务，普通的开发者可能难以快速的理解和应用，为了使在移动设备上进行机器学习的复杂任务变的简单，并且允许没有机器学习经验的应用开发人员实现机器学习的功能，简便而且符合开发人员编程语言习惯的异动单机器学习开发工具势的必出。

## Core ML

<img src="/img/ML/CoreML_vs_MLKit/coreml.png" />

Apple在2017年WWDC上发布了Core ML，并于今年更新为Core ML 2.0。Core ML使开发人员能够将机器学习模型集成到iOS或MacOS应用程序中，这是该领域的第一次重大尝试，最初，开发人员真的很喜欢它，原因有很多。 Core ML针对移动硬件性能进行了优化，可最大限度地减少内存占用和功耗。严格地在设备上运行还可确保用户数据安全，即使没有网络连接，应用程序也会运行。

Core ML最大的有点就是使用起来非常简单，开发人员只需要几行代码就可以集成完整的机器学习模型。自Core ML发布以来，已经有大量的移动应用程序使用了。但是这里要说的是，Core ML并不是万能的，要使用它做什么是有限制的，Core ML智能帮助开发者将训练好的机器学习模型集成到应用程序中，也就意味着在你的应用程序中只能用来进行预测推理，是不可能进行模型的训练学习的。

虽然如此，Core ML也是被证明对开发人员来说，非常有意义的。Core ML 2.0的发布，更是更进了一步，Apple表示Core ML 2.0的速度快了30%，这要归功于批量处理预测机制，而且它可以将模型的大小缩小到75%。

<img src="/img/ML/CoreML_vs_MLKit/coremlsturct.png" />

### Create ML

<img src="/img/ML/CoreML_vs_MLKit/createml-logo.png" />

Apple还在今年的WWDC上还发布了Create ML套件。 Create ML允许开发人员使用Swift和MacOS Playgrounds在Xcode中训练机器学习模型。并且号称没有ML经验的开发人员可以培训模型，而不必依赖其他开发人员。

<img src="/img/ML/CoreML_vs_MLKit/createml.png" />

通过形成一个完整的工具包，Create ML增强了Core ML的实用性。目前，Create ML支持三种数据类型：图像，文本和表格数据。有许多训练和测试算法，如随机森林分类器和支持向量机。创建ML还减少了训练的ML模型的大小，并提供了使用Create ML UI训练模型的方法。

<img src="/img/ML/CoreML_vs_MLKit/demo.png" />

## ML Kit

<img src="/img/ML/CoreML_vs_MLKit/mlkit.png" />

Firebase在Google I/O 2018大会上发布了ML Kit框架。ML Kit使开发人员能够以两种方式在移动应用中使用机器学习：开发人员既可以通过API在云中运行模型推理，也可以在设备上严格运行，就像使用Core ML一样。

ML Kit提供六种基本的API，可供开发人员使用，已提供的模型有：图像标注、文本识别（OCR）、地标检测、人脸检测、条形码扫描和智能回复。如果这些API不包括您的用例，那么您还可以上传TensorFlow Lite模型，ML Kit负责托管并为您的应用提供模型。

<img src="/img/ML/CoreML_vs_MLKit/mlkit-info.png" />

与云版本相比，ML Kit的设备版本提供的精度较低，但同时它为用户数据提供了更高的安全性。 ML Kit提供的基本API涵盖了移动平台上机器学习的所有常规用例，并且使用自定义训练模型的选项使ML Kit成为移动平台的完整机器学习解决方案。开发人员也可以选择将机器学习模型与应用程序分离，并在运行时为它们提供服务，从而减少应用安装规模的大小，确保模型始终保持最新。

<img src="/img/ML/CoreML_vs_MLKit/mlkit-dashboard.png" />


## 比较

Core ML和ML Kit都使开发人员能够在他们的应用程序中利用机器学习的强大功能，从而最终使大众可以使用机器学习的功能。

与Core ML相比，ML Kit具有一些优势。 ML Kit的主要优点是它支持iOS和Android，并且可以在两个平台上使用相同的API。 ML Kit有六个基本API，易于实现，不需要任何ML专业知识。如果您使用ML Kit提供的基本API，那么由于已经存在预训练模型，使用起来更加的方便。

ML Kit的另一个优点是它提供了设备上和基于云的API。 ML Kit中的设备上API可以快速工作，即使在没有互联网连接的情况下也能提供结果。基于云的API利用Google Cloud ML平台提供更高的准确性。 ML Kit的缺点是您可能需要根据使用情况将Firebase计划升级为付费计划。

如果您只对iOS开发感兴趣，那么与Create ML配对的Core ML可能会更有用。 Apple的工具允许您使用更少的代码行在您的应用程序中训练和实现ML模型。使用Create ML的训练模型比使用TensorFlow更容易，但TensorFlow能够提供更高级的训练算法。

Core ML和ML Kit都是很棒的工具，但每个都有局限性。了解您的确切用例以及您将支持的平台可以帮助您确定哪个选项最佳。

# 参考资料

* [Core ML官方文档](https://developer.apple.com/documentation/coreml)
* [ML Kit官方文档](https://developers.google.com/ml-kit/)


