---
layout: post
title: 移动端机器学习框架SNPE简介与实践
date: 2017-12-27
categories: 机器学习
author: 刘晓飞
--- 

### NPE SDK能够帮助开发者做什么事情？

Qualcomm骁龙神经处理引擎（Neural Processing Engine, NPE）SDK 能够帮助有意创建人工智能（AI）解决方案的开发者，在骁龙移动平台上（无论是CPU、GPU还是DSP）运行通过Caffe/Caffe2或TensorFlow训练一个或多个神经网络模型，且无需连接到云端，实现边缘计算。

NPE SDK能帮助开发者在骁龙设备上运行受过训练的神经网络并优化其性能。NPE SDK提供了模型转换和执行工具，以及针对核的API，利用功率和性能配置文件匹配所需的用户体验，优化和节约开发者的时间和精力。

NPE SDK支持卷积神经网络、长短期记忆网络（LSTM）和定制层，处理在骁龙移动平台上运行神经网络所需的大量繁重工作，为开发人员留出更多的时间和资源来专注于AI的创新应用体验。

<img src="/img/ML/mobile_ML_framework_SNPE/Qualcomm.png" />

### SDK主要特性有哪些？

- 支持Android和Linux运行环境，供执行神经网络模型
- 支持利用Qualcomm Hexagon DSP、Qualcomm Adreno GPU和 Qualcomm Kryo、CPU（NPE SDK支持Qualcomm Snapdragon 820、835、625、626、650、652、653、660、630、636和450）设备必须有libOpenCL.so，以支持Qualcomm Adreno GPU），为应用提供加速
- 支持Caffe、Caffe2和TensorFlow模型
- 提供控制运行时加载、执行和调度的多个API
- 用于模型转换的桌面工具
- 用于识别性能瓶颈的性能基准测试
- 示例代码和教程
- HTML文档

### NPE SDK适合哪些开发者？

使用骁龙NPE SDK开发AI需要满足以下几个前提，然后才可以开始创建解决方案。

- 你在一个或多个垂直领域需要运行卷积/LSTM模型，包括移动、汽车、IoT、AI、AR、无人机和机器人等
- 你了解如何设计和训练模型，或者已经有一个预训练的模型文件
- 你选择的框架是Caffe/Caffe2或TensorFlow
- 你可以使用Android编写Java应用，或者基于Android或Linux系统编写原生应用
- 你有Ubuntu 14.04开发环境
- 你有可用于测试应用程序的设备

### NPE SDK使用开发流程

为了让AI开发者更轻松，骁龙NPE SDK没有另行定义网络层库；发布时就支持Caffe/Caffe2和TensorFlow，开发人员可以选择使用自己熟悉的框架设计和训练网络。开发工作流程如下：

<img src="/img/ML/mobile_ML_framework_SNPE/NPE_workflow.png" />

完成模型的设计和训练后，模型文件需要转换成“.dlc”（Deep Learning Container）文件，供骁龙 NPE运行时使用。转换工具将输出转换信息，包括有关不受支持或非加速层的信息，开发者可以使用这些信息调整初始模型的设计。

### 搭建NPE SDK工作环境

#### 系统环境搭建

建议在专门机器上执行以下操作，以便更好地了解SDK依赖项：

1. 安装Ubuntu 14.04（官网推荐使用）
如果使用虚拟机安装，可以使用VirtualBox工具。虚拟机磁盘空间需要分配大些，建议分配30G，后续Android Studio需要比较大的磁盘空间。

2. 安装最新版Android Studio，地址：https://developer.android.google.cn/studio/index.html
通过Android Studio或独立安装最新版Android SDK。

3. 安装最新版Android NDK
通过Android Studio SDK Manager或独立安装。

4. 安装Caffe，GitHub：https://github.com/BVLC/caffe
安装说明：http://caffe.berkeleyvision.org/installation.html

		# this will build Caffe (and the pycaffe bindings) from source - see the official instructions for more information 

		sudo apt-get install cmake git libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libprotobuf-dev libsnappy-dev protobuf- compiler python-dev python-numpy
		
		git clone https://github.com/BVLC/caffe.git ~/caffe
		cd ~/caffe
		git reset --hard d8f79537
		
		mkdir build
		cd build
		cmake ..
		make all -j4
		make install

5. 安装TensorFlow（推荐版本1.0，GitHub：`https://github.com/tensorflow/tensorflow`）（可选）
	
	安装说明：https://www.tensorflow.org/install/

		# this will download and install TensorFlow in a virtual environment - see the official instructions for more information

		sudo apt-get install python-pip python-dev python-virtualenv 
		mkdir ~/tensorflow
		virtualenv -- system-site-packages ~/tensorflow
		source ~/tensorflow/bin/activate 

		pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none- linux_x86_64.whl

#### 安装NPE SDK

本步骤允许NPE SDK通过python API与Caffe和TensorFlow框架进行通信。在Ubuntu 14.04上安装SDK，请执行以下操作：

1. 下载最新的骁龙NPE SDK。 

	地址：https://developer.qualcomm.com/software/snapdragon-neural-processing-engine

	将.zip文件解压至适当位置（假定在~/snpe-sdk文件夹中）。

2. 安装缺少的系统包：

		# install a few more SDK dependencies, then perform a comprehensive check 

		sudo apt-get install python-dev python-matplotlib python-numpy python-protobuf python-scipy python-skimage python- sphinx wget zip 

		# verifies that all dependencies are installed
		source ~/snpe- sdk/bin/dependencies.sh 
		
		# verifies that the python dependencies are installed
		source ~/snpe- sdk/bin/check_python_depends.sh 

3. 在当前控制台窗口初始化Snapdragon NPE SDK环境。以后，每个新控制台需重复此操作：

		# initialize the environment on the current console 
		cd ~/snpe-sdk/ 
		# default location for Android Studio, replace with yours 
		export ANDROID_NDK_ROOT=~/Android/Sdk/ndk-bundle 
		source ./bin/envsetup.sh -c ~/caffe 
		# optional for this guide
		source ./bin/envsetup.sh -t ~/tensorflow 
		
	初始化过程将设置或更新 $SNPE_ROOT,?$PATH, $LD_LIBRARY_PATH, $PYTHONPATH, $CAFFE_HOME, $TENSORFLOW_HOME，此外，还在本地复制 Android NDK libgnustl_shared.so 库，更新 Android AAR 存档文件。

### 下载ML Models并转换为.DLC

NPE SDK没有绑定公开的模型文件，但包含一些脚本，可用于下载一些主流模型，并将其转换为Deep Learning Container（DLC）格式。脚本位于/models文件夹中，文件夹中还包含DLC模型。

1. 下载并转换经预先训练的Alexnet示例（Caffe格式）：

		cd $SNPE_ROOT 
		python ./models/alexnet/scripts/setup_alexnet.py -a ./temp-assets-cache -d

	提示：查看执行DLC转换的setup_alexnet.py脚本。您可能需要针对Caffe模型转换执行相同的操作。

2. 可选：下载并转换经预先训练的“inception_v3”示例（TensorFlow格式）：

		cd $SNPE_ROOT
		python ./models/inception_v3/scripts/setup_inceptionv3.py -a ./temp-assets-cache - d

3. 提示：查看setup_inceptionv3.py脚本，此脚本还对模型进行了量化，大小缩减了75％（91MB→23MB）。

### 构建示例Android APP

示例Android App的源代码演示了如何正确使用SDK。可以从ClassifyImageTask.java开始。示例Android App结合了Snapdragon NPE运行环境（/android/snpe-release.aar Android库提供）和上述Caffe Alexnet示例生成的DLC模型。

1. 复制运行环境和模型，为构建App作好准备

		cd $SNPE_ROOT/examples/android/image-classifiers

		# copies the NPE runtime library
		cp ../../../android/snpe- release.aar ./app/libs  

		# packages the Alexnet example (DLC, labels, imputs) as an Android resource file
		bash ./setup_models.sh 

	可选方法一：从Android studio构建Android APK：

	1. 启动Android Studio。
	2. 打开~/snpe-sdk/examples/android/image- classifiers文件夹中的项目。
	3. 如有的话，接受Android Studio建议，升级 构建系统组件。
	4. 按下“运行应用”按钮，构建并运行APK。

	可选方法二：从命令行构建Android APK：
	
		# Android SDK build dependencies on ubuntu 
		sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1 libbz2-1.0:i386 

		# build the APK
		./gradlew assembleDebug 

	上述命令需要将ANDROID_HOME和JAVA_HOME 设置为系统中的Android SDK和JRE/JDK所在位置。

	执行到这里完成后，示例App已经build完成，安装后如下所示：

	<img src="/img/ML/mobile_ML_framework_SNPE/app_demo.png" />

	使用Snapdragon NPE SDK制作了第一款示例应用。那么现在，可以开始创建属于自己的AI解决方案了！ SDK      随附文档中还有 API 文档、教程和架构详细资料。可以在浏览器中打开 `/doc/html/index.html`开始学习。

### 总结

NPE SDK 目前做的还不是非常完美，有些需要定制化的神经网络层可能在原生NPE中没有提供。但还好SDK提供了用户定义层（UDL）功能，通过回调函数可以自定义算子，并通过重编译C++代码将自定义文件编译到可执行文件中。如果开发就是使用的C++，那比较容易实现用户定义层，但如果是运行在Android上，开发者需要将上层Java代码通过JNI方式来调用NPE原生的C++编译好的.so文件，因为用户定义层的代码是不可能预先编译到NPE原生.so文件中的，必须重新开发NPE的JNI。