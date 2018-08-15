---
layout: post
title: 移动端机器学习框架 MDL 简介与实践
date: 2017-12-25 
categories: 机器学习
author: 刘晓飞
--- 

### Mobile-deep-learning（MDL）

MDL 是百度研发的可以部署在移动端的基于卷积神经网络实现的移动端框架，可以应用在图像识别领域。

具体应用：在手机百度 App 中，用户只需要点击自动拍开关，将手机对准物体，当手停稳的时候，它会自动找到物体并进行框选，无需拍照就可以发起图像的搜索功能。


<img src="/img/ML/mobile_ML_framework_MDL/pen.png" />

## 初识 MDL

### 运行示例程序

1. clone 项目代码,  [https://github.com/baidu/mobile-deep-learning](https://github.com/baidu/mobile-deep-learning "https://github.com/baidu/mobile-deep-learning")

	<img src="/img/ML/mobile_ML_framework_MDL/project_directory.png" />
	
2. 在 IDE 中导入 example

	<img src="/img/ML/mobile_ML_framework_MDL/import_project.png" />

3. 运行

	<img src="/img/ML/mobile_ML_framework_MDL/mouse.png" />

### 开发要求

1. 安装 NDK
2. 安装 Cmake
3. 安装 protocol buffers 

### 使用 MDL 库

1. 在 mac/linux 上执行测试

		# mac or linux:
		./build.sh mac
		cd build/release/x86/build
		./mdlTest

2. 在项目中使用mdl

### 开发

1. 编译MDL源码（Android）

		# android:
		# prerequisite: install ndk from google

		./build.sh android
		cd build/release/armv-v7a/build
		./deploy_android.sh
		adb shell
		cd /data/local/tmp
		./mdlTest

2. iOS

		# ios:
		# prerequisite: install xcode from apple

		./build.sh ios
		copy ./build/release/ios/build/libmdl-static.a to your iOS project

### 模型转换

MDL 需要与之兼容的模型才能使用，可以使用 MDL 提供的脚本将其他深度学习工具训练的模型转换为 MDL 模型。推荐使用 PaddlePaddle 模型。

1. 将 PaddlePaddle 模型转换成 MDL 模式

		# Environmental requirements
		# paddlepaddle
		cd tools/python
		python paddle2mdl.py

2.将 caffemodel 模型转换成 MDL 模式

	#Convert model.prototxt and model.caffemodel to model.min.json and data.min.bin that mdl use

	./build.sh mac
	cd ./build/release/x86/tools/build

	# copy your model.prototxt and model.caffemodel to this path

	./caffe2mdl model.prototxt model.caffemodel

	# the third para is optional, if you want to test the model produced by this script, provide color value array of an image as the third parameter ,like this:

	./caffe2mdl model.prototxt model.caffemodel data

	# the color value should in order of rgb,and transformed according to the model.

	# then you will get a new data.min.bin with test data inside

	# after this command, model.min.json data.min.bin will be created in current
	# some difference step you need to do if you convert caffe model to iOS GPU format
	# see this:
	open iOS/convert/iOSConvertREADME.md

### Android Sample 分析

下面以 Android 平台为例分析 MDL 在移动端平台上面的工作

1. 在项目中导入 libmdl.so 库
2. 初始化 mdl，加载 so 库，设置线程数量

		private void initMDL() {
			String assetPath = "mdl_demo";
			String sdcardPath = Environment.getExternalStorageDirectory() + File.separator + assetPath;
			copyFilesFromAssets(this, assetPath, sdcardPath);
			mdlSolver = new MDL();
			try {
				String jsonPath = sdcardPath + File.separator + type.name() + File.separator + "model.min.json";
				String weightsPath = sdcardPath + File.separator + type.name() + File.separator + "data.min.bin";
				Log.d("mdl","mdl load "+ jsonPath + "weightpath ="+weightsPath);
				mdlSolver.load(jsonPath, weightsPath);
				if (type == mobilenet) {
					mdlSolver.setThreadNum(1);
				} else {
					mdlSolver.setThreadNum(3);
				}

			} catch (MDLException e) {
				e.printStackTrace();
			}
		}

3. 拍摄照片

		Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
		// save pic in sdcard 
		Uri imageUri = Uri.fromFile(getTempImage());
		intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
		startActivityForResult(intent, TAKE_PHOTO_REQUEST_CODE);

4. 图像处理，包括设置采样率、图像缩放
5. 图像预测

		mdlSolver.predictImage(inputData);

6. 在MDL库中，predictImage 方法是进行图像预测的 JNI 方法。方法声明如下：

		JNIEXPORT jfloatArray JNICALL Java_com_baidu_mdl_demo_MDL_predictImage(JNIEnv *env, jclass thiz, jfloatArray buf)

	这里需要传入图像的3维数组结构，真正执行预测的是 Net#predict(data) 方法，Net 模块是 MDL 网络管理模块，主要负责网络中各层 Layer 的初始化及管理工作。开发者在调用预测方法的时候，只需要调用对应 java 的 predictImage 方法，传入图像数据即可。

7. 预测完成，在 demo 的界面中返回预测耗时和结果。

### 性能和兼容性

<img src="/img/ML/mobile_ML_framework_MDL/comparation.png" />

### 总结

MDL 在 Android 和 iOS 系统上性能表现十分出色，并且 API 设计也很简单易用，也支持其他的框架模型转换。总体来讲是一个非常优秀的移动端深度学习框架。

#### 参考：

[https://github.com/baidu/mobile-deep-learning](https://github.com/baidu/mobile-deep-learning "https://github.com/baidu/mobile-deep-learning")