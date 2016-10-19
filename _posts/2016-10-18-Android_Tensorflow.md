---
layout: post
title: Tensorflow 在 Android 平台的移植
date: 2016-10-18 
tags: 技术   
author: 小辉  
---


### TensorFlow 简介

2015 年 11 月 9 日，Google Research 发布了文章：[TensorFlow - Google’s latest machine learning system, open sourced for everyone](https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html)，正式宣布其新一代机器学习系统开源。

至于 Google 为什么要开源 TensorFlow，官方的说法是：

> If TensorFlow is so great, why open source it rather than keep it proprietary? The answer is simpler than you might think: We believe that machine learning is a key ingredient to the innovative products and technologies of the future. Research in this area is global and growing fast, but lacks standard tools. By sharing what we believe to be one of the best machine learning toolboxes in the world, we hope to create an open standard for exchanging research ideas and putting machine learning in products. Google engineers really do use TensorFlow in user-facing products and services, and our research group intends to share TensorFlow implementations along side many of our research publications.

[Here's Why Google Is Open-Sourcing Some Of Its Most Important Technology](http://www.forbes.com/sites/gregsatell/2016/07/18/heres-why-google-is-open-sourcing-some-of-its-most-important-technology/#373b1a53630c) 文章中援引了 TensorFlow 开发者的说法：

> The decision to open-source was the brainchild of Jeff Dean, who felt that the company’s innovation efforts were being hampered by the slow pace of normal science. Google researchers would write a paper, which would then be discussed at a conference some months later. Months after that somebody else would write another paper building on their work.  

> Dean saw that open-sourcing TensorFlow could significantly accelerate the process. Rather than having to wait for the next paper or conference, Google’s researchers could actively collaborate with the scientific community in real-time. Smart people outside of Google could also improve the source code and, by sharing machine learning techniques more broadly, it would help populate the field with more technical talent.

> “Having this system open sourced we’re able to collaborate with many other researchers at universities and startups, which gives us new ideas about how we can advance our technology. Since we made the decision to open-source, the code runs faster, it can do more things and it’s more flexible and convenient,” says Rajat Monga, who leads the TensorFlow team.

毫无意外地，TensorFlow 在 Github 上的 Repo 在很短的时间内就收获了大量的 `Star` 和 `Fork`，学术界和工业界都对其表示了巨大的兴趣，并投身于 TensorFlow 的社区和 Google 一起完善和改进 TensorFlow。

然而，当时在 Github 做基准测试、目前就职于 Facebook AI 部门的程序员 [Soumith](https://github.com/soumith) 发布了文章 [Benchmark TensorFlow](https://github.com/soumith/convnet-benchmarks/issues/66)（[中文解读](http://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=400451107&idx=1&sn=ddfd58a09319c1816c394d042a17c7f4&scene=0#wechat_redirect)），对 TensorFlow 和其他主流深度学习框架的性能进行了比较，结果差强人意。当然，Google 团队表示会继续优化，并在后面的版本中支持分布式。

2016 年 4 月 13 日，Google 通过文章 [Announcing TensorFlow 0.8 – now with distributed computing support!](https://research.googleblog.com/2016/04/announcing-tensorflow-08-now-with.html) 正式发布支持分布式的 TensorFlow 0.8 版本，结合之前对 CPU 和 GPU 的支持，TensorFlow 终于可以被用于实际的大数据生产环境中了。

2016 年 4 月 29 日，开发出目前最强围棋 AI 的 Google 旗下 DeepMind 宣布：[DeepMind moves to TensorFlow](https://research.googleblog.com/2016/04/deepmind-moves-to-tensorflow.html)，这在业界被认为 TensorFlow 终于可以被当作 TensorFlow 在工业界发展的里程碑事件，极大提升了 TensorFlow 使用者的研究热情。

[The Good, Bad & Ugly of TensorFlow](http://www.kdnuggets.com/2016/05/good-bad-ugly-tensorflow.html)（[中文翻译](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650715438&idx=2&sn=3dafb301ec8103fce7ad88d6039cb3ad)）对目前 TensorFlow 的优缺点做了详细的分析。

### Windows 平台

Tensorflow [官方文档](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) 中，对 Android Demo 的编译介绍时用到了 [bazel](https://www.bazel.io/versions/master/docs/install.html#ubuntu)，该工具对 Windows 平台的支持处于实验阶段，就不推荐了，Github 上有一个 [使用 NDK 在 Anroid Studio 中进行编译的示例工程](https://github.com/miyosuda/TensorFlowAndroidDemo)，大家可以 clone 下来使用。

### Ubuntu 14.04

这里假定 Ubuntu 14.04 系统上还没有 Android 开发环境。

#### 安装 Java 1.8

    $ sudo apt-get install software-properties-common
    $ sudo add-apt-repository ppa:webupd8team/java
    $ sudo apt-get update
    $ sudo apt-get install oracle-java8-installer

#### 配置 Java 环境变量，将下面的内容添加到 `/etc/environment`:

    JAVA_HOME="/usr/lib/jvm/java-8-oracle"

#### 安装 bazel

    $ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    $ curl https://bazel.io/bazel-release.pub.gpg | sudo apt-key add -
    $ sudo apt-get update && sudo apt-get install bazel
    $ sudo apt-get upgrade bazel

详细的说明可以参考 [bazel 的官方文档](https://www.bazel.io/versions/master/docs/install.html#ubuntu)。

#### 下载 tensorflow

    $ cd ~/
    $ git clone https://github.com/tensorflow/tensorflow.git


之后的步骤基本来自 [TensorFlow on Android](https://www.oreilly.com/learning/tensorflow-on-android) 的翻译：

#### 下载解压 Android SDK

    $ wget https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz
    $ tar xvzf android-sdk_r24.4.1-linux.tgz -C ~/tensorflow

更新 SDK：

    $ cd ~/tensorflow/android-sdk-linux
    # 如果希望在熟悉的 SDK Manager 中进行操作，可以去掉下面命令中的 --no-ui
    $ tools/android update sdk --no-ui

#### 下载解压 NDK

    $ wget https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip
    $ unzip android-ndk-r12b-linux-x86_64.zip -d ~/tensorflow

#### 下载 tensorflow 的 model

    $ cd ~/tensorflow
    $ wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -O /tmp/inception5h.zip
    $ unzip /tmp/inception5h.zip -d tensorflow/examples/android/assets/

#### 修改 WORKSPACE

    $ gedit ~/tensorflow/WORKSPACE

反注释 `android_sdk_repository` 和 `android_ndk_repository` 部分，用下面的内容替换：

    android_sdk_repository(
        name = "androidsdk",
        api_level = 24,
        build_tools_version = "24.0.3",
        # Replace with path to Android SDK on your system
        path = "/home/ross/Downloads/android-sdk-linux",
    )
    
    android_ndk_repository(
        name="androidndk",
        path="/home/ross/Downloads/android-ndk-r12b",
        api_level=24)

#### 编译 tensorflow 的 Android Demo App：

    $ cd ~/tensorflow
    $ bazel build //tensorflow/examples/android:tensorflow_demo

如果一切顺利就会在最后看到下面的提示：

    bazel-bin/tensorflow/examples/android/tensorflow_demo_deploy.jar
    bazel-bin/tensorflow/examples/android/tensorflow_demo_unsigned.apk
    bazel-bin/tensorflow/examples/android/tensorflow_demo.apk
    INFO: Elapsed time: 109.114s, Critical Path: 37.45s

### Android Demo 分析

整个 Demo 的目录结构和使用 Jni 的 Android 工程是相同的，在 `~/tensorflow/tensorflow/examples/android/jni` 目录下，放着 native 的代码：

    ├── imageutils_jni.cc
    ├── __init__.py
    ├── rgb2yuv.cc
    ├── rgb2yuv.h
    ├── yuv2rgb.cc
    └── yuv2rgb.h

Java interface 相关的 Java 类在 `https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android` 目录里面，可以考虑将其直接集成到自己的项目中。

Demo 所需的 native 实现在 `https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/jni` 目录里面。

`https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageListener.java` 里面定义了用到的 tensorflow model，protobuf 格式，识别结果的 labels 等：

        private static final Logger LOGGER = new Logger();
    
        private static final boolean SAVE_PREVIEW_BITMAP = false;
    
        private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
        private static final String LABEL_FILE =
                "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    
        private static final int NUM_CLASSES = 1001;
        private static final int INPUT_SIZE = 224;
        private static final int IMAGE_MEAN = 117;

如果想使用自己的模型，使用 tensorflow 解决其他的问题，通过修改上面提到的代码和模块来完成。[TensorFlow on Android](https://www.oreilly.com/learning/tensorflow-on-android) 文章就提到了具体的步骤。

最后，Tensorflow 也支持移植到 iOS 应用中，可以参考 TalkingData SDK Team 的技术博客文章 [iOS 开发迎来机器学习的春天--- TensorFlow](http://talkingdata.me/2016/07/07/iOS_TensorFlow/)。