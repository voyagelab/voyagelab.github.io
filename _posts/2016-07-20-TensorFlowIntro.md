---
layout: post
title: " TensorFlow 简介"
date: 2016-07-20 
tags: 机器学习   
author: 小辉  
---


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

### TensorFlow 学习资源

TensorFlow 使用 Python 作为主要接口语言，所以掌握 Python 在 Data Science 领域的知识就成为学习 TensorFlow 的必要条件。[A Complete Tutorial to Learn Data Science with Python from Scratch](http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) 就是一篇非常好的学习资料。

- [TensorFlow 官方文档](https://www.tensorflow.org/versions/r0.9/get_started/basic_usage.html) 从首个版本就非常详细。
- [LearningTensorFlow: A beginners guide to a powerful framework.](http://learningtensorflow.com/index.html)，包含详细的接口定义，各种学习资源和例子。
- [Hello, TensorFlow!](https://www.oreilly.com/learning/hello-tensorflow) Building and training your first TensorFlow graph from the ground up.
- [A noob’s guide to implementing RNN-LSTM using Tensorflow](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/)
- [Updated with Google’s TensorFlow: Artificial Intelligence, Neural Networks, and Deep Learning](https://kimschmidtsbrain.com/2015/10/29/artificial-intelligence-neural-networks-and-deep-learning/) 强烈推荐这篇文章，对AI、NN、DL 的发展历史以及其中的关键大牛的关键工作做了详细介绍。
- [DeepDreaming with TensorFlow](http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb) This notebook demonstrates a number of Convolutional Neural Network image generation techniques implemented with TensorFlow for fun and science
- [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) TensorFlow Tutorial with popular machine learning algorithms implementation. This tutorial was designed for easily diving into TensorFlow, through examples.
- [TensorFlow-Tutorials](https://github.com/nlintz/TensorFlow-Tutorials) Introduction to deep learning based on Google's TensorFlow framework. These tutorials are direct ports of Newmu's Theano Tutorials.
- [Dive Into TensorFlow, Part II: Basic Concepts](http://textminingonline.com/dive-into-tensorflow-part-ii-basic-concepts) TensorFlow 中基本概念的解释
- [TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html) 系列学习笔记，中文版
- [TensorFlow人工智能引擎入门教程所有目录](http://my.oschina.net/yilian/blog/664632#OSC_h2_1) 非常多作者学习和使用 TensorFlow 的经验文章。

深度学习不是一个突然出现的概念，而是从神经网络发展而来的，所以，学习 TensorFlow，对深度学习领域本身的发展历史有基本的了解有助于理解技术的发展。这方面有很多非常好的文章：

- [[Machine Learning & Algorithm] 神经网络基础](http://www.cnblogs.com/maybe2030/p/5597716.html) 
- [Deep Learning in Neural Networks: An Overview](http://people.idsia.ch/~juergen/DeepLearning8Oct2014.pdf)
- [Deep learning](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) by Yann LeCun, Yoshua Bengio& Geoffrey Hinton
- [A 'Brief' History of Neural Nets and Deep Learning](http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/) 一个系列，图文并茂，非常详细。
- [A Gentle Guide to Machine Learning](https://blog.monkeylearn.com/a-gentle-guide-to-machine-learning/) 条理非常清晰。
- [A Neural Network in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/) 非常好的从头开始实现一个神经网络的文章，对学习和理解神经网络中所用到的技术很有用。
- [Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks](https://iamtrask.github.io/2015/07/12/basic-python-network/) 系列文章，非常详细。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
- [Welcome to the Deep Learning Tutorial!](http://ufldl.stanford.edu/tutorial/) 斯坦福深度学习资料
- [Learning How To Code Neural Networks](https://medium.com/learning-new-stuff/how-to-learn-neural-networks-758b78f2736e#.xri3grne7)
- [Machine Learning in a Week](https://medium.com/learning-new-stuff/machine-learning-in-a-week-a0da25d59850#.3rx8phs9d) Deep Learning 的学习计划。
- [Conv-Nets-And-Gen](https://github.com/xhrwang/Conv-Nets-Series) TensorFlow 官方推荐的文章。
- [Convolutional Neural Networks backpropagation: from intuition to derivation](https://grzegorzgwardys.wordpress.com/2016/04/22/8/) 神经网络反向传播算法的详细解释。
- [RECURRENT NEURAL NETWORKS TUTORIAL, PART 1 – INTRODUCTION TO RNNS](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) RNN 的系列学习文章。
- [A Deep Dive into Recurrent Neural Nets](http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/) RNN 深度学习文章
- [How to implement a neural network](http://peterroelants.github.io/posts/neural_network_implementation_part01/) 神经网络的系列学习文章
- [An Interactive Node-Link Visualization of Convolutional Neural Networks](http://scs.ryerson.ca/~aharley/vis/) 非常好的可视化神经网络工作原理的博客。
- [How to Code and Understand DeepMind's Neural Stack Machine](https://iamtrask.github.io/2016/02/25/deepminds-neural-stack-machine/) 
- [Neural Networks Demystified](http://lumiverse.io/series/neural-networks-demystified) 解密神经网络的视频教程。
- [Fundamentals of Deep Learning – Starting with Artificial Neural Network](http://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/) 非常详细。
- [神经网络浅讲：从神经元到深度学习](http://www.cnblogs.com/subconscious/p/5058741.html) 中文资料中难得的非常详细的资料。
- [有趣的机器学习概念纵览：从多元拟合，神经网络到深度学习，给每个感兴趣的人](https://segmentfault.com/a/1190000005746236) 中文资源中关于神经网络到深度学习的历史讲解很有意思的文章。
- [卷积神经网络CNN经典模型整理Lenet，Alexnet，Googlenet，VGG，Deep Residual Learning](http://blog.csdn.net/xbinworld/article/details/45619685) 对不同的 CNN 模型做了详细的对比介绍。
- [反向传播神经网络极简入门](http://www.hankcs.com/ml/back-propagation-neural-network.html) 这是极简？BP 得多复杂？

