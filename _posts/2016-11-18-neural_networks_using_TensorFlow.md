---
layout: post
title: 使用 TensorFlow 实现神经网络的简介
date: 2016-11-20 
tags: 机器学习  
author: 柏信   
---

## 介绍

　　如果你一直在关注 数据科学 / 机器学习 ，你不能错过深度学习和神经网络的嗡嗡声。 组织正在寻找有深度学习技能的人，无论他们在哪里。 从运行竞争到开源采购项目和支付大额奖金，人们正在尽一切可能的事情来挖掘这个有限的人才库。 自行车工程师正在被汽车行业的大枪猎杀，因为行业处于过去几十年最大的破坏的边缘！

　　如果你对潜在客户的深度学习感到兴奋，但还没有开始你的旅程 - 我在这里启用它。 从本文开始， 我将撰写一系列关于深度学习的文章，涵盖流行的深度学习图书馆及其实践实施。

　　在本文中，我将向您介绍 TensorFlow。 阅读本文后，您将能够理解神经网络的应用，并使用 TensorFlow 来解决现实生活中的问题。 这篇文章将需要你知道神经网络的基础知识和熟悉编程。 虽然本文中的代码是在 python 中，我已经关注的概念，并保持作为语言不可知的尽可能。
让我们开始吧！

<div align="center">
	<img src="/img/tfimg/logo.jpg" height="300" width="500">  
</div> 

### 目录

* [什么时候应用神经网络？](#When-to-apply-neural-net)
* [通常神经网络能解决的问题](#solve-problems)
* [了解图像数据和主流库来解决它](#popular-libraries)
* [什么是 TensorFlow？](#What-is-TensorFlow)
* [TensorFlow 一个 典型 的 “ 流 ”](#A-typical-flow)
* [在 TensorFlow 中实现 MLP](#MLP)
* [TensorFlow 的限制](#Limitations-of-TensorFlow)
* [TensorFlow 与其他库](#vs-libraries)
* [从这里去哪里？](#Where-to-go-from-here)


### <a name="When-to-apply-neural-net"></a>什么时候应用神经网络？

　　神经网络已经在相当一段时间成为焦点。 对于神经网络和深度学习上这里有更详细的解释 [点击阅读](https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/) 。 其“更深”的版本在许多领域都有取得巨大的突破，如图像识别，语音和自然语言处理等。

　　主要的问题在如何用好神经网络？ 这个领域就像一个金矿，现在，每天都会有许多新发现。 为了成为这个“淘金热”的一部分，你必须记住几件事：

* **首先，神经网络需要明确和翔实的数据（主要是与大数据）训练**， 试着想象神经网络作为一个孩子。 它首先观察它的父母走路。 然后它试图自己走，并且每一步，孩子学习如何执行一个特定的任务。 它可能会下降几次，但经过几次不成功的尝试，它学会如何走路。 如果你不 ' 吨让它走，它可能不会永远学习如何走路。 你可以为孩子提供更多的曝光，越好。

* **谨慎的做法是利用神经网络的复杂问题，如图像处理，**  神经网络属于一类称为算法代表学习算法。 这些算法分解复杂的问题分为简单的形式，使他们成为可以理解的（或 “ 可表示 ”）。 认为它是咀嚼食物之前你吞咽。 这对于传统（非表示学习）算法将更难。

* **当有适当类型的神经网络，以解决这个问题，**  每个问题都有自己的曲折。 所以数据决定你解决问题的方式。 例如，如果问题是序列生成的问题，递归神经网络更合适。 然而，如果它是图像相关的问题，你可能会更好地采取卷积神经网络的变化。

* **最后但并非最不重要的， 硬件 要求是运行了深刻的神经网络模型的关键。** 神经网被 “ 发现 ” 很久以前，但他们在近年来的主要的原因是计算资源更好，更强大的光亮。 如果你想解决这些网络的现实生活中的问题，准备购买一些高端的硬件！

### <a name="solve-problems"></a>通常神经网络解决的问题

　　神经网络是一种特殊类型的机器学习（ML）算法。 因此，作为每个 ML 算法，它遵循数据预处理，模型建立和模型评估的通常 ML 工作流。 为了简明起见，我列出了如何处理神经网络问题的 TO DO 列表。

* 检查是否就是神经网络 使您隆起比传统的算法 有问题 （参见清单中的部分上方）
* 做一个调查哪个神经网络架构最适合所需的问题
* 定义神经网络架构，通过它选择任何语言/库。
* 将数据转换为正确的格式并分批分割
* 根据您的需要预处理数据
* 增强数据以增加大小并制作更好的训练模型
* 批次供给到神经网络
* 训练和监测培训和验证数据集的变化
* 测试您的模型，并保存以备将来使用

　　对于本文，我将专注于图像数据。 所以让我们明白，首先我们深入到TensorFlow。

### <a name="popular-libraries"></a>了解图像数据和流行的库来解决它

　　图像大多排列为 3-D 阵列，尺寸指高度、宽度和颜色通道。 例如，如果你在这一刻截取你的电脑， 它将首先转换成一个 3-D 数组，然后压缩它 '.jpeg' 或 '.png' 文件格式。

　　虽然这些图像对于人类来说很容易理解，但计算机很难理解它们。 这种现象称为“语义空隙”。 我们的大脑可以看看图像，并在几秒钟内了解完整的图片。 另一方面，计算机将图像看作一个数字数组。 所以问题是我们如何解释这个图像到机器？

　　在早期，人们试图将图像分解为机器的“可理解”格式，如“模板”。 例如，面部总是具有在每个人中有所保留的特定结构，例如眼睛，鼻子或我们的脸的形状。 但是这种方法将是乏味的，因为当要识别的对象的数量将增加，“模板”将不成立。

　　快进到2012年，一个深层神经网络架构赢得了 ImageNet 的挑战，一个着名的挑战，从自然场景中识别对象。 它继续在所有即将到来的 ImageNet 挑战中统治其主权，从而证明了解决图像问题的有用性。
人们通常使用哪些库/语言来解决图像识别问题？ 其中 [最近的一次调查](https://www.analyticsvidhya.com/blog/2016/08/deep-learning-path/) 我这样做，最流行的深度学习图书馆有接口的 Python ，其次是 Lua 中， Java 和 Matlab 的。 最流行的图书馆，仅举几例，是：

* [Caffe](http://caffe.berkeleyvision.org/)
* [DeepLearning4j](http://deeplearning4j.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Theano](http://www.deeplearning.net/software/theano)
* [Torch](http://torch.ch/)

现在，您了解了图像的存储方式以及使用的常用库，让我们看看 TensorFlow 提供的内容。

### <a name="What-is-TensorFlow"></a>什么是 TensorFlow？

让我们从官方定义开始.

　　“TensorFlow 是一个开源软件库，用于使用数据流图进行数值计算。 图中的节点表示数学运算，而图边表示在它们之间传递的多维数据阵列（也称为张量）。 灵活的架构允许您使用单一 API 将计算部署到桌面，服务器或移动设备中的一个或多个 CPU 或 GPU 。

![](http://www.tensorfly.cn/images/tensors_flowing.gif)     


　　如果这听起来有点可怕 - 不要担心。 这里是我简单的定义 - 看看 TensorFlow 作为没有什么，只是 numpy 与扭曲。 如果你以前工作过 numpy ，理解 TensorFlow 将是一块蛋糕！ numpy 和 TensorFlow 之间的主要区别是 TensorFlow 遵循惰性编程范例。 它首先构建一个所有操作的图形，然后当调用“会话”时，它“运行”图形。 它通过将内部数据表示更改为张量（也称为多维数组）来构建为可扩展的。 构建计算图可以被认为是 TensorFlow 的主要成分。 为了更多地了解一个计算图形的数学结构，阅读 [这篇文章](http://colah.github.io/posts/2015-08-Backprop/) 。

　　很容易将 TensorFlow 分类为神经网络库，但它不仅仅是这样。 是的，它被设计成一个强大的神经网络库。 但它有能力做更多的事情。 您可以在其上构建其他机器学习算法，如决策树或 k-最近邻 。 你可以从字面上做你通常会做的一切 numpy ！ 它适当地称为“类固醇上的 numpy” 

**使用 TensorFlow 的优点是：**

* **它有一个直观的结构** ，因为顾名思义它有 “张量流”， 你可以轻松地可视每图的每一个部分。
* **轻松地在 cpu / gpu 上进行分布式计算** 
* **平台的灵活性**  。 您可以随时随地运行模型，无论是在移动，服务器还是 PC 上。

### <a name="A-typical-flow"></a>TensorFlow 的典型 “流”

　　每个库都有自己的“实现细节”，即一种写其遵循其编码范例的方式。 例如，当实现 scikit-learn 时，首先创建所需算法的对象，然后在列车上构建一个模型并获得测试集的预测，如下所示：

```python

# define hyperparamters of ML algorithm
clf = svm.SVC(gamma=0.001, C=100.)
# train 
clf.fit(X, y)
# test 
clf.predict(X_test)
```

正如我前面所说，TensorFlow 遵循一种懒惰的方法。 在 TensorFlow 中运行程序的通常工作流程如下：

* **建立一个计算图**， 这可以是任何的数学运算 TensorFlow 支撑。
* **初始化变量**， 编译预先定义的变量
* **创建会话**， 这是 神奇的开始的 地方 ！
* **在会话中运行图**， 编译图形被传递到会话，它开始执行它。
* **关闭会话**， 结束这次使用。

TensoFlow　中使用的术语很少   

```
placeholder：将数据输入图形的一种方法
feed_dict：将数值传递到计算图的字典
```

让我们写一个小程序来添加两个数字！

```pyhton

# import tensorflow
import tensorflow as tf

# build computational graph
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

addition = tf.add(a, b)

# initialize variables
init = tf.initialize_all_variables()

# create session and run the graph
with tf.Session() as sess:
    sess.run(init)
    print "Addition: %i" % sess.run(addition, feed_dict={a: 2, b: 3})

# close session
sess.close()
```

### <a name="MLP"></a>在 TensorFlow 中实现神经网络

*注意：我们可以使用不同的神经网络体系结构来解决这个问题，但是为了简单起见，我们在深入实施中讨论了前馈多层感知器。*

让我们记住我们对神经网络的了解。

神经网络的典型实现如下：

* 定义要编译的神经网络体系结构
* 将数据传输到模型
* 在引擎盖下，数据首先被分成批次，以便它可以被摄取。 首先对批次进行预处理，扩增，然后送入神经网络进行训练
* 然后，模型被逐步地训练
* 显示特定数量的时间步长的精度
* 训练后保存模型供将来使用
* 在新数据上测试模型并检查其运行方式

在这里，我们解决了我们深刻的学习实践中的问题 - [确定数字](https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-digits/) 。 让我们花一点时间看看我们的问题陈述。

　　我们的问题是图像识别，以识别来自给定的 28×28 图像的数字。 我们有一个图像子集用于训练，其余的用于测试我们的模型。 所以首先，下载火车和测试文件。 数据集包含数据集中所有图像的压缩文件， train.csv 和 test.csv 都有相应的列车和测试图像的名称。 数据集中不提供任何其他功能，只是原始图像以 “.png” 格式提供。

　　正如你所知道的，我们将使用 TensorFlow 来创建一个神经网络模型。 所以你应该首先在你的系统中安装 TensorFlow 。 请参考 [官方的安装指南](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md) 进行安装，按您的系统规格。

我们将按照上述模板

* 让我们来 导入所有需要的模块

```python

%pylab inline

import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf

```

* 让我们来 设置一个种子值，这样我们就可以控制我们的模型随机性

```python

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

```

* 第一步是设置目录路径，以便保管！

```python

root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

```

* 现在让我们读取我们的数据集。 这些是.csv格式，并有一个文件名以及相应的标签

```python

train = pd.read_csv(os.path.join(data_dir，'Train'，'train.csv'))
test = pd.read_csv(os.path.join（data_dir，'Test.csv'))
sample_submission = pd.read_csv(os.path.join(data_dir，'Sample_Submission.csv'))
train.head()

```

|    | 文件名 | 标签 |
| -- |:-----:| ---:|
|  0 | 0.png |  4  |
|  1 | 1.png |  9  |
|  2 | 2.png |  1  |
|  3 | 3.png |  7  |
|  4 | 4.png |  3  |


* 让我们看看我们的数据是什么样子！ 我们读取我们的形象并显示出来。

```python

img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)

img = imread(filepath, flatten=True)

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

```

![](https://www.analyticsvidhya.com/wp-content/uploads/2016/10/3.png)       

上面的图像表示为 numpy 数组，如下所示

![](https://www.analyticsvidhya.com/wp-content/uploads/2016/10/one.png)       


* 为了方便数据操作，让我们 的存储作为 numpy 的阵列的所有图片

```python

temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
train_x = np.stack(temp)

temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
test_x = np.stack(temp)
```

* 由于这是典型的 ML 问题，为了测试我们的模型的正确功能，我们创建一个验证集。 让我们 以 70:30 的分割大小车组 VS 验证集

```python

split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

```

* 现在我们定义一些辅助函数，我们稍后在我们的程序中使用


```python

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, 784)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

```

* 现在是主要部分！ 让我们定义我们的神经网络架构。 我们定义一个神经网络具有 3 层; 输入，隐藏和输出。 输入和输出中的神经元数目是固定的，因为输入是我们的 28×28 图像，并且输出是表示类的 10×1 向量。 我们在隐藏层中取 500 神经元。 这个数字可以根据您的需要变化。 我们也 值 赋给 其余变量。 阅读 [神经网络的基础知识的文章](https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/) ，以了解更多的它是如何工作的深度。

```python

### set all variables

# number of neurons in each layer

input_num_units = 28*28

hidden_num_units = 500

output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

```

* 现在创建我们的神经网络计算图

```python

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

```

* 此外，我们需要定义神经网络的成本

```python

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

```

* 并设置优化器，即我们的反向传播算法。 这里我们使用 [Adam](https://arxiv.org/abs/1412.6980) ，这是梯度下降算法的高效变体。 有在 tensorflow 可用许多其它优化（参照 [此处](https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers) ）

```python

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

```
* 定义我们的神经网络结构后，让我们来 初始化所有的变量

```python

init = tf.initialize_all_variables()

```

* 现在让我们创建一个会话，并在会话中运行我们的神经网络。 我们还验证我们创建的验证集的模型准确性

```python

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
    
    print "\nTraining complete!"
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, 784), y: dense_to_one_hot(val_y.values)})
    
    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, 784)})
```

这将是上面代码的输出

```python

Epoch: 1 cost = 8.93566
Epoch: 2 cost = 1.82103
Epoch: 3 cost = 0.98648
Epoch: 4 cost = 0.57141
Epoch: 5 cost = 0.44550

Training complete!
Validation Accuracy: 0.952823	

```

* 为了测试我们用我们自己的眼睛的模式， 让我们来 想象它的预言

```python

img_name = rng.choice(test.filename)
filepath = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)

img = imread(filepath, flatten=True)
 
test_index = int(img_name.split('.')[0]) - 49000

print "Prediction is: ", pred[test_index]

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

```

```python

Prediction is:  8

```

![](https://www.analyticsvidhya.com/wp-content/uploads/2016/10/8.png)       




* 我们看到我们的模型性能是相当不错！ 现在 ，让我们 创建一个提交

```python

sample_submission.filename = test.filename
 
sample_submission.label = pred

sample_submission.to_csv(os.path.join(sub_dir, 'sub01.csv'), index=False)

```

并做了！ 我们刚刚创建了我们自己的训练神经网络！

### <a name="Limitations-of-TensorFlow"></a>TensorFlow 的限制

* 尽管 TensorFlow 是强大的， 它 仍然 是 一个低水平库。 例如，它可以被认为是机器级语言。 但对于大多数功能，您需要自己去模块化和高级接口，如 keras
* 它仍然在继续开发和维护，这是多么迷人啊！
* 它取决于你的硬件规格，越牛越好
* 仍然不是许多语言的 API。
* TensorFlow 中仍然有很多库需要手动包括在内，比如 OpenCL 支持。

上面提到的大多数是在TensorFlow开发人员的蓝图， 他们已经制定了一个路线图，指定库未来应该如何开发。

### <a name="vs-libraries"></a>TensorFlow 与其他库

　　TensorFlow 建立在类似的原理，如使用数学计算图表的 Theano 和 Torch 。 但是随着分布式计算的额外支持，TensorFlow 更好地解决复杂的问题。 此外，TensorFlow模型的部署已经被支持，这使得它更容易用于工业目的，打开商业图书馆，如 Deeplearning4j ，H2O 和 Turi。 TensorFlow 有用于 Python，C ++ 和 Matlab 的 API 。 最近还出现了对 Ruby 和 R 等其他语言的支持。因此，TensorFlow 试图获得通用语言支持。

### <a name="Where-to-go-from-here"></a>从这里去哪里？

　　以上你看到了如何用 TensorFlow 构建一个简单的神经网络。 这段代码是为了让人们了解如何开始实现 TensorFlow，所以带上一点盐。 记住，要解决更复杂的现实生活中的问题，你必须调整代码一点。

　　许多上述功能可以被抽象为给出无缝的端到端工作流。 如果你使用 scikit-learn ，你可能知道一个高级库如何抽象“底层”实现，给终端用户一个更容易的界面。 尽管 TensorFlow 已经提取了大多数实现，但是出现了高级库，如 TF-slim 和 TFlearn。

### 有用的资源
* [TensorFlow 官方库](https://github.com/tensorflow/tensorflow) 
* Rajat Monga（TensorFlow技术负责人） [“TensorFlow为大家”](https://youtu.be/wmw8Bbb_eIE)  的视频
* [一个专用资源的策划列表](https://github.com/jtoy/awesome-tensorflow/#github-projects)  


### 关于原文

感谢原文作者 [Faizan Shaikh](https://www.analyticsvidhya.com/blog/author/jalfaizy/) 的分享，
这篇文章是在 [An Introduction to Implementing Neural Networks using TensorFlow](https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/) 的基础上做的翻译，如果发现翻译中有不对或者歧义的的地方欢迎在下面评论里提问，我会加以修正 。



