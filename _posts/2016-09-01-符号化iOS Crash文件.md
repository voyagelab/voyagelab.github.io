---
layout: post
title: 符号化iOS Crash文件
date: 2016-09-01 
tags: 教程   
author: 柏信  
---


1、浏览器打开[TalingData官网](www.talkingdata.com)     
2、点击[应用统计分析](https://www.talkingdata.com/products.jsp?languagetype=zh_cn) --> [立即开始](https://www.talkingdata.com/app/new/main.html?zh_cn&1b504b736df6#/productCenter)       
3、选择你创建的产品（如果没有产品，点击创建新应用）。      
4、`选择用户和使用` / `错误报告` / `错误列表` ， 如下图 (如果你的App没崩溃过，下面列表将是空的)

![](/img/dSYM/01.png)

5、点击第一行的崩溃日志，进入到错误详情列表，如下：

![](/img/dSYM/02.png)   


6、点击第一列，会有一个错误详情的弹框，内容如下：

![](/img/dSYM/06.png)   

注意：红线 `3  Demo     0x00000001000d76a4 Demo + 46756	` 看不到详情信息，是由于App使用Xcode打包成.ipa时经过符号化处理，代码变成了二进制，人眼是无法直接识别的。

这时我们需要借助 dSYM 符号化工具 [dSYMTools](https://github.com/answer-huang/dSYMTools)，可以直接点击[这里下载](https://pan.baidu.com/s/1mg01Qha) ，下载完成后会得到一个 dSYM 应用，双击打开它。 

　　　　 ![](/img/dSYM/03.png) 
 
 打开后
 

![](/img/dSYM/04.png)   

把Demo. demo.app.dSYM 文件拖进去后，会让你选择对应的编译类型，编译类型在，TalkingData的错误详情下面有：`Architecture: arm64` ，然后选中dsYM文件的UUID就会自动填进去。

注意：dSYM 应用显示的UUID 和 TalingData 错误详情的UUID必须是一致的才行。

![](/img/dSYM/07.png)   

右图中可能有错误的地方，就是解析出来的结果。











