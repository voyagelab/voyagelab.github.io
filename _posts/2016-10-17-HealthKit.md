---
layout: post
title: HealthKit睡眠分析
date: 2016-10-17 
tags: 技术   
author: Robin  
---


随着移动手机设备硬件的进步，持续带来了各种人性化的指标分析体系，例如运动数据统计分析、饮食习惯统计分析等等，大大增强了人类对于自身各种活动的认知和理解。而在这个快节奏的时代，睡眠质量的分析比以往任何时候都更加的有意义，一场睡眠革命正在悄然的在人们的生活中崛起。通过针对睡眠质量的统计分析，能够清晰的获知人们睡眠的开始、结束、从而显露出睡眠的趋势等等。

苹果提供了一种非常酷的方式以用来于用户的个人健康信息进行数据交流，并且保持了数据安全和存储安全，这就是苹果在iOS系统中内建的健康应用 --- Health。而Health应用程序使用的技术框架是苹果专门打造的一款健康服务框架 --- HealthKit，开发者不仅可以使用此框架构建自身的健康类应用程序，而且还允许访问健康应用中的数据分析结果。

在Health应用程序中，并没有自动去统计用户的睡眠起始时间等数据，但是Health为我们提供了一套数据写入和分析功能，开发者可以依照特定的数据格式将睡眠数据写入到Health，通过Health分析后，得到数据的分析结果。

### 简介

HealthKit 框架提供了一种保存数据的加密数据库叫做`Health Store`，可以使用`HKHealthStore`类来访问此数据库，而且不仅在iPhone还是Apple Watch上都有自己的`Health Store`。健康数据可以在iPhone和Apple Watch之间进行同步，不仅如此，`Health Store`也会定期的清除老旧的数据，以节省存储空间。需要注意的是HealthKit和Health应用程序均不支持iPad。

如果需要针对健康数据构建一个iOS应用或者watchOS应用，HealthKit是首选的框架工具。HealthKit能够统一管理来自各种数据源的数据，根据用户的喜好自动合并来自不同数据源的数据，并且能够访问到每个数据来源平台的数据，并将数据合并。例如多个睡眠分析的应用程序均向健康应用写入了睡眠时间数据，在健康应用中，你可以看到健康应用将多个时间进行合并到了一起，统一作为睡眠分析的数据来源，并通过可视化进行图表展示，用户还可以在健康应用中选择日、周、月、年来查看合并后的睡眠情况。

![](/img/healthKit/healthkit.png)

这样不仅可以对用户的体征进行测量记录，运动健身统计或者饮食营养数据统计，还可以用于针对睡眠数据进行分析等。

在接下来，将使用Healthkit框架访问和保存用户的睡眠数据，了解用户的睡眠情况，同样的方法在watchOS上也是适用的，工程样例和代码使用了Swift 3.0 和Xcode 8 进行构建。

在开始之前，可以下载[Starter project](https://github.com/RobinChao/SleepAnalysis/blob/master/SleepAnalysisStarter.zip)，此开始工程中已经创建了用户界面以及一些方法，当运行了此开始工程后，可以看到界面上有一个计时器数据展示标签，按下开始按钮后，计时器读数将会持续变化。

### 使用HealthKit框架

我们的目标应用程序主要的功能是保存用户的睡眠分析信息和使用`开始`和`停止`按钮检索数据。为了使用HealthKit，你必须给予应用程序HealthKit功能程序包，选择工程当前的`target -> capabilities`，打开HealthKit功能程序包。

![](/img/healthKit/HealthKit-allow-1024x640.png)

开启此功能包以后，根据iOS10最新的权限管理机制，还需要在工程的info.plist文件中配置全新说明，针对HealthKit来说，由于需要进行数据的读写，因此需要配置`NSHealthUpdateUsageDescription`和`NSHealthShareUsageDescription`两个字段。

接下来，需要在`ViewController`类中创建一个`HKHealthStore`实例：

```swift
let healthStore = HKHealthStore()
```

之后将使用`HKHealthStore`实例访问HealthKit数据库。

如前所述，HealthKit需要用户授权才能够访问健康数据，所以必须首先向用户请求权限许可才能够访问（读/写）睡眠分析数据。因此在`ViewController`中的`viewDidLoad`方法中，进行权限的申请：


```swift
 override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let typestoRead = Set([
            HKObjectType.categoryType(forIdentifier: HKCategoryTypeIdentifier.sleepAnalysis)!
            ])
        
        let typestoShare = Set([
            HKObjectType.categoryType(forIdentifier: HKCategoryTypeIdentifier.sleepAnalysis)!
            ])
        
        self.healthStore.requestAuthorization(toShare: typestoShare, read: typestoRead) { (success, error) -> Void in
            if success == false {
                 NSLog(" Display not allowed")
            }
        }
    } 
```

这段代码将弹出申请权限提示页，用户可以允许和拒绝请求，在回调Block中，可以处理成功或者失败的组中结果，当用户拒绝了权限的申请请求，会有错误的信息提示。
但是为了测试方便，每次权限申请的时候，可以直接选择`允许`，允许访问设备上的健康数据。

![](/img/healthKit/Health-App-Permission.png)

### 写睡眠分析数据

在写数据之前，如何才能检索睡眠分析数据呢？根据苹果的[官方文档](https://developer.apple.com/reference/healthkit/hkcategoryvaluesleepanalysis)，每个睡眠分析样本只能有一个值。为了表示用户是`在床上`还是`睡着了`还是`醒着`，HealthKit使用两个或两个以上的重叠时期样本。通过比较这些样本的开始和结束时间，应用程序可以进行二次计算进行统计分析：

* 用户入睡的时间
* 在床上的时间占比
* 用户在床上醒来的次数
* 在床上和在床上睡着的时间总量

![](/img/healthKit/record_sleep_data-1024x525.png)

简而言之，可以按照以下的步骤来保存睡眠分析数据到HealthStore：

1. 定义两个`NSDate`对象，分别代表开始时间和结束时间
2. 使用`HKCategoryTypeIdentifierSleepAnalysis`创建`HKObjectType`实例
3. 创建一个新的`HKCategorySample`对象实例，通常情况下使用类别样本记录睡眠数据，个别的样本数据正好代表了用户是在床上还是睡着了，因此需要创建一个在床上的样本对象（inBedSample）和睡着了的样本对象（asleepSample）
4. 最后，使用`HKHealthStore`的`saveObject`方法，保存这些对象即可

```swift
 func saveSleepAnalysis() {
        if let sleepType = HKObjectType.categoryType(forIdentifier: HKCategoryTypeIdentifier.sleepAnalysis) {
            
            // we create new object we want to push in Health app
            
            let inBedSample = HKCategorySample(type:sleepType, value: HKCategoryValueSleepAnalysis.inBed.rawValue, start: self.alarmTime, end: self.endTime)
            
            // we now push the object to HealthStore
            
            healthStore.save(inBedSample, withCompletion: { (success, error) -> Void in
                
                if error != nil {
                    
                    // handle the error in your app gracefully 
                    return
                    
                }
                
                if success {
                    print("My new data inBedSample was saved in Healthkit")
                    
                } else {
                    // It was an error again
                    
                }
                
            })
            
            
            let asleepSample = HKCategorySample(type:sleepType, value: HKCategoryValueSleepAnalysis.asleep.rawValue, start: self.alarmTime, end: self.endTime)
            
            
            healthStore.save(asleepSample, withCompletion: { (success, error) -> Void in
                
                if error != nil {
                    
                    // handle the error in your app gracefully
                    return
                    
                }
                
                if success {
                    print("My new data asleepSample was saved in Healthkit")

                } else {
                    // It was an error again
                    
                }
                
            }) 
        }
    }
```

这个方法可以在希望保存睡眠分析数据的时候调用。

### 读取睡眠分析数据

为了读取或者叫检索睡眠分析数据，首先需要创建一个查询，在创建查询之前，定义需要的对象类型，可以使用`HKCategoryTypeIdentifierSleepAnalysis`中的`HKObjectType`来定义。或许还需要使用正则式来过滤`startDate`和`endDate`之间的数据，可以通过`HKQuery`的`predicateForSamplesWithStartDate`来创建一个正则式。为了使得数据能够带有顺序，比如按照时间升序或者降序，可以创建一个`sortDescriptor`来排序数据。

整个检索数据的代码段如下：

```swift
func retrieveSleepAnalysis() {
        
        // startDate and endDate are NSDate objects
        
       // ...
        
        // first, we define the object type we want
        
        if let sleepType = HKObjectType.categoryType(forIdentifier: HKCategoryTypeIdentifier.sleepAnalysis) {
            
            // You may want to use a predicate to filter the data... startDate and endDate are NSDate objects corresponding to the time range that you want to retrieve
            
            //let predicate = HKQuery.predicateForSamplesWithStartDate(startDate,endDate: endDate ,options: .None)
            
            // Get the recent data first
            
            let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
            
            // the block completion to execute
            
            let query = HKSampleQuery(sampleType: sleepType, predicate: nil, limit: 30, sortDescriptors: [sortDescriptor]) { (query, tmpResult, error) -> Void in
                
                if error != nil {
                    
                    // Handle the error in your app gracefully
                    return
                    
                }
                
                if let result = tmpResult {
                    
                    for item in result {
                        if let sample = item as? HKCategorySample {
                            
                            let value = (sample.value == HKCategoryValueSleepAnalysis.inBed.rawValue) ? "InBed" : "Asleep"
                            
                            print("Healthkit sleep: \(sample.startDate) \(sample.endDate) - value: \(value)")
                        }
                    }
                }
            }
            
            
            healthStore.execute(query)
        }
    
    }
```

这段代码查询了健康应用中的说有数据，然后进行了降序排列。每次查询都打印出了样本的开始时间和结束时间，以及在这个时间段内用户的睡眠状态，例如，在床上或者睡眠。代码中限制了数据条目的数量30条，这样查询会只检索30条数据，然后返回，也可以使用正则式来进行指定时间段数据的检索等。

### 应用测试

对于Demo应用程序来说，已经设定了一个`NSTimer`来对睡眠情况的计时。`NSDate`对象会在用户点击开始按钮后，记录下用户的睡眠开始时间点，点击停止按钮后，记录下睡眠结束的时间点，并最终用于保存睡眠数据的一部分。在结束按钮点击时，需要将记录时间段的数据保存，同时检索更新后的数据，用户界面展示。

```swift
    @IBAction func stop(_ sender: AnyObject) {
        endTime = Date()
        self.saveSleepAnalysis()
        self.retrieveSleepAnalysis()
        timer.invalidate()
    }
```

此时运行工程，点击开始按钮，等待一段时间后，点击停止按钮，完整后，打开健康应用，进入睡眠分析，就可以看到相关的数据了。

![](/img/healthKit/sleep-analysis-test.jpg)

### HealthKit应用的一些建议

HealthKit旨在提供一个公共的平台，应用程序开发人员能够很容易地共享和访问用户的数据，并且避免了可能的重复或者不一致的数据。苹果审核指南非常具体的指明了使用HealthKit和请求读/写数据的权限的要求，如果不满足这些要求，应用程序会被拒绝上架。并且针对用户数据的使用不当，也被会拒绝。这就意味着，不能够写入不合规或者不合理的数据。

获取完整的Demo应用，可以在[这里](https://github.com/RobinChao/SleepAnalysis)下载使用。
