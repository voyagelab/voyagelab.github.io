---
layout: post
title: Android 内存泄漏分析
date: 2018-09-05 
categories: Android
author: 韩广利
--- 


对于Android App内存泄漏问题开发者会比较困扰，并且类似问题难以复现，比较难定位，一但发生很可能是灾难性的，有千里之堤毁于蚁穴之势。App开发者对内存泄漏问题能够在测试阶段发现并且解决是非常重要的。


什么是内存泄漏？简单来讲是在开发应用时，没有释放掉不需要的内存资源。比如对象不需要了，但是由于没有释放对它的引用， GC(Garbage Collection) 无法回收相应的内存资源，这部分内存就无法被利用了。这种情况就是所谓的“内存泄漏”。

Android内存问题涉及的知识点还是非常多的，下面从开发者角度把内存泄漏问题分三个部分来介绍一下，目的是能够对于内存泄漏问题有一个清晰的了解和分析：

- Android内存管理
- 内存分析工具使用
- 内存泄漏demo


### 1.Android内存管理

Android每个App默认是运行在一个独立进程中， 这个进程运行在一个独立的VM（Virtual Machine）空间，可以参考下面JVM架构图。Android是怎么管理这些App的内存的呢？

<img src="/img/android_memory_leak_analysis/jvm_architecture.png" />

Android 4.4之前一直使用的是Dalvik虚拟机作为App的运行的VM， Android 4.4中引入了ART (Android Runtime)作为备选VM，Android 5.0起正式将ART作为默认VM。


ART相比Dalvik有以下优点：

**1) Ahead-of-time (AOT) compilation instead of Just-in-time (JIT)** 

Dalvik中采用的是JIT来做动态翻译的，将dex或odex中并排的Dalvik code运行态翻译成native code去执行。JIT的引入使得Dalvik提升了3~6倍的性能，而在ART中完全抛弃了Dalvik的JIT，使用了AOT，直接在安装时用dex2oat将其完全翻译成native code。这一技术的引入，使得VM执行指令的速度又一重大提升。

**2) Improved garbage collection**

|--|--|
|**Dalvik GC 的过程** |  **ART GC 的过程** |
|1、当GC被触发时候，其会去查找所有活动的对象，这个时候整个程序与虚拟机内部的所有线程就会挂起，这样目的是在较少的堆栈里找到所引用的对象。需要注意的是这个回收动作是和应用程序同时执行(非并发)。 </p>2、GC对符合条件的对象进行标记</p>3、GC对标记的对象进行回收 </p>4、恢复所有线程的执行现场继续运行|1、GC将会锁住java堆，扫描并进行标记</p>2、标记完毕释放掉java堆的锁，并且挂起所有线程</p>3、GC对标记的对象进行回收</p>4、恢复所有线程的执行现场继续运行</p>5、重复2-4直到结束|


ART主要的改善点在将其非并发过程改变成了部分并发。另外对内存重新分配管理，使得执行时间缩短，据官方测试数据，GC效率提高了l2倍。

**3) Improved memory usage and reduce fragmentation**

Dalvik的内存管理特点是内存碎片化严重，当然这也是Mark and Sweep算法带来的弊端。该算法分为两个阶段：标记(mark)和清除(sweep)。 

在标记阶段，collector从mutator根对象开始进行遍历，对从根对象可以访问到的对象都打上一个标识，一般是在对象的header中，将其记录为可达对象。

在清除阶段，collector对堆内存(heap memory)从头到尾进行线性的遍历，如果发现某个对象没有标记为可达对象-通过读取对象的header信息，则就将其回收。

<img src="/img/android_memory_leak_analysis/reduce_fragmentation.png" />

ART的解决方案：它将java分了一块空间命名为Large-Object-Space，这块内存空间的引入用来专门存放large object。同时ART又引入了moving collector的技术，即将不连续的物理内存块进行对齐。对齐了后内存碎片化就得到了很好的解决。Large-Object-Space的引入能够有效提高内存的利用率，根官方统计，ART的内存利用率提高了10倍左右。


Dalvik和ART都是使用paging和memory-mapping(mmapping)来管理内存的。这就意味着， 任何被分配的内存都会持续存在， 唯一释放这块内存方式就是释放对象引用(让对象GC Root不可达)， 故而让GC程序来回收内存。关于如何管理应用的进程与内存分配，Android官网有比较详细说明。


### 2.内存分析工具使用

工欲善其事，必先利其器。接下来讲讲工具：

接下来会介绍三个内存分析工具：Android Studio自带的Memory Monitor、MAT工具和插件工具Leakcanary

#### 工具1：Memory Monitor：

Memory Monitor 是 Android Studio内置的， 官方的内存监测工具，它是图形化的展示当前应用的内存状态， 包括已分配内存， 空闲内存， 内存实时动态等。

<img src="/img/android_memory_leak_analysis/memory_monitor.png" />

图中标注1：GC按钮， 点击执行一次GC操作。

图中标注2：Dump Java Heap按钮， 点击会在该调试工程的captures目录下生成一个类似这样"com.talkingdata.demo_2017.08.09_13.35.hprof"命名的hprof文件。针对文件的分析可以参考Google官网描述：HPROF Analyzer。

图中标注3：Allocation Traking按钮， 点击一次开始， 再次点击结束， 同样会在captrures目录生成一个文件， 类似" com.talkingdata.demo_2017.08.09_13.35.alloc"命名的alloc文件，针对文件的分析可以参考Google官网描述：Allocation Tracker。

#### 工具2：MAT

Eclipse MAT(Memory Analyzer)是一个快速且功能丰富的Java Heap分析工具, 可以帮助我们寻找内存泄露, 减少内存消耗。

MAT可以分析程序(成千上万的对象产生过程中)生成的Heap dumps文件, 它会快速计算出对象的Retained Size, 来展示是哪些对象没有被GC, 自动生成内存泄露疑点的报告。详细的使用方法可以查阅官方文档。下面简单介绍常用的方式：

**获取heap dumps**

可以使用 Android Studio 获取 heap dump。点击 Monitors 中的 Dump Java Heap 按钮后，会得到一个 .hprof 文件。生成的 .hprof 文件默认在项目的根目录的 captures 目录下。

<img src="/img/android_memory_leak_analysis/heap_dumps.png" />

因为MAT是用来分析Java程序的hprof文件的，和Android导出的hprof文件的格式有一定区别，所以需要转换为标准格式。有两种方式可供选择：Android SDK中给我们提供了转换的工具，即 platform-tools/hprof-conv ，使用如下命令即可转换我们的hprof文件格式：

hprof-conv [源hprof文件的路径] [输出的hprof文件路径]
或者选择Captures选项卡，右键相应的 .hprof 文件，并选择 Export to standard .hprof。

<img src="/img/android_memory_leak_analysis/hprof.png" />

接下来，我们就可以使用MAT工具进行内存分析了。

**使用MAT分析：**

MAT提供了许多视图供我们使用，具体如下：

打开MAT工具，并加载之前导出的hprof文件，会进入Overview界面。可以从界面中看到Retained Size最大的几个对象。

<img src="/img/android_memory_leak_analysis/mat.png" />

- Histogram

	它列出了按类别分组的对象，MAT可以非常快速地计算各类别的大小和个数，并显示在列表中，这是深入分析的重要指标。它有多种分组方式：Group by class/Group by superclass/Group by class loader/Group by package。甚至还可以按线程分组，不过这需要打开thread_overview。

	<img src="/img/android_memory_leak_analysis/histogram.png" />

- Dominator Tree

	Dominator Tree列出了最大的对象。下一级别会显示那些被立即阻止垃圾回收的对象。右键单击可以查看传出和传入的引用或查看Path to GC Roots，以查看保留对象的引用链。

	<img src="/img/android_memory_leak_analysis/dominator_tree.png" />

- Path to GC Roots

	GC Roots的路径显示了阻止对象被垃圾回收的引用链。有黄点的对象是GC Roots，即被假定为活着的对象。通常GC Roots是当前在线程或系统类的调用堆栈上的对象。用这个方法可以快速找到对象没有被回收的原因。

	<img src="/img/android_memory_leak_analysis/path_to_gc_roots.png" />


#### 工具3：Leakcanary

LeakCanary是square出的一款开源的用来做内存泄露检测的工具。
被测试App集成LeakCanary之后, 工具检测到潜在的内存泄露后, 会弹出Toast提示，并在测试手机桌面生成一个Leaks的icon:

<img src="/img/android_memory_leak_analysis/leakcanary.png" />

<img src="/img/android_memory_leak_analysis/leakcanary_icon.png" />


点击该icon进入Leaks界面, 可以比较清晰的看到内存泄露疑点：

<img src="/img/android_memory_leak_analysis/leakcanary_info.png" />

对于源码感兴趣的同学可以参考下面：

**源码文件结构说明：**

1. ├── AbstractAnalysisResultService.java   
2. ├── ActivityRefWatcher.java  -- Activity监控者，监控其生命周期  
3. ├── AndroidDebuggerControl.java --Android Debug控制开关，就是判断Debug.isDebuggerConnected()  
4. ├── AndroidExcludedRefs.java   -- 内存泄漏基类  
5. ├── AndroidHeapDumper.java     --生成.hrpof的类  
6. ├── AndroidWatchExecutor.java  -- Android监控线程，延迟5s执行  
7. ├── DisplayLeakService.java    -- 显示通知栏的内存泄漏，实现了AbstractAnalysisResultService.java  
8. ├── LeakCanary.java            --对外类，提供install(this)方法  
9. ├── ServiceHeapDumpListener.java   
10. └── internal --这个文件夹用于显示内存泄漏的情况（界面相关）  
11.     ├── DisplayLeakActivity.java --内存泄漏展示的Activity  
12.     ├── DisplayLeakAdapter.java  --内存泄漏展示ListView适配器  
13.     ├── DisplayLeakConnectorView.java --内存泄漏展示连接器  
14.     ├── FutureResult.java  
15.     ├── HeapAnalyzerService.java 在另一个进程启动的Service,用于接收数据并发送数据到界面  
16.     ├── LeakCanaryInternals.java  
17.     ├── LeakCanaryUi.java  
18. └── MoreDetailsView.java  

**工作机制：**

- RefWatcher创建

	<img src="/img/android_memory_leak_analysis/refWatcher.png" />

- watch()方法使用

	<img src="/img/android_memory_leak_analysis/watch.png" />	

	1. RefWatcher.watch() 创建一个 KeyedWeakReference 到要被监控的对象。
	2. 然后在后台线程检查引用是否被清除，如果没有，调用GC。
	3. 如果引用还是未被清除，把 heap 内存 dump 到 APP 对应的文件系统中的一个 .hprof 文件中。
	4. 在另外一个进程中的 HeapAnalyzerService 有一个 HeapAnalyzer 使用HAHA 解析这个文件。
	5. 得益于唯一的 reference key, HeapAnalyzer 找到 KeyedWeakReference，定位内存泄露。
	6. HeapAnalyzer 计算 到 GC roots 的最短强引用路径，并确定是否是泄露。如果是的话，建立导致泄露的引用链。
	7. 引用链传递到 APP 进程中的 DisplayLeakService， 并以通知的形式展示出来。

- 检测 Activity

	1. 在 `Application onCreate()` 中调用 `LeakCanary.install(this)`

		Github 示例：

			public class ExampleApplication extends Application {  
			  @Override   
			  public void onCreate() {  
			    super.onCreate();  
			    if (LeakCanary.isInAnalyzerProcess(this)) {  
			      // This process is dedicated to LeakCanary for heap analysis.  
			      // You should not init your app in this process.  
			      return;  
			    }  
			    LeakCanary.install(this);  
			    // Normal app init code...  
			  }  
			}  

	2. `LeakCanary.install()` 会返回一个 `RefWatcher`

			public static RefWatcher install(Application application) {  
			    return refWatcher(application).listenerServiceClass(DisplayLeakService.class)  
			        .excludedRefs(AndroidExcludedRefs.createAppDefaults().build())  
			        .buildAndInstall();  
			}  

	3. `buildAndInstall()` 同时也会启用一个 `ActivityRefWatcher`，用于自动监控调用`Activity.onDestroy()` 之后泄露的 `activity`。

			public RefWatcher buildAndInstall() {  
			    RefWatcher refWatcher = build();  
			    if (refWatcher != DISABLED) {  
			     LeakCanary.enableDisplayLeakActivity(context);  
			     ActivityRefWatcher.install((Application) context, refWatcher);  
			    }  
			    return refWatcher;  
			  }  
			  
			public void watchActivities() {  
			    // Make sure you don't get installed twice.  
			    stopWatchingActivities();  
			    //注册 LifecycleCallbacks,用于观察activity是否被回收  
			    application.registerActivityLifecycleCallbacks(lifecycleCallbacks);  
			    }  
			  
			void onActivityDestroyed(Activity activity) {  
			  refWatcher.watch(activity);  
			}  

		注：registerActivityLifecycleCallbacks 时API 14引入的监控方式，如要兼容 API 14 以下版本，请重写Activity onDestroy()在其中调用refWatcher.watch(activity)

- 检测Fragment

		public abstract class BaseFragment extends Fragment {  
		 @Override   
		 public void onDestroy() {  
		     super.onDestroy();  
		     RefWatcher refWatcher = ExampleApplication.getRefWatcher(getActivity());  
		     refWatcher.watch(this);  
		    }  
		}

- 检测其他对象

		RefWatcher refWatcher = ExampleApplication.getRefWatcher(this);  
		refWatcher.watch(Object); 

- 查看Log

		...
		xxx:leakcanary D/LeakCanary: In com.talkingdata.demo:1.0:1.
		xxx:leakcanary D/LeakCanary: * com.talkingdata.demo.app.AppBaseFunction has leaked:
		xxx:leakcanary D/LeakCanary: * GC ROOT android.location.LocationManager$ListenerTransport.mListener
		xxx:leakcanary D/LeakCanary: * references com.talkingdata.demo.BaseActivity$LocationTracker.mContext
		xxx:leakcanary D/LeakCanary: * leaks com.talkingdata.demo.app.AppBaseFunction instance
		xxx:leakcanary D/LeakCanary: * Retaining: 1.2 kB.
		xxx:leakcanary D/LeakCanary: * Reference Key: 11489a95-635c-44f5-831d-38fa21bb0595
		xxx:leakcanary D/LeakCanary: * Device: Huawei google Nexus 6P angler
		xxx:leakcanary D/LeakCanary: * Android Version: 8.0.0 API: 26 LeakCanary: 1.6-SNAPSHOT 
		xxx:leakcanary D/LeakCanary: * Durations: watch=5016ms, gc=168ms, heap dump=1248ms, analysis=89686ms
		xxx:leakcanary D/LeakCanary: * Details:
		...

### 3.内存泄漏Demo


假设有一个单例的ListenerManager，可以add/remove Listener，有一个Activity，实现了该Listener，且这个Activity中持有大对象BigObject，BigObject中包含一个大的字符串数组和一个Bitmap List。

代码片段如下：

**ListenerManager**

	public class ListenerManager {  
	    private static ListenerManager sInstance;  
	    private ListenerManager() {}  
	    private List<SampleListener> listeners = new ArrayList<>();  
	    public static ListenerManager getInstance() {  
	        if (sInstance == null) {  
	            sInstance = new ListenerManager();  
	        }  
	        return sInstance;  
	    }  
	    public void addListener(SampleListener listener) {  
	        listeners.add(listener);  
	    }  
	    public void removeListener(SampleListener listener) {  
	        listeners.remove(listener);  
	    }  
	}  

**MemoryLeakActivity**

	public class MemoryLeakActivity extends AppCompatActivity implements SampleListener {  
	    private BigObject mBigObject = new BigObject();  
	    @Override  
	    protected void onCreate(Bundle savedInstanceState) {  
	        super.onCreate(savedInstanceState);  
	        setContentView(R.layout.activity_memory_leak);  
	        ListenerManager.getInstance().addListener(this);  
	    }  
	    @Override  
	    public void doSomething() {  
	    }  
	}

#### 1) 使用Memory Monitor分析

启动我们要检测的Activity(MemoryLeakActivity)， 然后退出， 在monitor中查看内存变化。操作步骤和结果如下：

步骤1：点击"Analyzer Tasks"视图中的启动按钮， 启动分析。

步骤2：查看"Analysis Result"中的分析结果， 点击"Leaked Activityes"中的具体实例， 该实例的引用关系将会展示在"Reference Tree"视图中。

步骤3：根据"Reference Tree"视图中的引用关系，查找leak的activity， 也就是谁Dominate这个activity对象。

可以看到是ListenerManager的静态单例sInstance最终支配了MemoryLeakActivity. sIntance连接到GC Roots， 故而导致MemoryLeakActivity GC Roots可达， 导致activity无法被回收。

<img src="/img/android_memory_leak_analysis/monitor_leakActivity.png" />

<img src="/img/android_memory_leak_analysis/monitor_hprof.png" />

**Heap Viewer查看内存消耗**

上述步骤，可以让我们快速定位可能的内存泄露。除了内存泄露， 还有内存消耗过大。我们可以在Heap Viewer中查看分析内存的消耗点， 如下：

<img src="/img/android_memory_leak_analysis/heap_viewer.png" />


#### 2) 使用MAT工具分析

相对与Android Studio的Memory Monitor， HPROF工具来说， MAT的使用显得更加生涩、难以理解些，但是MAT功能很全面。

Android Studio导出的hprof文件需要转换下才可以在MAT中使用，转换命令如下：

`$ hprof-conv com.anly.samples_2016.10.31_15.07.hprof mat.hprof`

**Histogram定位内存消耗**

<img src="/img/android_memory_leak_analysis/histogram_analysis.png" />

MAT中很多视图的第一行， 都可以输入正则， 来匹配我们关注的对象实例。

**Dominate Tree查看支配关系**

<img src="/img/android_memory_leak_analysis/dominate_tree_analysis.png" />

**使用OQL查询相关对象**

对于Android App开发来说， 大部分的内存问题都跟四大组件， 尤其是Activity相关， 故而我们会想查出所有Activity实例的内存占用情况， 可以使用OQL来查询：

<img src="/img/android_memory_leak_analysis/oql.png" />

**GC路径定位问题**

上面几个视图都可以让我们很快速的找到内存的消耗点，接下来我们要分析的就是为何这些个大对象没有被回收。对象没有被回收是因为他有到GC Roots的可达路径。那么我们就来分析下这条路径(Path to GC Roots)， 看看是谁在这条路中"搭桥"。

如下， 进入该对象的"path2gc"视图：

<img src="/img/android_memory_leak_analysis/path2gc_view.png" />

<img src="/img/android_memory_leak_analysis/path2gc_hprof.png" />

会发现与HPROF Analyzer异曲同工， 找出了是ListenerManager的静态实例导致了MemoryLeakActivity无法回收。

#### 3) 使用Leakcanary工具分析


**步骤1：加入LeakCanary**

app的build.gradle中加入:

	dependencies {  
	   debugCompile 'com.squareup.leakcanary:leakcanary-android:1.5'  
	   releaseCompile 'com.squareup.leakcanary:leakcanary-android-no-op:1.5'  
	   testCompile 'com.squareup.leakcanary:leakcanary-android-no-op:1.5'  
	}  

Application中加入:

	public class SampleApplication extends Application {  
	  
	    @Override  
	    public void onCreate() {  
	        super.onCreate();  
	  
	        LeakCanary.install(this);  
	    }  
	} 

**步骤2：操作要检测的界面， 查看结果**

当发生可疑内存泄露时， 会在桌面生成一个"Leaks"的图标， 点击进去可以看到内存泄露的疑点报告:

<img src="/img/android_memory_leak_analysis/leakcanary_result.png" />

可以看到内存泄漏的分析结果和之前两个工具结果一致。

内存问题的分析, 无外乎分析对象的内存占用(Retained Size), 找出Retained Size大的对象, 找到其直接支配(Immediate Dominator), 跟踪其GC可达路径(Path to GC Roots), 从而找到是谁让这个大对象活着，对于上面三种工具可以混合使用，一般情况下Android Studio自带的工具结合LeakCanary就能分析内存问题，MAT有更专业的一些功能，比如Heap比较等等值得探索。

