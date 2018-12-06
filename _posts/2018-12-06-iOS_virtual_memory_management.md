---
layout: post
title: iOS虚拟内存管理
date: 2018-12-06 
categories: iOS
author: 张永超
--- 

<img src="/img/iOS_virtual_memory_management/memory_use.png" />

## 虚拟内存概述


虚拟内存是一种允许操作系统避开设备的物理RAM限制的内存管理机制。虚拟内存管理器为每个进程创建一个逻辑地址空间或者虚拟内存地址空间，并且将它分配为相同大小的内存块，可称为页。处理器与内存管理单元MMU维持一个页表来映射程序逻辑地址空间到计算机RAM的硬件地址。当程序的代码访问内存中的一个地址时，MMU利用页表将指定的逻辑地址转换为真实的硬件内存地址，这种转换自动发生并且对于运行的应用是透明的。

就程序而言，在它逻辑地址空间的地址永远可用。然而，当应用访问一个当前并没有在物理RAM中的内存页的地址时，就会发生页错误。当这种情况发生时，虚拟内存系统调用一个专用的页错误处理器来立即响应错误。页错误处理器停止当前执行的代码，定位到物理内存的一个空闲页，从磁盘加载包含必要数据的页，同时更新页表，之后返回对程序代码的控制，程序代码就可以正常访问内存地址了，这个过程被称为分页。

如果在物理内存中没有空闲页，页错误处理器必须首先释放一个已经存在的页从而为新页提供空间，如何释放页由系统平台决定系统。在OS X，虚拟内存系统常常将页写入备份存储，备份存储是一个基于磁盘的仓库，包含了给定进程内存页的拷贝。将数据从物理内存移到备份存储被称为页面换出；将数据从备份存储移到物理内存被称为页面换入。在iOS，没有备份存储，所以页永远不会换出到磁盘，但是只读页仍可以根据需要从磁盘换入。

在OS X 和iOS中，页大小为4kb。因此，每次页错误发生时，系统会从磁盘读取4kb。当系统花费过度的时间处理页错误并且读写页，而并不是执行代码时，会发生磁盘震荡（disk thrashing）。

无论页换出／换入，磁盘震荡会降低性能。因为它强迫系统花费大量时间进行磁盘的读写。从备份存储读取页花费相当长的时间，       并且比直接从RAM读取要慢很多。如果系统从磁盘读取另一个页之前，不得不将一个页写入磁盘时，性能影响会更糟。

## 虚拟内存的限制

在iOS开发的过程中，难免手动去申请内存，目前大多数的移动设备都是ARM64的设备，即使用的是64位寻址空间，而且在iOS上       通过malloc申请的内存只是虚拟内存，不是真正的物理内存，那么在iOS设备上为什么会出现申请了2-3G就会出现申请失败呢？

<img src="/img/iOS_virtual_memory_management/malloc.png" />

当申请分配一个超大的内存时，iOS系统会按照 nano_zone 和 scalable_zone 的设计理念进行内存的申请，申请原理如下：

<img src="/img/iOS_virtual_memory_management/szone_malloc_should_clear.png" />

- 小于1k的走 `tiny_malloc`
- 小于15k或者127k的走 `small_malloc` （视不同设备内存上限而不同）
- 剩下的走 `large_malloc`

由于我们分配的非常大，我们可以确定我们的逻辑是落入 large_malloc 中。需要特别注意的是： **large_malloc 分配内存的基本单位是一页大小，而对于其他的几种分配方式，则不是必须按照页大小进行分配**。

由于 `large_malloc` 这个函数本身并没有特殊需要注意的地方，我们直接关注其真正分配内存的地方，即 `allocate_pages` ，如下所示：

<img src="/img/iOS_virtual_memory_management/mach_vm_map_code.png" />

从上不难看出，如果分配失败，就是提示报错。而 mach_vm_map 则是整个内存的分配核心。

<img src="/img/iOS_virtual_memory_management/vm_map.png" />


概括来说， `vm_map` 代表就是一个进程运行时候涉及的虚拟内存， `pmap` 代表的就是和具体硬件架构相关的物理内存。（这里我们暂时先不考虑 `submap` 这种情况）。

`vm_map` 本身是进程（或者从Mach内核的角度看是task的地址分布图）。这个地址分布图维护着一个 双向列表 ，列表的每一项都是 `vm_entry_t` ，代表着虚拟地址上连续的一个范围。而 `pmap` 这个结构体代表了个硬件相关的内存转换：即利用 `pmap` 这个结构体来描述抽象的物理地址访问和使用。


## 进程（任务）的创建

对于在iOS上的进程创建和加载执行Mach-O过程，有必要进行一个简单的介绍，在类UNIX系统本质上是不会无缘无故创建出一个       进程的，基本上必须通过 fork 的形式来创建。无论是用户态调用 posix 相关的API还是别的API，最终落入内核是均是通过函
数 fork_create_child 来创建属于Mach内核的任务。实现如下：

<img src="/img/iOS_virtual_memory_management/fork_create_child.png" />

- 要注意的就是**Mach**内核里面没有进程的概念，只有任务，进程是属于BSD之上的抽象。它们之间的联系就是通过指针建立，
child_proc->task = child_task 。

fork 出来的进程像是一个空壳，需要利用这个进程壳去执行科执行文件编程有意义的程序进程。从XNU上看，可执行文件的类型有如下分类：

<img src="/img/iOS_virtual_memory_management/file_type.png" />

常用的通常是 Mach-o 文件：

<img src="/img/iOS_virtual_memory_management/mach_0.png" />

上面的代码基本上都是在对文件进行各种检查，然后分配一个预使用的进程壳，之后使用 load_machfile 加载真正的二进制文件。

<img src="/img/iOS_virtual_memory_management/load_machfile.png" />

- 利用 `pmap_create` 创建硬件相关的物理内存抽象。利用  `vmap_create` 创建虚拟内存的地址图。ARM64下的页是16k一个虚拟页对应一个物理页。


这里需要重点关注 vm_map_create 0 和 vm_compute_max_offset(result->is64bit) ，代表着当前任务分配的虚拟内存地址的上下限， vm_compute_max_offset 函数实现如下：

<img src="/img/iOS_virtual_memory_management/vm_compute_max_offset.png" />

pmap_max_offset 函数实现如下：

<img src="/img/iOS_virtual_memory_management/pmap_max_offset.png" />

这里的关键点代码是：

<img src="/img/iOS_virtual_memory_management/key_code.png" />

max_offset_ret 这个值就代表了我们任务对应的 vm_map_t 的最大地址范围，比如说这里是8.375GB。


## 虚拟内存分配的限制

之前提到了 `large_malloc` 会走入到最后的 `vm_map_enter` ，那么我们来看看 `vm_map_enter` 的实现：


<img src="/img/iOS_virtual_memory_management/vm_map_enter.png" />

- 注意点1：基本上就是检查页的权限等，iOS上不允许可写和可执行并存。 
- 剩下的就是作各种前置检查。

如果上述代码不够清晰明了，如下这段代码可以更加的简洁：


<img src="/img/iOS_virtual_memory_management/easy_code.png" />

- 整个这段代码的意思是，就是要我们要找个一个比我们这个 start 地址大的 vm_entry_t 。最终的目的是为了在两个已经存在 vm_entry_t 之间尝试插入一个能包含从 start 到 start + size 的新的 vm_entry_t 。
- 如果没找到的话，就尝试利用 vm_map_lookup_entry 找一个 preceding 我们地址的的 vm_entry_t 。

	当找到了一个满足 start 地址条件的 vm_entry_t 后，剩下就是要满足分配大小 size 的需求了。

	<img src="/img/iOS_virtual_memory_management/while.png" />

- 判断 start + size 是不是可以正好插入在 vm_entry_t 代表的地址范围的空隙内，如果一直遍历到最后的任务地址上限都找不到，那就说明不存在我们需求的连续的虚拟内存空间用于作分配了。



## 总结

除了本文说明的虚拟内存分配的连续性限制以外，虚拟内存作为堆内存分配的一种，在布局范围上也有限制。更多详细的信息可参考如下链接。

- [XNU](https://github.com/opensource-apple/xnu "XNU")
- [Memory management](https://developer.apple.com/library/archive/documentation/General/Conceptual/DevPedia-CocoaCore/MemoryManagement.html "Memory management")
- [iOS内存管理](https://www.jianshu.com/p/4f49c5c81021 "iOS内存管理")
- [理解 iOS 的内存管理](https://blog.devtang.com/2016/07/30/ios-memory-management/ "理解 iOS 的内存管理")