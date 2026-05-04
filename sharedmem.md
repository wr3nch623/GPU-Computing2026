## Shared memory

Global memory is the worse, cause it is slow. 

The blocks are conseptually linked in the shared memory, since they cooperate in the same stream multiprocessor. The registers are mapped into warps and threads. Caches are hardware managed, and we can do some stuff like change the behavuoir, the L1 size and not much else. Look at the fucking manual. 

Two components: off chips and on chip, we focus on the on chip at the level of the cache and shared memory. In Ampere we only have the L1 and shared memory, but from hopper we can cluster across the SM: we can move data from the shared memory. This is just a note, but is something that might be useful.

Roughly we have 200KB per SM to play with, if you saturate the shared memory you have 20TB/s as bandwidth. 

The shared memory is a software managed scratchpad, where you can allocate and do some work (you already have the L1 as cache). We use it to reduce global traffic (since global memory is not fast), maximise the data reuse (mimic the role of the cache). You manage the locality here, so this is needed to be learned. The concept is to offload the memory near the compute part to make everything faster. 

BEWARE: Some stuff you will find out at runtime, since nvidia has a JIT when execute to allocate memory shit will break when you 

With shared memory we need to make sure that is in a coherent state, that it will be copied into the global memory after computation. The shared memory has the same problem with the global memory: accessing pattern. Usually the best thing is to access it linearly. 

The sync barrier force that before incrementing the instruction in the PC you wait for all the others, so you know they can manage a race condition in the global that will be propagated in the shared memory.

If two thread access for the same bank they will be serialized so it will use multiple iteration creating delay, called bank If two thread access for the same bank they will be serialized so it will use multiple iteration creating delay, called bank conflict. Sometimes the compiler notices this and sends as broadcast so that it wont be an issue sometimes. This is a problem for naive prefix sum and generic matrix multiplication. Stride is the common solution for this type of model, if you have issues you can use coprime numbers as stride so that tipically they avoid collisions, also using padding can work, not great memory side but still a solution

From Ampere we can move data in async, so we can implement a pipeline which is called in different ways in architectures but it essentially does the same thing.


# Memory management part 3


Cuda is block level programming

An abstraction layer is provided for the OS level in CUDA runtime from the driver. When copying data the CPU initiate the operation with the MMU to copy data to device. CUDA Stream has some operations which may be on the system or GPU side, managed by the driver. 

We don't execute kernel, once the code is pushed it triggers the PC to increment, which are not blocking. 

To get the time from gpu first you synchronize then get time of day. We will use CUDA Events for that.

When a pointer from host tries to access the GPU the DMA needs to translate this address cause it works using physical addresses

Different approaches regarding the approaches of memory transfer, which performance is limited by the PCIe bus and data transfered, which also depends on other factors(maybe?).

## Pageable memory

The performance depends on the OS, also there is host to host copy if i remain on the same device. cudaMemory is blocking because of the translation, so we don't use the DMA

## Pinned or page-locked
We don't use OS virtual memory, triggering the transfer with the DMA, so the copy is asynchronous cause you can overlap data transfer and kernel exec keeping everything in use. To do this you use cudaMallocHost which is very expensive, and once is allocated you can mode everything to that address. 

## Unified memory







We can sync at the warp level. With warp-centric operation we look at shuffling, which is a communication pattern to mode data across threads.:w
