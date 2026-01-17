# CUDA Vector Sum Learning Report

Teaching myself CUDA through vector addition by writing my first kernel to understanding register pressure, SM occupancy, memory coalescing, and using Nsight Compute to figure out why my "optimizations" made things slower.

<h2>Overview</h2>

This project implements five progressively "optimized" CUDA kernels for vector addition, benchmarks them, and analyzes why the naive implementation won in this case.


<h2>Kernels Implemented</h2>

| Vector Sum Kernel                  | Techniques                             | Register Count
|------------------------------------|----------------------------------------|-------------------------
| 'vectorSum'                        | Naive (1 thread : 1 element)           |
| 'gridStrideVectorSum'              | Grid Stride                            |
| 'vectorizedVectorSum'              | Vectorization (float4)                 |
| 'gridVectorizedVectorSum'          | Grid Stride + Vectorization            |
| 'ILPVectorizedGridVectorSum'       | Grid Stride + Vectorization + ILP=2    |
|                                    |                                        |


<h2>Benchmark Results</h2>
<h3>Hardware</h3>
On current GPU (RTX 4060):
- 24 SMs
- 6 max residental blocks per SM \n
- 1,536 max residental thread per SM --> overflow will cause waves, (needing to queue blocks, which is not necessarily a bad thing, but queueing blocks has overhead) \n
- 65,536 max registers per SM --> overflow causes register spilling into L1 cache (slower) \n
- 36,864 max threads for GPU \n
- 1,572,864 max registers for GPU\n
