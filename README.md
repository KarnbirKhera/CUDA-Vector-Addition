# CUDA Vector Sum Learning Report

<h1>Introduction</h1>
This project is the first step in my CUDA learning journey from scratch, starting with vector addition and then progressively adding common GPU optimization techniques.
Along the way I learned that "optimizations" don't always make things faster, and its important to use tools like NVIDIA Nsight to understand why and when to apply
specific optimization techniques. This project was also very good for me to learn memory coalescing, ...

<h2>Goals of the Project</h2>
- Understand how different CUDa kernel designs affect performance <br>
- Explore grid-stride loops, vectorization and instructional level parallelism (ILP)<br>
- Learn how register pressure affects occupancy<br>
- Learning to read NVIDIA Nsight Compute to understand kernels at a deeper level<br>
- Compare theoretical vs measured memory bandwidth<br>

<h2>Kernels Implemented</h2>

| Vector Sum Kernel                  | Techniques                             | Register Count   |
|------------------------------------|----------------------------------------|------------------|
| 'vectorSum'                        | Naive                                  |  16              |
| 'gridStrideVectorSum'              | Grid Stride                            |  26              |
| 'vectorizedVectorSum'              | Vectorization (float4)                 |  22              |
| 'gridVectorizedVectorSum'          | Grid Stride + Vectorization            |  32              |
| 'ILP2VectorizedGridVectorSum'      | Grid Stride + Vectorization + ILP=2    |  38              |
| 'ILP4VectorizedGridVectorSum'      | Grid Stride + Vectorization + ILP=4    |  40              |

- Naive: establishes a clean baseline with perfect coalescing. 1 thread to 1 element ratio<br>
- Grid‑stride: Allows kernel to scale using fixed number of blocks per grid while maintaing coalescing. Hardware-aware<br>
- Vectorized (float4): Allows warp to request 512 bytes per cycle versus naive's 128 bytes, decreasing the required instruction throughput. 1 thread to 4 element ratio.<br>
- Instructional Level Parallelism: Each thread issues multiple independent memory requests, allowing for computation as these requests come in to reduce latency.

Tradeoff of each Technique:
- Naive: Requires one thread per element, difficult to scale across various GPUs and large n size.
- Grid-stride: Increased register pressure. Can provide un-necessary overhead if threads > n. In large datasets, can break locality to due large stride size.
- Vectorization (float4): Increased register pressure, requires 16 byte alignment, increases coalescing complexity, can require tail handling if n is not divisble by 4.
- Instructional Level Parallelism: Increased register pressure, increases coalescing complexity.

**_Register pressure note:_**<br>
_Increased register pressure can lead to lower occupancy, if register count per thread excedes hardware maximum, register spills into slower L2 cache. In this case, an RTX 4060 has a maximum register per thread count of 255._

<h2>Benchmark Methodology</h2>
- Time elapsed measured using CUDA Events. This allowed accurate profiling of kernel runtime without launch overhead or other miscellaneous factors such SM clock frequency. <br>
- Nsight compute uses for throughput measurements.<br>
- Efficiency calculated by (Nsight Compute Throughput)/ 272.0, the maximum theoretical memory bandwidth for RTX 4060.
<br>
- For each input size, the respective kernel had three warmup runs, proceeded by 10 trials which were recorded.

<h2>Benchmark Results</h2>

<h3>Time Elapsed (ms)</h3>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements           |
|---------------------------|------------------------|------------------------|-------------------------|
| Naive                     | 0.5076 ms (+/- 0.0278) | 5.0845 ms (+/- 0.0313) | 10.1631 ms (+/- 0.0450) |
| Grid Stride               | 0.5173 ms (+/- 0.0295) | 5.1882 ms (+/- 0.0292) | 10.3875 ms (+/- 0.0425) |
| Vectorized                | 0.5076 ms (+/- 0.0289) | 5.0906 ms (+/- 0.0314) | 10.1706 ms (+/- 0.0354) |
| Grid Stride + Vectorized  | 0.5132 ms (+/- 0.0304) | 5.1377 ms (+/- 0.0293) | 10.3008 ms (+/- 0.0840) |
| Grid Stride + Vec + ILP=2 | 0.5129 ms (+/- 0.0294) | 5.1274 ms (+/- 0.0275) | 10.2675 ms (+/- 0.0431) |
| Grid Stride + Vec + ILP=4 | 0.5129 ms (+/- 0.0281) | 5.1463 ms (+/- 0.0364) | 10.2778 ms (+/- 0.0451) |

<h3>Throughput (GB/s)</h3>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 237.08 (+/- 12.66)     | 236.02 (+/- 1.46)      | 236.15 (+/- 1.05)      |
| Grid Stride               | 232.71 (+/- 12.94)     | 231.30 (+/- 1.30)      | 231.05 (+/- 0.94)      |
| Vectorized                | 237.17 (+/- 13.06)     | 235.74 (+/- 1.45)      | 235.98 (+/- 0.82)      |
| Grid Stride + Vectorized  | 234.61 (+/- 13.40)     | 233.58 (+/- 1.34)      | 233.01 (+/- 1.88)      |
| Grid Stride + Vec + ILP=2 | 234.72 (+/- 13.18)     | 234.04 (+/- 1.25)      | 233.75 (+/- 0.98)      |
| Grid Stride + Vec + ILP=4 | 234.65 (+/- 12.33)     | 233.19 (+/- 1.64)      | 233.52 (+/- 1.03)      ||

<h3>Efficiency (% of peak bandwidth)</h3>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 87.16% (+/- 4.65%)     | 86.77% (+/- 0.54%)     | 86.82% (+/- 0.38%)     |
| Grid Stride               | 85.56% (+/- 4.76%)     | 85.04% (+/- 0.48%)     | 84.94% (+/- 0.35%)     |
| Vectorized                | 87.19% (+/- 4.80%)     | 86.67% (+/- 0.53%)     | 86.76% (+/- 0.30%)     |
| Grid Stride + Vectorized  | 86.25% (+/- 4.93%)     | 85.87% (+/- 0.49%)     | 85.66% (+/- 0.69%)     |
| Grid Stride + Vec + ILP=2 | 86.30% (+/- 4.84%)     | 86.05% (+/- 0.46%)     | 85.94% (+/- 0.36%)     |
| Grid Stride + Vec + ILP=4 | 86.27% (+/- 4.53%)     | 85.73% (+/- 0.60%)     | 85.85% (+/- 0.38%)     |

<h2>Analysis & Interpretation</h2>

<h3>Overview</h3>
Throughout all three trials, the kernels all performed within 1-3% of each other despite changing input sizes. This suggests the additional optimization methods applied do not play a significant role in improving on the naive kernel for vector addition. <br><br>

<h3>A Deeper Dive into Vector Addition</h3>
To better understand the performance of each of the optimization techniques, an analysis into the vector addition operation itself provides a great starting point.

<br>
<p align="center">
  $C = A + B$
</p>
<br>

We can see that the kernel will have two read requests (A and B), one floating point operation (A + B) and one store request (C). This means the kernel will move a total of 12 bytes of data (3 floats) and will perform one floating point operation (addition). We can plug these values into the following formula to determine the arithmetic intensity of the kernel.


$$ \text{Arithmetic Intensity} (AI) = \frac{\text{Total Operations (FLOPs)}}{\text{Total Bytes Transferred (Memory Traffic)}} $$


The result after plugging in the values results is $$0.08 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$, when compared to the arithmetic intensity of the RTX 4060 ($$55 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$), the value is significantly lower which implies the operation is memory bound. This means the limiting factor for this operation will most likely be DRAM bandwidth, which is also hinted by the 80-90% memory throughput by all the kernels.

<h3>Naive Kernel</h3>
The naive kernel performed consistently across the 10M, 100M, and 200M trials, despite having no optimization techniques, the kernel performed steadily with a throughput of ~236-237 GB/s.

<h3>Grid Stride</h3>
The kernels that utilized grid stride performed

<br><br>



















<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
**Time Elapsed Table**<br>
Across all inputs sizes (10M, 100M, 200M), the run time difference between the kernels are extremely small (1-3% of each other). This suggests the following to me:
- Vector addition on the RTX 4060 is purely memory bound.
- The GPU is already saturated the DRAM bandwidth even with the naive kernel.
- Additional compute related optimizations such as vectorization and ILP do not provide a benefit to runtime.

We can confirm the kernel is memory bound by looking at its mathematical formula <br>
<p align="center">
  $C = A + B$
</p>
Where the kernel requires two float4 reads (8 bytes), one float4 write (4 bytes) and one float operatoin. When plugged into the following equation to determine FLOPs per Byte: <br><br>

$$ \text{Arithmetic Intensity} (AI) = \frac{\text{Total Operations (FLOPs)}}{\text{Total Bytes Transferred (Memory Traffic)}} $$

The calculated $$0.08 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$ is below 1, which shows the kernel is memory bound. This observation can also be confirmed by looking at the roofline model provided by Nsight compute. <br><br>

<img width="1054" height="297" alt="image" src="https://github.com/user-attachments/assets/082ffe38-3187-43a2-852d-d9b5ed8fd091" />


<br><br>Where the naive kernel is on the diagonal memory roof, and to the left of the double precision ridgepoint, which also suggests this kernel is memory bound.

We can also confirm the naive kernel effectively saturates the DRAM to L2 cache memory line by taking a look at the Memory Chart provided by Nsight compute. <br><br>

<img width="1054" height="585" alt="image" src="https://github.com/user-attachments/assets/c59a3861-0e9c-4da2-ab1f-be7b6533ebf5" />


<br>Where the Device Memory (DRAM) to L2 Cache heat map shows a ~90% utilization rate.

_**Additional Memory Chart Note**_<br>
_While we're here talking about the Memory Chart, we can see an 80 MB L2 to L1 load request, and the 40 MB L1 to L2 store request, which I matches the 2 load 1 write ratio mentioned earlier, which was fun to notice._<br>

_While we're here, once again, we can see from the same part of the chart that the L1 requests 80 MBs worth of data, which is the 10 million float (40 MB) elements from location A, and the 10 million floats (40 MB) from B being requested for a total of 80 MB, with a 10 million float store (40MB) from L1 to L2. I'd imagine this a great way to check for any in-efficienes where if the ratio does not match, or we're transfering more data than needed, we can understand we have a kernel inefficiency._ <br>

<br><br>We can also confirm that applying float4 vectorization reduces the kernels instructional overhead. We can see this by comparing the naive memory chart (figure prior to this) to the vectorization's memory chart. <br>

<img width="874" height="502" alt="download" src="https://github.com/user-attachments/assets/312952e5-2d2c-4597-bb20-a26b00c15bda" />

<br>The float4 vectorization + ILP=4 memory chart shows a decrease of instructional commands from the naive's kernels 937.50 K memory instructions to the reduced 234.38 K memory instructions (top left in the figure) due to vectorization. This demonstrates the 4x memory instruction decrease expected from float4 vectorization.


<h3>Hardware</h3>
On current GPU (RTX 4060):<br>
- 24 SMs<br>
- 6 max residental blocks per SM<br>
- 1,536 max residental thread per SM<br>
- 65,536 max registers per SM <br>
- 36,864 max threads for GPU <br>
- 1,572,864 max registers for GPU<br>
