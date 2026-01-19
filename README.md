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
| Naive                     | 0.5554 ms (+/- 0.1128) | 5.1167 ms (+/- 0.1525) | 10.1299 ms (+/- 0.0416) |
| Grid Stride               | 0.5173 ms (+/- 0.0309) | 5.2019 ms (+/- 0.1493) | 10.3402 ms (+/- 0.0535) |
| Vectorized                | 0.5062 ms (+/- 0.0273) | 5.4275 ms (+/- 0.2707) | 10.1386 ms (+/- 0.0474) |
| Grid Stride + Vectorized  | 0.5131 ms (+/- 0.0332) | 5.5656 ms (+/- 0.1066) | 10.2593 ms (+/- 0.0793) |
| Grid Stride + Vec + ILP=2 | 0.5108 ms (+/- 0.0324) | 5.6078 ms (+/- 0.0986) | 10.2351 ms (+/- 0.0501) |
| Grid Stride + Vec + ILP=4 | 0.5143 ms (+/- 0.0399) | 5.6131 ms (+/- 0.0852) | 10.2347 ms (+/- 0.0450) |

<h3>Throughput (GB/s)</h3>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 223.31 (+/- 35.75)     | 234.72 (+/- 6.51)      | 236.93 (+/- 0.97)      |
| Grid Stride               | 232.73 (+/- 13.01)     | 230.86 (+/- 6.20)      | 232.11 (+/- 1.20)      |
| Vectorized                | 237.74 (+/- 12.38)     | 221.65 (+/- 11.02)     | 236.73 (+/- 1.11)      |
| Grid Stride + Vectorized  | 234.81 (+/- 14.44)     | 215.69 (+/- 4.13)      | 233.95 (+/- 1.80)      |
| Grid Stride + Vec + ILP=2 | 235.84 (+/- 14.39)     | 214.05 (+/- 3.80)      | 234.49 (+/- 1.15)      |
| Grid Stride + Vec + ILP=4 | 234.63 (+/- 16.74)     | 213.83 (+/- 3.28)      | 234.50 (+/- 1.03)      |

<h3>Efficiency (% of peak bandwidth)</h3>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 82.10% (+/- 13.14%)    | 86.29% (+/- 2.39%)     | 87.11% (+/- 0.36%)     |
| Grid Stride               | 85.56% (+/- 4.78%)     | 84.88% (+/- 2.28%)     | 85.33% (+/- 0.44%)     |
| Vectorized                | 87.40% (+/- 4.55%)     | 81.49% (+/- 4.05%)     | 87.03% (+/- 0.41%)     |
| Grid Stride + Vectorized  | 86.33% (+/- 5.31%)     | 79.30% (+/- 1.52%)     | 86.01% (+/- 0.66%)     |
| Grid Stride + Vec + ILP=2 | 86.70% (+/- 5.29%)     | 78.70% (+/- 1.40%)     | 86.21% (+/- 0.42%)     |
| Grid Stride + Vec + ILP=4 | 86.26% (+/- 6.16%)     | 78.62% (+/- 1.21%)     | 86.21% (+/- 0.38%)     |

<h2>Analysis & Interpretation</h2>

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

<img width="1106" height="573" alt="image" src="https://github.com/user-attachments/assets/50b38b74-c142-4dd5-a1dd-1e79d2eaef6a" />

<br>Where the Device Memory (DRAM) to L2 Cache heat map shows a ~90% utilization rate.

_**Additional Memory Chart Note**_<br>
_While we're here talking about the Memory Chart, we can see an 80 MB L2 to L1 load request, and the 40 MB L1 to L2 store request, which I matches the 2 load 1 write ratio mentioned earlier, which was fun to notice._<br>

_While we're here, once again, we can see from the same part of the chart that the L1 requests 80 MBs worth of data, which is the 10 million float (40 MB) elements from location A, and the 10 million floats (40 MB) from B being requested for a total of 80 MB, with a 10 million float store (40MB) from L1 to L2. I'd imagine this a great way to check for any in-efficienes where if the ratio does not match, or we're transfering more data than needed, we can understand we have a kernel inefficiency._ <br>

<br><br>We can also confirm that applying float4 vectorization reduces the kernels instructional overhead. We can see this by comparing the naive memory chart (figure prior to this) to the vectorization's memory chart. <br>

<img width="1093" height="562" alt="image" src="https://github.com/user-attachments/assets/151f4076-ccb4-4b7b-9a47-199156d40e89" />

<br>The float4 vectorization + ILP=4 memory chart shows a decrease of instructional commands from the naive's kernels 937.50 K memory instructions to the reduced 234.38 K memory instructions (top left in the figure) due to vectorization. This demonstrates the 4x memory instruction decrease expected from float4 vectorization.


<h3>Hardware</h3>
On current GPU (RTX 4060):<br>
- 24 SMs<br>
- 6 max residental blocks per SM<br>
- 1,536 max residental thread per SM<br>
- 65,536 max registers per SM <br>
- 36,864 max threads for GPU <br>
- 1,572,864 max registers for GPU<br>
