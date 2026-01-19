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

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 0.5064 ms (+/- 0.0232) | 5.0811 ms (+/- 0.0344) | 10.1844 ms (+/- 0.0495)| 
| Grid Stride               | 0.5169 ms (+/- 0.0272) | 5.1938 ms (+/- 0.0249) | 10.4098 ms (+/- 0.0542)|
| Vectorized                | 0.5062 ms (+/- 0.0218) | 5.1051 ms (+/- 0.0455) | 10.1796 ms (+/- 0.0416)|
| Grid Stride + Vectorized  | 0.5148 ms (+/- 0.0303) | 5.1525 ms (+/- 0.0355) | 10.3025 ms (+/- 0.0733)|
| Grid Stride + Vec + ILP=2 | 0.5144 ms (+/- 0.0257) | 5.1321 ms (+/- 0.0352) | 10.2989 ms (+/- 0.0526)|
| Grid Stride + Vec + ILP=4 | 0.5181 ms (+/- 0.0260) | 5.1429 ms (+/- 0.0300) | 10.3335 ms (+/- 0.0941)|

<h3>Throughput (GB/s)</h3>

| Technique                 | 10M Elements               | 100M Elements               | 200M Elements             |
|---------------------------|----------------------------|-----------------------------|---------------------------|
| Naive                     | 250.990 (+/- 1.936)        | 201.345 (+/- 4.095)         | 248.635 (+/- 0.283)       | 
| Grid Stride               | 223.361 (+/- 51.589)       | 243.254 (+/- 0.447)         | 241.734 (+/- 0.354)       |
| Vectorized                | 250.649 (+/- 1.834)        | 231.984 (+/- 22.051)        | 248.480 (+/- 0.354)       |
| Grid Stride + Vectorized  | 236.874 (+/- 36.313)       | 190.673 (+/- 10.590)        | 244.850 (+/- 0.463)       |
| Grid Stride + Vec + ILP=2 | 225.437 (+/- 45.420)       | 240.270 (+/- 15.513)        | 244.861 (+/- 0.444)       |
| Grid Stride + Vec + ILP=2 | 190.537 (+/- 56.401)       | 245.813 (+/- 0.505)         | 244.785 (+/- 0.443)       |

<h3>Efficiency (% of peak bandwidth)</h3>

| Technique                 | 10M Elements               | 100M Elements               | 200M Elements               |
|---------------------------|----------------------------|-----------------------------|-----------------------------|
| Naive                     |      92.27% (+/- 0.71%)    |      74.01% (+/- 1.51%)     |      91.41% (+/- 0.10%)     | 
| Grid Stride               |      82.14% (+/- 18.96%)   |      89.40% (+/- 0.16%)     |      88.84% (+/- 0.13%)     |
| Vectorized                |      92.17% (+/- 0.67%)    |      85.33% (+/- 8.11%)     |      91.35% (+/- 0.13%)     |
| Grid Stride + Vectorized  |      87.12% (+/- 13.35%)   |      70.12% (+/- 3.89%)     |      90.03% (+/- 0.17%)     |
| Grid Stride + Vec + ILP=2 |      82.91% (+/- 16.70%)   |      88.32% (+/- 5.70%)     |      90.03% (+/- 0.16%)     |
| Grid Stride + Vec + ILP=4 |      70.06% (+/- 20.74%)   |      90.41% (+/- 0.19%)     |      89.99% (+/- 0.16%)     |

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
