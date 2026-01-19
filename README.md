# CUDA Vector Sum Learning Report

Teaching myself CUDA through vector addition by writing my first kernel to understanding register pressure, SM occupancy, memory coalescing, and using Nsight Compute to figure out why my "optimizations" made things slower.

<h2>Overview</h2>

This project implements five progressively "optimized" CUDA kernels for vector addition, benchmarks them, and analyzes why the naive implementation won in this case.


<h2>Kernels Implemented</h2>

| Vector Sum Kernel                  | Techniques                             | Register Count
|------------------------------------|----------------------------------------|-------------------------
| 'vectorSum'                        | Naive (1 thread : 1 element)           |  16
| 'gridStrideVectorSum'              | Grid Stride                            |  26
| 'vectorizedVectorSum'              | Vectorization (float4)                 |  22
| 'gridVectorizedVectorSum'          | Grid Stride + Vectorization            |  32
| 'ILP2VectorizedGridVectorSum'      | Grid Stride + Vectorization + ILP=2    |  38
| 'ILP4VectorizedGridVectorSum'      | Grid Stride + Vectorization + ILP=4    |  40


<h2>Benchmark Results</h2>

<h3>Time Elapsed (measured using CUDA Events)</h3>

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
| Naive                     | 250.990 (+/- 1.936)   | 201.345 (+/- 4.095)    | 248.635 (+/- 0.283)  | 
| Grid Stride               | 223.361 (+/- 51.589)  | 243.254 (+/- 0.447)    | 241.734 (+/- 0.354)  |
| Vectorized                | 250.649 (+/- 1.834)   | 231.984 (+/- 22.051)   | 248.480 (+/- 0.354)  |
| Grid Stride + Vectorized  | 236.874 (+/- 36.313)  | 190.673 (+/- 10.590)   | 244.850 (+/- 0.463)  |
| Grid Stride + Vec + ILP=2 | 225.437 (+/- 45.420)  | 240.270 (+/- 15.513)   | 244.861 (+/- 0.444)  |
| Grid Stride + Vec + ILP=2 | 190.537 (+/- 56.401)  | 245.813 (+/- 0.505)    | 244.785 (+/- 0.443)  |

<h3>Efficiency (% of peak bandwidth)</h3>

| Technique                 | 10M Elements    | 100M Elements   | 200M Elements   |
|---------------------------|-----------------|-----------------|-----------------|
| Naive                     |<ins>87.5%</ins> |<ins>85.5%</ins> | <ins>86.7%</ins>| 
| Grid Stride               |      85.1%      |      71.8%      |      84.7%      |
| Vectorized                |      86.6%      |      73.3%      |      86.6%      |
| Grid Stride + Vectorized  |      80.2%      |      72.1%      |      85.8%      |
| Grid Stride + Vec + ILP=2 |      85.6%      |      71.6%      |      85.7%      |


<h3>Hardware</h3>
On current GPU (RTX 4060):<br>
- 24 SMs<br>
- 6 max residental blocks per SM<br>
- 1,536 max residental thread per SM<br>
- 65,536 max registers per SM <br>
- 36,864 max threads for GPU <br>
- 1,572,864 max registers for GPU<br>
