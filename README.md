# CUDA Vector Sum Learning Report

<h1>Introduction</h1>
This project is the first step in my CUDA learning journey from scratch, starting with vector addition and then progressively adding common GPU optimization techniques.
Along the way I learned that "optimizations" don't always make things faster, and its important to use tools like NVIDIA Nsight to understand why and when to apply
specific optimization techniques. This project was also very good for me to learn memory coalescing, ...

<h1>Goals of the Project</h1>
- Understand how different CUDA kernel designs affect performance <br>
- Explore grid-stride loops, vectorization and instructional level parallelism (ILP)<br>
- Learn how register pressure affects occupancy<br>
- Learning to read NVIDIA Nsight Compute to understand kernels at a deeper level<br>
- Compare theoretical vs measured memory bandwidth<br>

<h1>Kernels Implemented</h1>

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
- Vectorized (float4): Allows warp to request 512 bytes per cycle versus naive's 128 bytes, decreasing the required memory instruction to DRAM. 1 thread to 4 element ratio.<br>
- Instructional Level Parallelism: Each thread issues multiple independent memory requests, allowing for computation as these requests come in to reduce latency.

Tradeoff of each Technique:
- Naive: Requires one thread per element, difficult to scale across various GPUs and large n size.
- Grid-stride: Increased register pressure. Can provide un-necessary overhead if threads > n. In large datasets, can break locality to due large stride size.
- Vectorization (float4): Increased register pressure, requires 16 byte alignment, increases coalescing complexity, can require tail handling if n is not divisble by 4.
- Instructional Level Parallelism: Increased register pressure, increases coalescing complexity.

**_Register pressure note:_**<br>
_Increased register pressure can lead to lower occupancy, if register count per thread excedes hardware maximum, register spills into slower L2 cache. In this case, an RTX 4060 has a maximum register per thread count of 255._

<h1>Benchmark Methodology</h1>
- Time elapsed measured using CUDA Events. This allowed accurate profiling of kernel runtime without launch overhead or other miscellaneous factors such SM clock frequency. <br>
- Nsight compute uses for throughput measurements.<br>
- Efficiency calculated by (Nsight Compute Throughput)/ 272.0, the maximum theoretical memory bandwidth for RTX 4060.
<br>
- For each input size, the respective kernel had three warmup runs, proceeded by 10 trials which were recorded.

<h1>Benchmark Results</h1>

<h2>Time Elapsed (ms)</h2>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements           |
|---------------------------|------------------------|------------------------|-------------------------|
| Naive                     | 0.5076 ms (+/- 0.0278) | 5.0845 ms (+/- 0.0313) | 10.1631 ms (+/- 0.0450) |
| Grid Stride               | 0.5173 ms (+/- 0.0295) | 5.1882 ms (+/- 0.0292) | 10.3875 ms (+/- 0.0425) |
| Vectorized                | 0.5076 ms (+/- 0.0289) | 5.0906 ms (+/- 0.0314) | 10.1706 ms (+/- 0.0354) |
| Grid Stride + Vectorized  | 0.5132 ms (+/- 0.0304) | 5.1377 ms (+/- 0.0293) | 10.3008 ms (+/- 0.0840) |
| Grid Stride + Vec + ILP=2 | 0.5129 ms (+/- 0.0294) | 5.1274 ms (+/- 0.0275) | 10.2675 ms (+/- 0.0431) |
| Grid Stride + Vec + ILP=4 | 0.5129 ms (+/- 0.0281) | 5.1463 ms (+/- 0.0364) | 10.2778 ms (+/- 0.0451) |

<h2>Throughput (GB/s)</h2>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 237.08 (+/- 12.66)     | 236.02 (+/- 1.46)      | 236.15 (+/- 1.05)      |
| Grid Stride               | 232.71 (+/- 12.94)     | 231.30 (+/- 1.30)      | 231.05 (+/- 0.94)      |
| Vectorized                | 237.17 (+/- 13.06)     | 235.74 (+/- 1.45)      | 235.98 (+/- 0.82)      |
| Grid Stride + Vectorized  | 234.61 (+/- 13.40)     | 233.58 (+/- 1.34)      | 233.01 (+/- 1.88)      |
| Grid Stride + Vec + ILP=2 | 234.72 (+/- 13.18)     | 234.04 (+/- 1.25)      | 233.75 (+/- 0.98)      |
| Grid Stride + Vec + ILP=4 | 234.65 (+/- 12.33)     | 233.19 (+/- 1.64)      | 233.52 (+/- 1.03)      ||

<h2>Efficiency (% of peak bandwidth)</h2>

| Technique                 | 10M Elements           | 100M Elements          | 200M Elements          |
|---------------------------|------------------------|------------------------|------------------------|
| Naive                     | 87.16% (+/- 4.65%)     | 86.77% (+/- 0.54%)     | 86.82% (+/- 0.38%)     |
| Grid Stride               | 85.56% (+/- 4.76%)     | 85.04% (+/- 0.48%)     | 84.94% (+/- 0.35%)     |
| Vectorized                | 87.19% (+/- 4.80%)     | 86.67% (+/- 0.53%)     | 86.76% (+/- 0.30%)     |
| Grid Stride + Vectorized  | 86.25% (+/- 4.93%)     | 85.87% (+/- 0.49%)     | 85.66% (+/- 0.69%)     |
| Grid Stride + Vec + ILP=2 | 86.30% (+/- 4.84%)     | 86.05% (+/- 0.46%)     | 85.94% (+/- 0.36%)     |
| Grid Stride + Vec + ILP=4 | 86.27% (+/- 4.53%)     | 85.73% (+/- 0.60%)     | 85.85% (+/- 0.38%)     |

<h1>Analysis & Interpretation</h1>

<h2>Overview</h2>
Throughout all three trials, the kernels all performed within 1-3% of each other despite changing input sizes. This suggests the additional optimization methods applied do not play a significant role in improving on the naive kernel for vector addition. 
<br><br>
To keep analysis focused and to avoid noise from measurement variance, all kernel specific interpetations will focus on the 200M element trials which showed the lowest standard deviation.
<br><br>

<h2>A Deeper Dive into Vector Addition</h2>
To better understand the performance of each of the optimization techniques, an analysis into the vector addition operation itself provides a great starting point.

<br>
<p align="center">
  $C = A + B$
</p>
<br>

We can see that the kernel will have two read requests (A and B), one floating point operation (A + B) and one store request (C). This means the kernel will move a total of 12 bytes of data (3 floats) and will perform one floating point operation (addition). We can plug these values into the following formula to determine the arithmetic intensity of the kernel.


$$ \text{Arithmetic Intensity} (AI) = \frac{\text{Total Operations (FLOPs)}}{\text{Total Bytes Transferred (Memory Traffic)}} $$


The result after plugging in the values results is $$0.08 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$, when compared to the arithmetic intensity of the RTX 4060 ($$55 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$), the value is significantly lower which implies the operation is memory bound. This along with the consistent 80% to 90% DRAM bandwidth usage across all techniques and n sizes suggests this kernel is memory bound.

<h2>Naive Kernel</h2>
The naive kernel set a strong baseline in the trial with coalesced memory access and minimal instructional overhead (only one if statement). Each thread loads 2 elements, does one floating point operation, and stores 1 element. <br><br>

Because vector addition operation has a very low arithmetic intensity, and the naive kernel has a low register count of 16 which allows for more parallelism, the naive kernel already saturates the DRAM bandwidth, achieving a memory throughput of ~236 GB/s which is ~87% of the peak bandwidth without any optimizations. Any optimization to this kernel would have to reduce memory traffic through techniques like lower precision, caching or kernel fusing, the latter two not being possible with a simple kernel like this one.

<h2>Grid Stride</h2>
Grid stride detaches the 1 thread to 1 element ratio found in the naive kernel, allowing it to run on any GPU and scale to any n size. In this case, grid stride provided an unnecessary instructional overhead as well as increased register pressure. For large n sizes, grid stride also reduced locality when paired with ILP where the large strides required a single thread
to call for multiple sectors for away from one another. <br><br>

Because the orginial kernel was memory bound, the extra instructions and register pressure simply added overhead resulting in a 1-3% decrease in memory throughput. Grid stride is still a very effective technique that allows the kernel to scale, but not effective in raw performance for a memory bound kernel.

<h2>Vectorization</h2>
Vectorization (float4) reduces the memory instructions to the DRAM by having a single thread call for 4 floats at once, rather in the naive where a thread calls for a single float utilizing only 4 bytes out of 32 bytes provided by the request. Float4 vectorization allows perfect utilization of this minimum 32 byte memory transaction from the DRAMs 32 byte sectors.<br><br>

Although vectorization is more efficient when it comes to memory instructions, it does not tackle the memory bound nature of the kernel. The kernel does perform at a similar level as the naive likely because the only additional instructional overhead is in the tail end of the kernel.

<h2>Instructional Level Parallelism</h2>
Instructional Level Parallelism allows for latency hiding by issuing multiple independent memory transactions at the same time. This allows the arithmetic operations to take place while the memory from the DRAM is already traveling to the thread for the next arithmetic operation.<br><br>

While ILP provides latency hiding, it does not tackle the memory bound nature of the kernel. The kernel performs slightly worse than the naive likely due to the increased instructional overhead.<br><br>

<h1>Nsight Compute Findings</h1>

While CUDA timing shows us the runtime of each kernel, Nsight Compute allows us to dig deeper into the more interesting and fundamental parts of our kernel. Getting to explore the roofline model, memory charts, occupancy, warp stall reasons (which has been very helpful) and cache acitivty has been incredbily helpful for understanding how the GPU responds to each of the optimization techniques. These insights helped me connect the benchmark results to the underlying hardware behavior which I will break down in this section for each optimization in the 200M element trials as it had the lowest standard deviation.
<br><br>

<h2>Naive</h2>
<h3>Memory Throughput</h3>
<img width="876" height="178" alt="image" src="https://github.com/user-attachments/assets/4e3e5646-8ae2-41a0-8c1b-af8822f83e96" />

The naive kernel achieves 93.93% DRAM throughput, showing that the naive kernel effectively saturates the DRAM bandwidth without any optimization techniques.
<h3>Roofline Placement</h3>
<img width="1710" height="348" alt="image" src="https://github.com/user-attachments/assets/b48c384b-bc11-4cff-9d6e-c929c5647132" />

The naive kernel has an arithmetic intensity of 0.08 FLOPs/byte according to the roofline model, and sits on the memory roof to the left of the double precision ridge point, which confirms the kernel is memory bound. This matches our early calculation of 12 bytes moved per one floating point operation resulting in 0.08 FLOPs/byte.

<h3>Warp Stall Reasons</h3>
<img width="1714" height="83" alt="image" src="https://github.com/user-attachments/assets/de7453c1-625d-402f-8211-7a49978b1d23" />

Long Scoreboard Stalls represent 96% of the total 179.5 cycles being stalled in this kernel. Long Scoreboard Stalls tell us there is a memory dependency chain which requires the kernel to repeatedly wait for the memory to arrive before it can continue. Upon looking into the Source tab of Nsight compute, the line C[i] = A[i] + B[i] has a memory dependency where we must wait for values A[i] and B[i] to arrive before we can calculate C[i] resulting in our Long Scoreboard Stall. While I am still learning to read SASS, it was interesting to actually seeing the FADD (FP32 add) instruction was depending on the load instructions before it of A[i] and B[i] before calculating.

<h3>Achieved Occupancy</h3>
The naive kernel has an Achieved Occupancy of 84.00%, with an Achieved Active Warps per SM of 40.32, where in comparison 48 active warps per SM would result in 100% achieved occupancy. This high occupancy percentage is due to the fact the naive kernel only requires 16 registers, meaning we will hit the 1,536 max threads per SM before we hit the 65,536 max registers per SM count. Assuming an Achieved Active Warps per SM of 40.32, this means per SM we use a total of ~1,290 threads out of the maximum 1,536 threads. This makes sense because if we were to add another block of 256 threads (determined at kernel launch), we would exceed the maxmimum thread limit count per SM hence the kernel only uses 5 blocks (84% of occupancy) vs 6 blocks (100% of occupancy).
<br><br>
This points out a great distinction between theoretical occupancy and the actual achieved occupancy. Theoretical occupancy assumes with a block size of 256 threads, each SM can actively operate with 6 blocks each, reaching the 1,536 maximum threads per SM limit (100% occupancy), in reality, each SM handled 5 blocks or 1,280 threads. This is likely because of some small implicit overhead such as from launching the blocks themselves, or switching between blocks, that prevents the last jump from 5 blocks (1,280 threads) to 6 blocks (1,536 threads) per SM. If I were to assume, a smaller thread per block size would result in higher occupancy, but doubling the amount of blocks required would likely increase the block lanuch overhead resulting in either little to no gain.

<h3>Cache Behavior</h3>
<img width="407" height="598" alt="image" src="https://github.com/user-attachments/assets/a1fe5f9b-8f91-42e5-8cee-7a5720233b57" /><br>
The naive kernel has an L1 cache hit rate of 0% which is expected as vector add fundamentally does not reuse data. Although, the L2 cache does have a hit rate of 31.53% which is very unexpected. This is interesting because say in the naive kernel, a single thread requests 4 bytes of data, at a warp level memory call, that is 128 bytes. This fits perfectly into the 128 byte cache line of the RTX 4060, and also perfectly accesses four 32 byte sectors in the DRAM, so at the moment this is a mystery to me.

After looking into why this might be the case, the following provides some insight to this mysterious 31% L2 cache hit rate. To better understand why this might be the case, I isolated the vector add kernel added a write only variant, and a read only variant with the following results. <br>

<h4>The Orginal Naive</h4>
<img width="463" height="495" alt="image" src="https://github.com/user-attachments/assets/33f3b2d1-8977-4b21-b5eb-369e0e3febc1" />
<img width="1239" height="444" alt="image" src="https://github.com/user-attachments/assets/6b4278a0-255c-47f7-ae0d-2574383160c1" />
<br>
We can see from the Nsight compute, the L2 cache hit rate is ~31%, which is supported by the detailed L2 report. This percentage is made up of of 54186884 reads (A[i] and B[i]) and 25001191 writes (C[i]).


<h4>The Read-only Naive</h4>
<img width="455" height="508" alt="image" src="https://github.com/user-attachments/assets/1c569573-727e-4e51-bea2-29f6adba7ab4" />
<img width="1355" height="451" alt="image" src="https://github.com/user-attachments/assets/8f518125-1821-40bc-bf02-2917b3484e1a" />
<br>
We can see from the Nsight compute, the L2 cache hit rate is ~0.07%, which is supported by the detailed L2 report.  This percentage is made up of 52477053 reads (A[i] and B[i]) and the 601 writes can be assumed to be some type of write overhead.

<h4>The Write-only Naive</h4>
<img width="451" height="515" alt="image" src="https://github.com/user-attachments/assets/564b7637-e51f-4ec8-841c-703378b9e2ba" />
<img width="1327" height="445" alt="image" src="https://github.com/user-attachments/assets/dc05e6a7-6fc3-4cc9-a36a-f4a7ff4ef0f7" />

<br>
We can see fromt he Nsight compute, the L2 cache hit rate is ~96%, which supported by the detailed L2 report. This percentage is made up of 1,538,775 reads and 25,000,362 writes (C[i]). The 1,538,775 reads is very interesting as the code itself has no A[i] and B[i] reads. After further analysis:<br>

<img width="1413" height="188" alt="image" src="https://github.com/user-attachments/assets/ed3db8eb-4af4-4984-b1cd-bc7dd2b9ccb1" />

<br>
From the following SASS panel for the naive kernel we see a "Store to Global Memory" STG.E instruction. From my current understanding on the RTX 4060 hardware architecture the way the STG.E instruction works.<br><br>

- When writing to DRAM, if cache HIT: A single write instruction
- When writing to DRAM, if cache MISS: A single read instruction, followed later by a write instruction

For this kernel we have a single 4 byte float write per thread, on a warp scale this results in a perfect 128 byte write to the 128 byte cache line. This means for every new warp write, there is a very high likelyhood the required four 32 byte sectors in the DRAM are not in the L2 cache. This means every warp write will result in a cache line miss, because of this the kernel is required to do a DRAM read of the four 32 byte sectors in the DRAM before writing. This is supported by following images where the L2 lts__t_sectors_op_read.sum is 1,538,775, and in the Nsight compute kernel we have a total sector misses to device of 1,508,103, this eludes these L2 cache misses result in a signfiicant amount of reads made by the kernel. This GPU behavior of cache miss resulting in a read is supported by the following NVIDIA post https://forums.developer.nvidia.com/t/how-do-gpus-handle-writes/58914/5.

After looking into this even more, I found that this read write policy is very intentional and is actually fundamental to the hardware/software co-design of not just the RTX 4060, but for all modern GPUs. The reason for this read write policy is very interesting because it solves a problem that before this I had never even thought of.

Say our kernel was able to write directly into the DRAM, bypassing the L2 cache. At first I thought this was a great idea, we'd avoid the L2 cache entirely and I assumed it would result in faster performance. The problem with this is the DRAM can only load a single cache line into its buffer at a time. To write to the DRAM, you first have to open the cache line, this means using the DRAM's 2-8kb cache to load the entire row, this is a ~50 nanosecond action, then writing to the row which is often ~20 nanosecond action. At first this may seem trivial, but from the perspective of parallel programming this can prove very in-efficient.

Say we have the following scenario where both SM One and SM Two are writing to the DRAM in parallel.

- SM One: Opens cache line A and it is loaded into the DRAM's single row buffer (50 nanoseconds) and writes a single 16 byte value (20 nanoseconds)
- SM Two: Opens cache line B and it is loaded into the DRAM's single row buffer (50 nanoseconds) and writes a single 16 byte value (20 nanoseconds)
- SM One: Has to again open cache line A into the DRAM's single row buffer (50 nanoseconds) and writes a 16 byte value (20 nanoseconds)
- SM Two: Has to again open cache line B into the DRAM's single row buffer (50 nanoseconds) and writes a 16 byte value (20 nanoseconds)

The in-efficiency lies where the DRAM's single row buffer does not allow the full strength of parallel programming to shine, and this is where the L2 cache comes in. The way the L2 cache works is the following:

- SM One: Writes to L2 cache with a single value, targeted at memory sector A in the DRAM
- SM Two: Writes to L2 cache with a single value, targeted at memory sector B in the DRAM
- SM One: Writes to L2 cache with a single value, targeted at memory sector A in the DRAM
- SM Two: Writes to L2 cache with a single value, targeted at memory sector B in the DRAM

- The L2 cache: Opens a cache line to memory sector A (50 nanoseconds) and batch writes two 16 byte values (~20 nanoseconds)
- The L2 cache: Opens a cache line to memory sector B (50 nanoseconds) and batch writes two 16 byte values (~20 nanoseconds)

Rather than needing to re-open cache lines, the L2 cache gathers all the needed values to be written for each memory sector, then does a single read followed by a batch write. This allows turns an in-efficient 280 nanosecond operation into an efficient 140 nanosecond operation.

Now going back to the STG.E, it turns out after more research the .E is actually a modifier for the STG command when it comes to the caching policy. The .E represents that this data will follow a normal replacement policy, meaning it is neither elevated to be removed first, or elevated to be kept. This adds an interesting layer of granularity because while we cannot change the fundamental nature of the read write structure, we can although modify the L2 cache policy of the data we send.

Looking at vector add kernel, we know that the entire process itself is purely streamed data, meaning data is loaded once, and never used or needed again. We can actually modify this cache policy with a ".cs" modifier which hints to the PTX to SASS compiler that the data is not needed once used, allowing it to be evicted first, which should in theory in-directly allow for more data to flow through the L2 cache. The reason this would theoretically allow more data to flow is because the default cache policy evicts the oldest or least recently used, but using the ".cs" should save the compiler time and compute as it does not need to find/calculate the oldest cache line. 

However, I would imagine this comes at a cost of having to "tag" or use meta-data to convey to the compiler that this specific cache line can be evicted first. I would once again imagine, when n is low the cost of adding this "tag" would likely overweigh its need, whereas if we have a large n size, the "tag" overhead becomes minimial.

There is also another cache policy modifier known as .lu (Last Use). This means once the cache line is utilized, the cache line is disposed of even if the L2 cache is not full. This likely has the same n size use case mentioned for .cs (Cached Streaming).


All three kernel types, naive, naive cache streamed and naive last use performed within 1-2% of each other on n sizes 50M, 100M and 200M. One thing to note is the difference in the PM Sampling timeline for the STG.E (naive) and the STG.E.EF (cached streaming) policies.

STG.E (naive)
<img width="1673" height="231" alt="image" src="https://github.com/user-attachments/assets/1c0bfa3c-9613-4fd4-b580-781c510e4dc4" />

STG.E.EF (cached streaming)
<img width="1653" height="222" alt="image" src="https://github.com/user-attachments/assets/c0010cea-f1c1-4ed5-8814-f237ce2d2445" />

The naive STG.E kernel produced a very consistent DRAM and L2 cache throughput. This is because the default caching policy (STG.E) marks cache lines that are either the oldest or least used for eviction when a new cache line is needed. This naturally populates the write back queue to the DRAM slowly, allowing the write back queue to drain as needed between reads, keeping the queue at a steady level.

The naive cached stream kernel produced a very "bursty" DRAM and L2 cache throughput. This is because when using cached stream policy (STG.E.EF), as soon as a cache line is dirty (data is modified), the cache line is marked to be evicted. This means when a new cache line is needed, the write back queue to the DRAM fills very quickly, faster than the memory controller can drain between reads. Once the DRAM write bandwidth reaches a threshold, the write back queue is drained by batch writing into the DRAM. We can observe this by the peaks followed by the trough pattern in the DRAM Write Bandwidth row. When the write queue is draining to the DRAM, any warp that needs to write or read will likely stall, whereas any warp performing compute will be uneffected. I'd imagine this is very useful for batched computations for things like FlashInfer where the KV cache is calculated by pages/batches (although I have yet to implement any sort of attention, this is a hypothesis).


<br><br><br>

<h2>Grid Stride + Vectorized vs Naive</h2>

**Performance Result:** 1.19% slower than Naive <br>
**Memory Throughput:** 1.32% slower than Naive

---

<h3>What Improved</h3>

1. Significant Instruction Reduction (-82.37%): 100M -> 17.5M
   - Why this happened:
     - Vectorization: Requests 4 floats with a single memory transaction, rather than four seperate memory transactions like in this naive, reducing the number of instructions.
     - Grid Stride: Grid Stride allows for a significant reduction in executed instructions but a more implicit and interesting way. When we look at the source counter of the naive kernel, 18.75% of all instructions are executed because of the _int i = blockIdx.x * blockDim.x + threadIdx.x_, this allows to get the threads id relative to the block its in. The interesting thing is for our naive kernel has 200,000,000 threads, which means every single thread executing this. With grid stride though, we are using 144 threads which means a significant reduction in the number of times we need to calculate this, which is supported by the grid stride instructions executed source counter for this line being just 0.02%.


2. Higher Occupancy (+7.80%): 85.36% -> 92.02%
   - Why this happened:
     - Grid Stride: The fixed block configuration of grid stride allowed for the number of blocks to perfectly fit the 24 SMs which each holds 6 blocks each for a total of 144 blocks, increasing occupancy.
    
3. More Active Warps per Scheduler (+8.16%): 10.12 -> 10.94 warps
   - Why this happened:
     - This is directly related to occupancy, where a higher occupancy often means each scheduler within the SM has access to more warps.

<h3>Why It's Still Slower</h3>

1. Increased Warp Cycles Per Instruction (+519.14): 180.16 -> 1115.44 cycles
   - Why this happened:
     - Grid Stride: Kernel must keep track of the iteration index, compare the iteration index to the n size and increment the iteration index. This creates instruction dependency chains where one must be done before the other. This instruction dependency likely contributes to the increased warp cycles required on average per instruction.
     - Vectorization: There is a minimal difference in terms of stall time when calling a single float versus a float4 because on a warp scale both are coalesced to the 32 byte sectors of the DRAM. After looking into the SASS, the increase in instructions was from:
       - _int n4 = n / 4:_ This is used to calculate the number of elements that are divisible and can be operated using float 4, and was responsible for 23.08% of all instructions executed
       - _Tail Handling:_ The tail handling was responsible for 7.69% of all instructions executed
       - **Note:** I believe in most vectorized kernels, the data is often padded by the host before being sent to the device, which would circumvent this problem completely. For the sake of learning why this is done in the first place, padding was not used.
   
2. Reduction in Eligible Warps (-76.20%): 0.06 -> 0.02 warps
   - Why this happened:
     - Grid Stride: The use of grid stride effectively reduced the number of blocks from the naive's 781,250 to 144. While this 144 count allowed for better occupancy, this does not give the SM enough warps to switch to when one warp stalls for latency hiding. Latency hiding is important especially for memory bound kernels as it allows the scheduler to switch to different warps when the current one is stalled, this allows each SM to always be busy rather than stalling waiting on memory.
     - Vectorization: The reason why vectorization plays a role in reducing the elgible warps is because of the required tail handling for when the data is not divisible by 4. This tail handling makes each warp to stall for longer, which means the scheduler will have even less warps to switch to as every warp will stall for a longer duration.<br><br>
       
      _**Note:** When I was first trying to understand why vectorization may cause an increased number of stalls, I thought it may have been because we we're calling for four floats per thread, rather than just one in the naive. After looking more into it and updating my mental model of vectorization I came to the following conclusion. While it is true we are increasing the bandwidth saturation and the required computation per thread, the actual speed at which the thread recieves the four floats is very similar to the naive._
   
      _This is because every thread deals with 4 floats, totaling to 16 bytes. On a warp scale where instructions are executed, this results in a call for 512 bytes of data. This perfectly coalesces with the 32 byte sectors of the DRAM. When we compare the naive kernel that requests for 128 bytes per cycle with the vectorized kernels which requests for 512 bytes, we can say that the vectorized kernel is four times more efficient when it comes to memory instructions!_


<br>
<h3>Why Grid Stride + Vectorized is Slower than the Naive</h3>
The Grid Stride + Vectorized kernel is effectively slower than the naive because the grid stride significantly decreases the parallelism of the kernel by reducing the number of warps available that the scheduler can switch to, and the tail handling of the vectorization increases the stall duration of each of those warps. These techniques combined essentially transform the kernel from not only just memory bound, but also latency bound where we simply don't have enough warps to switch between to hide the cycles required for each instruction.








<br><br><br><br><br><br><br><br><br><br>

























<h3>Hardware</h3>
On current GPU (RTX 4060):<br>
- 24 SMs<br>
- 6 max residental blocks per SM<br>
- 1,536 max residental thread per SM<br>
- 65,536 max registers per SM <br>
- 36,864 max threads for GPU <br>
- 1,572,864 max registers for GPU<br>
