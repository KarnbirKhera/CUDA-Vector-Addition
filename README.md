# CUDA Vector Sum Learning Report

<h1>Introduction</h1>
This project is the first step in my CUDA learning journey from scratch, starting with vector addition and then progressively adding common GPU optimization techniques.
Along the way I learned that "optimizations" don't always make things faster, and its important to use tools like NVIDIA Nsight to understand why and when to apply
specific optimization techniques. This project was also very good for me to learn memory coalescing, ...

<h1>Goals of the Project</h1>
- Understand how different CUDA kernel designs affect performance <br>
- Explore grid-stride loops, Vectorization and Instruction Level Parallelism (ILP)<br>
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
- Instruction Level Parallelism: Each thread issues multiple independent memory requests, allowing for computation as these requests come in to reduce latency.

Tradeoff of each Technique:
- Naive: Requires one thread per element, difficult to scale across various GPUs and large n size.
- Grid-stride: Increased register pressure. Can provide un-necessary overhead if threads > n. In large datasets.
- Vectorization (float4): Increased register pressure, requires 16 byte alignment, increases coalescing complexity, can require tail handling if n is not divisble by 4.
- Instruction Level Parallelism: Increased register pressure, increases coalescing complexity.

**_Register pressure note:_**<br>
_Increased register pressure can lead to lower occupancy, if register count per thread exceeds hardware maximum, register spills into slower L2 cache. In this case, an RTX 4060 has a maximum register per thread count of 255._

<h1>Benchmark Methodology</h1>
- Time elapsed measured using CUDA Events. This allowed accurate profiling of kernel runtime without launch overhead or other miscellaneous factors such SM clock frequency. <br>
- Nsight compute uses for throughput measurements.<br>
- Efficiency calculated by (Nsight Compute Throughput)/ 272.0, the maximum theoretical memory bandwidth for RTX 4060.
<br>
- For each input size, the respective kernel had three warmup runs, followed by 10 trials which were recorded.

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







<h1>A Deeper Dive into Vector Addition</h1>
To better understand the performance of each of the optimization techniques, an analysis into the vector addition operation itself provides a great starting point.

<br>
<p align="center">
  $C = A + B$
</p>
<br>

We can see that the kernel will have two read requests (A and B), one floating point operation (A + B) and one store request (C). This means the kernel will move a total of 12 bytes of data (3 floats) and will perform one floating point operation (addition). We can plug these values into the following formula to determine the arithmetic intensity of the kernel.


$$ \text{Arithmetic Intensity} (AI) = \frac{\text{Total Operations (FLOPs)}}{\text{Total Bytes Transferred (Memory Traffic)}} $$


The result after plugging in the values results is $$0.08 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$, when compared to the arithmetic intensity of the RTX 4060 ($$55 \text{ } \frac{\text{FLOPs}}{\text{Byte}}$$), the value is significantly lower which implies the operation is memory bound. This along with the consistent 80% to 90% DRAM bandwidth usage across all techniques and n sizes suggests this kernel is memory bound.






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

<h4>The Original Naive</h4>
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

> _Note: I later discovered this model was incomplete. The next section "Revisiting the L2 Cache Write Behavior" depicts the behavior more accurately._

For this kernel we have a single 4 byte float write per thread, on a warp scale this results in a perfect 128 byte write to the 128 byte cache line. This means for every new warp write, there is a very high likelyhood the required four 32 byte sectors in the DRAM are not in the L2 cache. This means every warp write will result in a cache line miss, because of this the kernel is required to do a DRAM read of the four 32 byte sectors in the DRAM before writing. This is supported by following images where the L2 lts__t_sectors_op_read.sum is 1,538,775, and in the Nsight compute kernel we have a total sector misses to device of 1,508,103, this suggests these L2 cache misses result in a signfiicant amount of reads made by the kernel. This GPU behavior of cache miss resulting in a read is supported by the following NVIDIA post https://forums.developer.nvidia.com/t/how-do-gpus-handle-writes/58914/5.

After looking into this even more, I found that this read write policy is very intentional and is actually fundamental to the hardware/software co-design of not just the RTX 4060, but for all modern GPUs. The reason for this read write policy is very interesting because it solves a problem that before this I had never even thought of.

Say our kernel was able to write directly into the DRAM, bypassing the L2 cache. At first I thought this was a great idea, we'd avoid the L2 cache entirely and I assumed it would result in faster performance. The problem with this is the DRAM can only load a single cache line into its buffer at a time. To write to the DRAM, you first have to open the cache line, this means using the DRAM's 2-8kb cache to load the entire row, this is a ~50 nanosecond action, then writing to the row which is often ~20 nanosecond action. At first this may seem trivial, but from the perspective of parallel programming this can prove very in-efficient.

Say we have the following scenario where both SM One and SM Two are writing to the DRAM in parallel.

- SM One: Opens cache line A and it is loaded into the DRAM's single row buffer (50 nanoseconds) and writes a single 16 byte value (20 nanoseconds)
- SM Two: Opens cache line B and it is loaded into the DRAM's single row buffer (50 nanoseconds) and writes a single 16 byte value (20 nanoseconds)
- SM One: Has to again open cache line A into the DRAM's single row buffer (50 nanoseconds) and writes a 16 byte value (20 nanoseconds)
- SM Two: Has to again open cache line B into the DRAM's single row buffer (50 nanoseconds) and writes a 16 byte value (20 nanoseconds)

The in-efficiency lies where the DRAM's single row buffer does not allow the full strength of parallel programming to shine, and this is where the L2 cache comes in. The way the L2 cache works is the following:

> Note from future self:
> Each memory bank in the DRAM has its own row buffer, not a single row buffer for the entire DRAM.

- SM One: Writes to L2 cache with a single value, targeted at memory sector A in the DRAM
- SM Two: Writes to L2 cache with a single value, targeted at memory sector B in the DRAM
- SM One: Writes to L2 cache with a single value, targeted at memory sector A in the DRAM
- SM Two: Writes to L2 cache with a single value, targeted at memory sector B in the DRAM

- The L2 cache: Opens a cache line to memory sector A (50 nanoseconds) and batch writes two 16 byte values (~20 nanoseconds)
- The L2 cache: Opens a cache line to memory sector B (50 nanoseconds) and batch writes two 16 byte values (~20 nanoseconds)

Rather than needing to re-open cache lines, the L2 cache gathers all the needed values to be written for each memory sector, then does a single read followed by a batch write. This turns an inefficient 280 nanosecond operation into an efficient 140 nanosecond operation.

Now going back to the STG.E, it turns out after more research the .E is actually a modifier for the STG command when it comes to the caching policy. The .E represents that this data will follow a normal replacement policy, meaning it is neither elevated to be removed first, or elevated to be kept. This adds an interesting layer of granularity because while we cannot change the fundamental nature of the read write structure, we can also modify the L2 cache policy of the data we send.

Looking at vector add kernel, we know that the entire process itself is purely streamed data, meaning data is loaded once, and never used or needed again. We can actually modify this cache policy with a ".cs" modifier which hints to the PTX to SASS compiler that the data is not needed once used, allowing it to be evicted first, which should in theory in-directly allow for more data to flow through the L2 cache. The reason this would theoretically allow more data to flow is because the default cache policy evicts the oldest or least recently used, but using the ".cs" should save the cache controller time and compute as it does not need to find/calculate the oldest cache line. 

However, I would imagine this comes at a cost of having to "tag" or use meta-data to convey to the compiler that this specific cache line can be evicted first. I would once again imagine, when n is low the cost of adding this "tag" would likely outweigh its need, whereas if we have a large n size, the "tag" overhead becomes minimial.

There is also another cache policy modifier known as .lu (Last Use). This means once the cache line is utilized, the cache line is disposed of even if the L2 cache is not full. This likely has the same n size use case mentioned for .cs (Cached Streaming).


All three kernel types, naive, naive cache streamed and naive last use performed within 1-2% of each other on n sizes 50M, 100M and 200M. One thing to note is the difference in the PM Sampling timeline for the STG.E (naive) and the STG.E.EF (cached streaming) policies.

STG.E (naive)
<img width="1673" height="231" alt="image" src="https://github.com/user-attachments/assets/1c0bfa3c-9613-4fd4-b580-781c510e4dc4" />

STG.E.EF (cached streaming)
<img width="1653" height="222" alt="image" src="https://github.com/user-attachments/assets/c0010cea-f1c1-4ed5-8814-f237ce2d2445" />

The naive STG.E kernel produced a very consistent DRAM and L2 cache throughput. This is because the default caching policy (STG.E) marks cache lines that are either the oldest or least used for eviction when a new cache line is needed. This naturally populates the write back queue to the DRAM slowly, allowing the write back queue to drain as needed between reads, keeping the queue at a steady level.

The naive cached stream kernel produced a very "bursty" DRAM and L2 cache throughput. This is because when using cached stream policy (STG.E.EF), as soon as a cache line is dirty (data is modified), the cache line is marked to be evicted. This means when a new cache line is needed, the write back queue to the DRAM fills very quickly, faster than the memory controller can drain between reads. Once the DRAM write bandwidth reaches a threshold, the write back queue is drained by batch writing into the DRAM. We can observe this by the peaks followed by the trough pattern in the DRAM Write Bandwidth row. When the write queue is draining to the DRAM, any warp that needs to write or read will likely stall, whereas any warp performing compute will be unaffected. I'd imagine this is very useful for batched computations for things like FlashInfer where the KV cache is calculated by pages/batches (although I have yet to implement any sort of attention, this is a hypothesis).
<br><br><br>





<h2>Revisiting the L2 Cache Write Behavior</h2>
Well this is very, very exciting! While making a LinkedIn post about the L2 cache behavior I learned about because it was genuinely so interesting to me, I noticed that the data I had didn't necessarily fully match with my understanding of how the L2 cache worked. Specifically, my theory prior to now was the following:<br>

- Case One: When writing to DRAM, if cache HIT: A single write instruction
- Case Two: When writing to DRAM, if cache MISS: A single read instruction, followed later by a write instruction

While this does capture some of the naunces of the L2 cache, it over generalizes the behavior of the L2 cache where it assumes all writes are hits because the data we are modifying is either already in the L2 cache, or is in the L2 cache because of a prior implicit read. The data that I mentioned in the section prior actually supports the idea that my old theory did not fully capture how the L2 cache behaves. The specific data that caught my eye was the following: <br>

- lts__t_sector_op_write_hit_rate.pct = 100.00%
- lts__t_sectors_op_read.sum = 1,538,775
- lts__t_sectors_op_write.sum = 25,000,362

Lets start with why I thought my theory was correct. The 100% hit rate specifically for writes led me to believe that my old theory was true at the time where in either case a hit is always guaranteed. The reason I believe my theory was wrong because of the latter two data points. <br>

If my prior theory was correct, for 25,000,362 sector writes, we would have approximately the same number of reads. This is because naive vector add is perfectly coalesced where a single warp will request to write 128 bytes, which perfectly fits 4 sectors in the DRAM. This means that in my old theory, we should see the same number of reads as the write because each warp will always need to call for new DRAM sectors to modify, but by looking at our data, 1.5M reads is significantly below 25M writes which implies the 1.5M must be some sort of overhead, and not related to the writes done by the L2.

After realzing this inconsistentency, I ended up doing some research to understand what might explain this L2 cache behavior. While I wasn't able to find any publically facing NVIDIA documentation to explain this behavior, the following two sources I believe capture what I saw with my data.

To start off, I'll explain the first source from my prior section (https://forums.developer.nvidia.com/t/how-do-gpus-handle-writes/58914/5) which I believe explains half of the behavior we saw with our L2 cache. This post mentions the following two possibilities for the write mechanic used by the L2 Cache:

- When performing a write, The L2 cache marks the valid and invalid bytes (bytes that were changed versus not) using a mask. The DRAM then reads this mask when the cache line is evicted from the L2 and updates the values accordingly.
- When performing a write, the L2 cache immediately reads the sector from the DRAM on a write miss, and then the L2 cache updates the needed values of the sector just read, and the values in the DRAM are updated once the cache line has been evicted (This is the read-write policy is the theory I orginally adopted).

Now our second source "Exploring Modern GPU Memory System Design Challenges through Accurate Modeling" (https://arxiv.org/abs/1810.07269) published on arXiv provides empirical evidence for a different write policy named "write-validate" based on the Volta architecture.

- What write-validate does: When the L2 recieves a write, it doesn't fetch from DRAM. It instead writes the bytes directly into the sector (in the L2 cache) and sets the corresponding bits modified using a write mask, marking the written sector as valid and modified.
- What happens at eviction: If the mask is full (all 32 bytes of the sector are written), the sector is written back to the DRAM without reading the sector beforehand. If the mask is partial, meaning the modified bytes were less than 32, then the missing bytes must first be read from the DRAM, then used to produce a complete write mask before writing to the DRAM.

- This brings up the question that upon recieving a write request, does the L2 cache immediately read the DRAM, or does it wait until after the cache line is modifed and evicted? The researchers used the following experiment to answer this question. They first modified a few bytes in a sector, and then immediately afterwards, read the same sector which resulted in a miss in the L2 cache. This experiment proves that the L2 cache does not commit a read to the DRAM upon recieving a write, because if it had, the L2 sector would have resulted in a hit by the researchers.


To confirm this theory using my own data, I did the following experiment with two write only kernels.

- Coalesced Write Only: A naive write only kernel where we are not reading any values just writing the value 1.0f with perfect coalescing.
- Uncoalesced Write Only: Similar to the coalesced kernel, but the thread index was multiplied by 8 to make sure for every sector, we only write 4 bytes out of the given 32 byte sector for uncoalesced access.

The Coalesced Write Only kernel was given 10 million threads to write across 10 million elements, and the Uncoalesced Write Only kernel was given 10 million threads to write across 80 million elements to make up for the stride value of 8. If the theory by the paper holds true, we should see the following results:

- For the Coalesced Write Only kernel, the ratio between number of written sectors verus those read should be near zero. This is because all of our writes are coalesced, and for every sector we modify, we always modify all 32 bytes with that sector. This means for every write, we should not have a following read.
- For the Uncoalesced Write Only kernel, the ratio between the numbers of written sectors versus those read should be near 1:1. This is because for any given sector, we only modify 4 out of the 32 bytes, this means the L2 cache will need to read the DRAM sector to fully complete its write mask before evicting the dirty cache line.

<h3>Results</h3>
  
Coalesced Write Only:
- Device Memory:
  - Load (Sectors): 37,508
  - Store (Sectors): 733,320 
- **Calculated Load/Store Ratio:**
  -  **1 load : 19.55 stores**

Uncoalesced Write Only:
- Device Memory:
  - Load (Sectors): 11,016,972
  - Store (Sectors): 10,017,664
- **Calculated Load/Store Ratio:**
  - **1.09 loads : 1 store**

Coalesced Write Only Results:
  - From the coalesced access results we can deduce the following. When writes are perfectly coalesced, the L2 cache has no reason to perform a read of the DRAM as the modified cache line already has a fully 32 byte valid write mask. Thus, the cache line can be properly evicted to the DRAM. This is supported by the low ratio of loads to stores, where one could conclude the 37,508 loads are likely an overhead of the kernel, and not related to the writes performed

Uncoalesced WRite Only Results:
  - From the uncoalesced access results, we can deduce the following. When writes are not coalesced, the L2 cache has to perform a read into the DRAM for every incomplete sector we do not fully modify. This is supported by the near 1:1 ratio of loads to writes.

<h3>Summary of Experiment</h3>
Based off the following results on the Ada Lovelace architecture (RTX 4060), one can theorize that the L2 cache uses a write-validate policy as follows:
  - If the write(s) performed fully modify all of the 32 bytes of the given sector, the L2 cache does not need to read the sectors' contents in the DRAM, and can properly evict the sector in its write back buffer.
  - If the write(s) performed do not fully modify the 32 bytes of the given sector, the L2 cache needs to perform a read of the sector in the DRAM before it can properly evict the sector in its write back buffer.
  
While I've learned that the explicit cost of uncoalesced access is essentially wasting precious bandwidth, it has been very interesting to learn that on the Ada Lovelace architecture, there is also an implicit read cost as well!

Circling back to the origin of this investigation which was why vector add had a ~31% hit rate despite being a fully streaming kernel, I have the following claim. Vector add performs two reads, A and B, and performs a single write C. We've already confirmed using our read only kernel that each A and B read has a hit rate of ~0.07%, which leaves us to understand why the hit rate of our single write is always 100%. 

>Note in the section before, I isolate the vector add kernel into their read and write variants isolated. After looking back now with the experience I have reading Nsight compute, I can see that even the orginial vector add kernel was hinting that the write hit rate was a 100% using the lts__t_sector_op_write_hit_rate.pct metric, meaning we did not need to isolate those variants. None the less, the process itself was very fun even if it might have been not been needed.

The reason why I believe the hit rate of write is always a 100% is because no matter what case we hit, whether that be coalesced or uncoalesced access, the L2 will always allocate a sector locally on a write. This means once the kernel sends the write request and it reaches the L2 cache, the write always has a way to reach the required DRAM sector. <br><br><br>


> I did end up making a LinkedIn post on this, where I drew up the following image to demonstrate what the partial write process looks like under the write-validate policy. I hope this helps those whom are visual learners!
><img width="1450" height="1246" alt="image" src="https://github.com/user-attachments/assets/310d10b2-f188-4042-9d17-acd40f34d481" />





<h2>Grid Stride vs Naive</h2>

**Performance Result:** 3.07% slower than the Naive <br>
**Memory Throughput:** 2.73% less than the Naive

---

<h3>What Improved</h3>

**1. Higher Occupancy (+15.66%): 85.61% -> 99.02%**

  - Why this happened:
    - The naive kernel uses 781,250 blocks at 256 threads each to process all 200,000,000 elements. This means there is likely some overhead due to the number of blocks we launch that prevents each SM from fully occupying the maxmimum 6 active blocks per SM, which is mentioned in the Achieved Occupancy section under the Nsight Compute Analysis for the Naive kernel. In the grid stride kernel however, we launch 144 blocks, which reduces the block overhead faced in the naive kernel allowing for a higher occupancy per SM.

<h3>Why It's Still Slower</h3>

**1. Significant Decrease in Eligible Warps Per Scheduler [warp] (-52.67%): 0.06 -> 0.03**

  - Why this happened:
    - This is a direct cause of the decreased number of blocks from the naive. While we increase occupancy to 99.02%, we effectively remove the kernels ability to latency hide using other warps. Where say a warp stalls on a memory request, in the naive kernel the scheduler is able to switch to another warp to continue actively working, but in the grid stride kernel there is simply not enough warps that are not stalled to switch to.








<h2>Vectorized (float4) vs Naive</h2>

**Performance Result:** 0.46% slower than the Naive<br>
**Memory Throughput:** 0.05% less than the Naive<br>

---

<h3>What Improved</h3>

1. **Significant Decrease in Executed Instructions (-59.37%): 200,000,000 -> 40,625,063**

  - Why this happened:
    - This is the direct reason why when it comes to reducing instructions, why Vectorization is such a great tool. Rather than a single thread requesting a single float, we request for 4 floats at once using a single memory request. At first when I learned about Vectorization it seemed very inadvertent because while we request 4 floats with a single memory request, doesn't that mean we need to move 4x the amount of data therefore, likely take four times the amount of time? It turns out it actually takes nearly the exact amount of time, and the reason it does is actually very exciting and goes to the core of the GPU architecture.
    
    - The reason why recieving 4 floats at once is very similar to the amount of time it takes to recieve a single float is because of the way the DRAM's architecture is setup. When we request a single 4 byte float, we are essentially eating the cost of reading the DRAM sector for just a 1/4th of the data it holds, although note in reality instructional commands are executed at the warp level so we would actually use all 32 bytes in said sector. When we use Vectorization, we still eat the same cost of reading the DRAM sector, but we are using 16 bytes out of the 32 bytes that sector holds so we are essentially getting more data per read, and again at a warp level we are actually using all of the data contained within this DRAM sector.

While understanding this, I thought why dont we use float8 instead? Because wouldn't 8 floats mean the thread would perfectly request 32 bytes of data, which perfectly fits the 32 byte DRAM sector using just a single memory request? The reasonining why this wouldn't be as efficient is again leads us back to the GPU architecture, specifically to the size of the 128 bit memory bus from the L1 cache to the registers.<br><br>

If we were to use float8, that would mean each thread requires 8 (floats) * 4 (size of float) = 32 bytes, which converted to bits is 256 bits. This means for every thread, we would would need a very costly two read operations which is very in-efficient compared to float4 where we request 4 (floats) * 4 (size of float) = 16 bytes which is 128 bits, which only requires a single read transaction as it fits perfecetly within the L1 to register memory bus. This means the reason why float4 works so great is because we're balancing a fine line where we make sure to utilize all the data we can from a single DRAM read, while also making sure not to tip over into needing two costly DRAM reads. <br><br><br>

<h3>Vectorization Width Analysis</h3>
While float8 is not supported in CUDA, so our earlier float8 analysis is just theory, lets do deep dive into float, float2, float4 and the theoretical float8 to understand the best use case for each.

- float:
   - A thread uses a single LDG.32 bit load instruction for a 4 byte float, on a warp scale thats 128 bytes. These 128 bytes require a single cycle to be processed by the 128 byte L2 cache line. These 128 bytes fit into the 4 KB row buffer of the DRAM, and also perfectly fits into four 32 byte sectors of the DRAM.
 - float2:
   - A thread uses a single LDG.64 load instruction for a 8 byte float2, on a warp scale thats 256 bytes. These 256 bytes require two cycles to be processed by the 128 byte L2 cache line. These 256 bytes fit into the 4 KB row buffer of the DRAM, and also perfectly fits into eight 32 byte sectors of the DRAM.
 - float4:
   - A thread uses a single LDG.128 load instruction for a 16 byte float4, on a warp scale thats 512 bytes. These 512 bytes require 4 cycles to be processed by the 128 byte L2 cache line. These 512 bytes fit into the 4 KB row buffer of the DRAM, and also perfectly fits into sixteen 32 byte sectors of the DRAM.
 - float8:
   - A thread uses two LDG.128 load instructions for a 32 byte float8, on a warp scale thats 1024 bytes. These 1024 bytes require 8 cycles to be processed by the 128 byte L2 cache line. These 1024 bytes fit into the 4 KB row buffer of the DRAM, and also perfectly fits into thirty two 32 byte sectors of the DRAM.

 Now lets dig into the pros and cons of each.
 
 **Float:**
   - Pros
     - Great if register pressure per thread is a limiter.
     - Great for parallelism if register count per thread is the limiter.
   - Cons
     - When compared to float2, we are producing 2x the instructions needed for the same amount of data.
     - When compared to float4, we are producing 4x the instructions than needed for the same amount of data.
   - Use Case:
     - If the bottleneck is register pressure, we can trade register pressure for instructional pressure.
        
 **Float2:**
   - Pros
     - Great compared to float because 2x less instructional pressure
     - Great compared to float4 because slightly less register pressure
   - Cons
     - When compared to float, we have a slight increase in register pressure. 
     - When compared to float4, we are producing 2 times the instructions than needed for the same amount of data float4 produces.
   - Use Case:
     - If we want to slightly trade increased register pressure, for decreased instructional pressure.
       
 **Float4:**
   - Pros
     - Great compared to float because 4x less instructional pressure
     - Great compared to float2 because 2x less instructional pressure
   - Cons
     - Compared to float, modererate increased register pressure
     - Compared to float2, slight increased register pressure
   - Use Case
     - If we want to trade moderate increase in register pressure for a moderate decrease in instructional pressure.
     - Float4 is effectively the tipping point where we move the most amount of data for the least amount of instructional pressure.
        
 **Float8:**
   - Inefficient as it requires two LDG.128 loads instructions.
   - Upon looking into why an LDG.256 doesn't exist, its likely because the width of the data that can move from the register to the load store unit is 128 bits (matching our prior LDG.128 instruction).
      
      
     

<h3>Why it's slower</h3>

**1. Signfiicant Decrease in Eligible Warps Per Scheduler [warp] (-56.40%): 0.06 -> 0.03**
  - Why this happend:
    - To use Vectorization withinn this context where we do not use grid stride, we must launch n/4 threads. This means although the techinique itself does not reduce parallelism, to ensure we compute all the given elements and nothing less and nothing more, we must launch less warps compared to the naive which means less warps to switch to for the SM when one stalls.<br><br>
      
    > _Note: Vectorization itself does not reduce parallelism, rather without grid stride we must launch a quarter of the threads compared to the naive. This means the reduction in parallelism is because of the launch condition and not the technique itself._
    
      <br><br>











<br><br><br>

<h2>Naive vs Grid Stride + Vectorized</h2>

**Performance Result:** 1.19% slower than Naive <br>
**Memory Throughput:** 1.32% less than Naive<br>

---

<h3>What Improved</h3>

1. Significant Instruction Reduction (-82.37%): 100M -> 17.5M
   - Why this happened:
     - Vectorization: Requests 4 floats with a single memory transaction, rather than four seperate memory transactions like in this naive, reducing the number of instructions.
     - Grid Stride: Grid Stride allows for a significant reduction in executed instructions but a more implicit and interesting way. When we look at the source counter of the naive kernel, 18.75% of all instructions are executed because of the _int i = blockIdx.x * blockDim.x + threadIdx.x_, this allows to get the threads id relative to the block its in. The interesting thing is for our naive kernel has 200,000,000 threads, which means every single thread executing this. With grid stride though, we are using 144 blocks which means a significant reduction in the number of times we need to calculate this, which is supported by the grid stride instructions executed source counter for this line being just 0.02%.


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

  > Note from future self: <br><br>
  > Both vectorization and grid stride reduce the number of instructions by a factor of 6. While the kernel still faces the same long scoreboard stall (memory bound), the reduced number of instructions likely inflates the Warp Cycles per Instruction value.

   
2. Reduction in Eligible Warps (-76.20%): 0.06 -> 0.02 warps
   - Why this happened:
     - Grid Stride: The use of grid stride effectively reduced the number of blocks from the naive's 781,250 to 144. While this 144 count allowed for better occupancy, this does not give the SM enough warps to switch to when one warp stalls for latency hiding. Latency hiding is important especially for memory bound kernels as it allows the scheduler to switch to different warps when the current one is stalled, this allows each SM to always be busy rather than stalling waiting on memory.
     - Vectorization: The reason why Vectorization plays a role in reducing the elgible warps is because of the required tail handling for when the data is not divisible by 4. This tail handling makes each warp to stall for longer, which means the scheduler will have even less warps to switch to as every warp will stall for a longer duration.
     >Note: Vectorization allows for 4 floats per single memory instruction. This will result in fewer warps in flight meaning a decrease in the number of eligible warps the scheduler is able to switch to.









<br><br><br>



<h2>ILP=4 vs Naive</h2>

**Performance Result:** 0.36% slower than Naive <br>
**Memory Throughput:** 0.35% less than Naive<br>


---

<h3>Current Theory</h3>

1A. Executed Instruction per Cycle (-16.89%) 0.22 -> 0.19<br>
1B. Eligible Warps per Scheduler (+0.17%) 0.06 -> 0.07
   - What this means:
     - With my current mental model, these two statistics point out as interesting to me, and I think at the moment they may show something positive despite a reduced IPC often meaning the kernel is less performative. To start from the beginning, vector add is a memory bound kernel, this means to squeeze out more performance from my current understanding we have the following avenues to pursue
       - Increase parallelism to increase latency hiding by increasing the number of eligible warps per scheduler
       - Reduce memory dependencies to decrease long scoreboard stalls
       - Reduce computation dependency to either reduce long scoreboard stalls, or allow for more active occupancy by reducing register count if that is the limitng factor
     - I believe what the ILP=4 kernel is doing is the first, but not using more warps, but rather using latency hiding at the instruction level. In hindsight, I suppose the name makes sense where schedulers can hide latency by switching to different warps, but ILP allows for latency hiding at the thread/instruction level.
    
     - The reason why the two statistics I chose stood out to me and why I think their a positive is the following
       - A decrease in IPC by -16.89%
         - Now usually a decrease in IPC indicates that we are executing less instructions per cycle meaning we are doing less work per cycle. The thing that was interesting to me though was compared to every other variant of vector add which often lost 50-60% of IPC, the ILP kernel only lost 16.89%. This tells me while the other kernels suffered a loss in IPC because of the technique itself (e.g. grid stride or vectorized), ILP may be suffering not from the technique itself but because we are launching only a fourth of the number of blocks as the naive. This is important because the more blocks we have the more parallelism we can achieve, at least until of course we get diminishing returns and the overhead of adding blocks becomes significant.
       - A increase in Eligible Warps per Scheduler by +0.17%
         - Now this is the statistic that makes me think my theory might be true. Every other kernel variant had more occupancy than the naive, but suffered signfiicantly when it came to the number of eligible warps it had, but ILP is the opposite. This makes me think again the limiting factor for this kernel is not the technique itself, but rather the schedulers limited number of blocks to switch to compared to the naive. <br>
       - This makes me think that if ILP were to have the same number of blocks as the naive, it may perform better than naive as both will have the same level of warp parallelism.


<h3>Testing the Theory</h3>
 To test this theory, I will be comparing Naive + Grid Stride and ILP=2 + Grid Stride.
 
   - I will be using Grid Stride so that both kernels have the same number of fixed blocks, and not based off of how many elements are processed per thread.
   - I will be using ILP=2 instead of ILP=4 because upon profiling ILP=4, the kernel is faced with LG stalls due to the memory bandwidth not being able to keep up with the number of instructions per thread.<br><br><br><br>




     
<h2>Results of ILP=2 + Grid Stride vs Naive + Grid Stride Theory</h2>

**Performance Result:** 0.16% slower than Naive + Grid Stride<br>
**Memory Throughput:** 0.12% less than Naive + Grid Stride<br>


---
What this tells me is the following:
  - Despite the difference betweenn both kernels being the ILP unrolls the loop once, both kernels perform at very similar speeds, throughput and eligible warps where the difference in SM frequency will likely play the most crucial factor in which kernel is faster. This tells me that while my theory was incorrect as ILP did not speed up the kernel beyond what I could consider a negligible/noise difference.
  - The specific part of my theory that failed was the assumption that ILP could combat the memory bound nature of vector add by increasing parallelism at the instruction level. To update my mental model based off this experiment I would say while we may increase the instruction level parallelism of the kernel, if a warp is stalled and the SM is unable to switch another warp that is not, the kernel still faces the exact same problem as the naive even if we allow parallelism at the instruction level. <br><br><br><br>



<h1>What I learned</h1>

<h2>Vector Add</h2>
The vector add kernel is memory bound. Any optimizations to be made to improve the kernels run time must address this. Any optimization that does not will often add needless overhead.

<h2>Nsight Compute</h2>
Now this was one of the most enjoyable parts of this project. While at first this project started as a way to learn grid stride, vectorization and ILP, it slowly became a vehicle to understand how memory bound kernels operate.<br><br>

From this project I learned:
- When the write we commit to the L2 cache does not modify the entire 32 byte sector, the L2 cache must read the DRAM sector first before writing to the DRAM. I find this implicit read, and the process it took for me to discover it genuinely enjoyable and exciting. The idea of having this niche detail in my mental model excites me for the future when I will be able to use it. I can already imagine during say sparse writes, knowing this is the difference between performing a single write instruction, or a read followed by a write.
  
- Another thing I enjoyed learning about was the intentionality of the hardware. In past projects where I've built things like full stack applications to ML models, there has always been a layer of abstraction that has prevented me from digging deep in to finding out why something worked the specific way that it did. When it comes to CUDA and low level systems work, quite literally everything has a reason. Whether that be why the cache line is 128 bytes, and how it perfectly fits 4 sectors of the DRAM, or the reason why float4 is so spectacular is because its the perfect tipping point where we can maximize the 128 bit register file using a single LDG.128 instruction, it all has a reason.

  The L2 cache investigation took me two days, and then a few days later I had realized I was wrong and I had to spend another day understanding why. While this might seem mundane at first, being able to connect how the software beautifully uses the hardware to maximize throughput has genuinely, without a single doubt, been the best computer science experience as a undergraduate student. While I have not taken a parallel programming class, so I am unsure about the exact depth of which these many layers interact, and how we can use them to our advantage, I have no doubt that learning and updating my mental model will genuinely be enjoyable.

  The funny thing is the night before I figured this out, I remember thinking all of the possiblities of why the L2 cache policy was the way it was, and I remember finding the arXiv article and all the dots suddenly connecting. This quite literally made me so estatic that I made sure to write "holy macroni" in this github readme so I wouldn't forget that hey! lets test this theory in the morning, it actually might be the reason why your seeing the numbers your seeing! 


<h1>What I Would Do Differently</h1>

If I were to restart this vector add project from scratch with the knowledge I have now (and any project from now on) this would be my process:

**1. I would disect the vector add equation itself before I even think to write a single piece of code. This gives me the theoretical FLOPs/byte. This is important because before we even start, we should understand what our expected bottlenecks will be.**

$$ \text{Arithmetic Intensity} (AI) = \frac{\text{Total Operations (FLOPs)}}{\text{Total Bytes Transferred (Memory Traffic)}} $$

  - For vector add we concluded that it performs two reads (A and B) a single write (C) and performs a single floating point operation (add). This results in a FLOPs/byte of 0.08, which between compute and memory bound, this tells us our kernel is memory bound.
  
  - While this provides a great insight into what our bottleneck might be between compute and memory, it turns out theres a even sneakier possible bottleneck. It turns out another potential bottleneck is actually latency! Latency bound is when the number of instructions being performed is significantly more than the L2 cache can handle. From my current understanding, this results in the write-back buffer in the L2 cache to be very convoluted to the point where if the write buffer is full, the L2 cache has to stop all reads and write warps to evict all of the dirty cache lines.<br><br><br><br><br><br>








**2. Once the equation is analyzed, I would do the following which I recently learned about. This process is a lot more complicated (which makes it alot more interesting), but offers a significant theoretical view into our kernel.**

<h3>A. DRAM Bandwidth Bound </h3>
How long would the kernel take if the DRAM delivered bytes at its theoretical maximum speed (seconds)?<br><br>


$$ T_{DRAM} = \frac{\text{Total Bytes Transferred}}{\text{Peak DRAM Bandwidth}} $$

Needed information to calculate:
- **Kernel:** N, Reads per Element, Writes per Elements, sizeof(type)
- **Hardware:** Peak DRAM bandwidth




<br><br><br>






<h3>B. Compute Bound </h3>
How long would the kernel take if every core computed at its theoretical maximum (seconds)?<br><br>


$$ T_{Compute} = \frac{\text{Total FLOPs}}{\text{Peak FLOPS}} $$

$$ \text{Peak FLOPS} = \text{CUDA Cores} \times \text{Clock Speed} \times 1 \text{ (Two if Fused Multiply Add)} $$


> NVIDIA GPUs can specifically perform the Fused Multiply Add operation in a single cycle!. This is amazing because if we go back to our knowledge of neural networks, the fundamental equation used across neural networks is weight * input + bias. I also believe the origin of the neural network, the perceptron, uses the same equation! One could assume that the hardware was specifically tailored for this exact equation and purpose!


Needed information to calculate:
- **Kernel:** N, FLOPs per element, whether the operation is Fused Multipy Add or not
- **Hardware:** CUDA cores, clock speed

<br><br><br>




<h3>C. Latency Bound </h3>
Are there enough memory requests in the pipeline right now to keep the DRAM busy, or is the DRAM idle? (Percentage) <br><br>

$$ \text{Latency Efficiency} = \frac{\text{Bytes in Flight}}{\text{Bytes in Flight Needed}} \times 100\% $$

$$ \text{Bytes in Flight} = \text{Total Warps} \times \text{Concurrent Memory Requests per Warp} \times \text{Bytes per Request} $$

$$ \text{Bytes in Flight Needed} = \text{Peak Bandwidth} \times \text{DRAM Latency} $$


Needed information to calculate:
- **Kernel:** How many independent memory requests can be inflight at once per thread, Bytes per memory request
- **Hardware:** Peak DRAM bandwidth, DRAM latency, Max warps per SM, Number of SMs<br>
  _Note: Can use actual achieved maximum warps instead_

<br><br><br>





<h3>D. L2 Cache Bandwidth Bound</h3>
Can the L2 cache feed data to the SMs fast enough, or is it a bottleneck? (seconds)<br><br>

$$ T_{L2} = \frac{\text{Total Bytes through L2}}{\text{L2 Bandwidth}} $$


Needed information to calculate:
- **Kernel:** Total bytes passing through L2, can differ from DRAM due to write-validate reads or cache hits
- **Hardware:** L2 bandwidth


<br><br><br>




<h3>E. Instruction Issue Bound</h3>
Can the warp scheduler issue instructions (math, memory, control flow, etc) fast enough? (seconds)<br><br>

$$ T_{Issue} = \frac{\text{Total Warp Instructions}}{\text{Warp Instruction Issue Rate}} $$

$$ \text{Warp Instruction Issue Rate} = \text{Schedulers per SM} \times \text{Num SMs} \times \text{Clock Speed} $$

Needed information to calculate:
- **Kernel:** Total instructions per element (from Nsight)
- **Hardware:** Schedulers per SM, Number of SMs, Clock speed, Warp size


<br><br><br>




<h3>F. Load/Store Unit Throughput Bound</h3>
Can the Load Store Unit in each SM, which only translates memory instructions to actual memory requests, issue them fast enough? (seconds)

$$ T_{LSU} = \frac{\text{Total Warp Memory Instructions}}{\text{LSU Issue Rate}} $$

$$ \text{Total Warp Memory Instructions} = \frac{N \times \text{Memory Ops per Element}}{\text{Warp Size}} $$

$$ \text{LSU Issue Rate} = \text{LSUs per SM} \times \text{Num SMs} \times \text{Clock Speed} $$


Needed information to calculate:
- **Kernel:** N, Memory operations per element (loads + stores)
- **Hardware:** Load Store Units per SM, Number of SMs, Clock speed, Warp size

<br><br><br>




<h3>G. Shared Memory Bandwidth Bound</h3>
Can shared memory serve all the reads and writes required fast enough? (seconds)<br><br>

$$ T_{SMEM} = \frac{\text{Total Shared Memory Transactions}}{\text{Shared Memory Bandwidth}} $$

$$ \text{Shared Memory Bandwidth} = \text{Banks per SM} \times \text{Bytes per Bank per Cycle} \times \text{Clock Speed} \times \text{Num SMs} $$

Needed information to calculate:
- **Kernel:** Shared memory transaction per element, bank conflict multiplier (1x if no conflicts)
- **Hardware:** Banks per SM, Bytes per bank per cycle, Clock speed, Number of SMs

<br><br><br>



<h3>H. PCIe Transfer Bound</h3>
Is the kernel waiting on data moving from the CPU and GPU, rather than actually computing? (seconds) <br><br>

$$ T_{PCIe} = \frac{\text{Total Bytes Transferred (Host} \leftrightarrow \text{Device)}}{\text{PCIe Bandwidth}} $$



Needed information to calculate:
- **Kernel:** Total bytes sent to device + Total bytes read back from device
- **Hardware:** PCIe bandwidth

<br><br><br>




<h3>I. Theoretical Runtime</h3>
The slowest bottleneck determines the kernel's speed.<br><br>

$$ T_{Kernel} = \max(T_{DRAM},\ T_{Compute},\ T_{L2},\ T_{LSU},\ T_{Issue},\ T_{SMEM},\ T_{PCIe}) $$


After looking into it, there are many other ways to be bounded as well! This is great because it allows us to diagnose our kernel at a much deeper level before we even write the kernel. The reason I wrote the time to write this all out is because I think it might actually be a very, very helpful way to know what optimization technique to use for this kernel and all the kernels beyond. I believe this might be the case because say we're DRAM Bandwidth bound, these equations tell us what variables contribute to our kernel's bottleneck, and what potential optimization technique we can perform to allow for more throughput!

To test this theory, lets apply the following equations to vector add where they apply!


<br><br><br><br>


<h2>Vector Add Equations</h2>

<h3>DRAM Bandwidth Bound</h3>
<img width="2083" height="501" alt="image" src="https://github.com/user-attachments/assets/07362b37-4a8b-4e01-af16-7f5e696d4915" />

<br><br><br>

<h3>Compute Bound</h3>
<img width="2103" height="697" alt="image" src="https://github.com/user-attachments/assets/8835f4e6-db46-46c3-8ba8-6cdd7d78ee86" />



<br><br><br>

<h3>Hardware</h3>
On current GPU (RTX 4060):<br>
- 24 SMs<br>
- 6 max residential blocks per SM<br>
- 1,536 max residential thread per SM<br>
- 65,536 max registers per SM <br>
- 36,864 max threads for GPU <br>
- 1,572,864 max registers for GPU<br>
