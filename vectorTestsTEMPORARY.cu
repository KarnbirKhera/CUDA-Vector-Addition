#include <gtest/gtest.h>
#include "vectorSumHeader.cuh"

/*
* On current GPU (RTX 4060):
*  - 24 SMs
*  - 6 max residental blocks per SM
*  - 1,536 max residental thread per SM --> overflow will cause waves, (needing to queue blocks, which is not necessarily a bad thing, but queueing blocks has overhead)
*  - 65,536 max registers per SM --> overflow causes register spilling into L1 cache (slower)
*  - 36,864 max threads for GPU
*  - 1,572,864 max registers for GPU
*/


/**
 * Naive Vector Sum Kernel:
 * -------------------------------------------------------------------------------------
 * Calculations:
 *  blockPerGrid = Calculated from formula to ensure 1 thread : 1 element ratio is met.
 *  threadsPerBlock: Common rule of thumb is multiples of 32 (32 threads per warp), results in 128, 256 or 512.
 *      
 *      128 threadsPerBlock results in low occupancy (not utilizing all threads)
 *      256 threadsPerBlock results in the highest occupancy without changing # of blocks
 *      512 threadsPerBlock results in 3 blocks per SM, but would require changing # of blocks.
 * 
 *  From my current understanding, more threads per block means more threads are able to communicate when it comes to shared memory.
 *  Bigger blocks means less granularity where while we have less scheduling overhead/block create overhead, but can result in less blocks to fill when one completes early.
 * 
 * With this configuration:
 *  - 1,536 threads per SM
 *  - 24,576 registers per SM (16 registers per thread for this kernel, from NVIDIA Nsight)
 *  - 36,864 threads total on GPU 
 *  - 786,432 registers total on GPU
 * 
 * Potential Improvement: 
 *  - Use of grid stride would detach the one thread to one element ratio, allowing the kernel to scale to n, resulting in the kernel being hardware-aware.
 *  - Use of vectorization (float4) would allow a single thread to process four floats per cycle. Reduces memory overhead by allowing one instructional call to global memory, rather than 4 instructional calls.
 */
TEST(NaiveVectorSumTest, naiveVectorSum) {
    
    int n = 1000000;
    size_t size = n * sizeof(float);

    //Allocate memory on CPU
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Add values to h_a and h_b
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    //Transfer from CPU to GPU
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    //Execute kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; //Common formula used to calculate # of threads needed to cover all elements to match 1:1 ratio
    vectorSum<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,d_c,n);

    //Transfer results from GPU to CPU
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    EXPECT_FLOAT_EQ(h_c[0], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n/2], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n-1], 3.0f);
    EXPECT_FLOAT_EQ(h_c[12345], 3.0f);
    EXPECT_FLOAT_EQ(h_c[999999], 3.0f);

    free(h_a);
    free(h_b);
    free(h_c);
}

/**
 * Grid Stride Vector Sum Kernel:
 * -------------------------------------------------------------------------------------
 * Calculations:
 *  blockPerGrid = 24 * 6 = 144 blocks
 *  threadsPerBlock: Common rule of thumb is multiples of 32 (32 threads per warp), results in 128, 256 or 512.
 *      
 *      128 threadsPerBlock results in low occupancy (not utilizing all threads)
 *      256 threadsPerBlock results in the highest occupancy without changing # of blocks
 *      512 threadsPerBlock results in 3 blocks per SM, but would require changing # of blocks.
 * 
 * With this configuration:
 *  - 1,536 threads per SM
 *  - 39,936 registers per SM (26 registers per thread for this kernel, from NVIDIA Nsight)
 *  - 36,864 threads total on GPU 
 *  - 958,464 registers total on GPU
 * 
 *  Potential Improvement: While the kernel is no longer constrained by n size, memory throughput could be improved through vectorization.
 */
TEST(GridStrideVectorSumTest, GridStrideVectorSum) {
    int n = 1000000;
    size_t size = n * sizeof(float);

    //Allocate memory on CPU
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Add values to h_a and h_b
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    //Transfer from CPU to GPU
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    //Execute kernel
    int threadsPerBlock = 256;
    gridStrideVectorSum<<<144,threadsPerBlock>>>(d_a,d_b,d_c,n);

    //Transfer results from GPU to CPU
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    EXPECT_FLOAT_EQ(h_c[0], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n/2], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n-1], 3.0f);
    EXPECT_FLOAT_EQ(h_c[12345], 3.0f);
    EXPECT_FLOAT_EQ(h_c[999999], 3.0f);

    free(h_a);
    free(h_b);
    free(h_c);
}

/**
 * Vectorized (float4) Vector Sum:
 * -------------------------------------------------------------------------------------
 * Calculations:
 * blockPerGrid = Calculated from formula to ensure 1 thread : 1 element ratio is met.
 * threadsPerBlock: Common rule of thumb is multiples of 32 (32 thread per warp), results in 128, 256 or 512.
 *                  128 threadsPerBlock results in low occupancy (not utilizing all threads)
 *                  256 threadsPerBlock results in the highest occupancy without changing # of blocks
 *                  512 threadsPerBlock results in 3 blocks per SM, but would require changing # of blocks.
 * 
 * 
 * With this configuration:
 *  - 1,536 threads per SM
 *  - 33,792 registers per SM (22 registers per thread for this kernel, from NVIDIA Nsight)
 *  - 36,864 threads total on GPU 
 *  - 811,008 registers total on GPU
 * 
 *  Potential Improvement: While the kernel decreases instructional calls to global memory through vectorization, could be improved by grid stride to make thread count independent from n. 
 */
TEST(vectorizedVectorSum, vectorizedVectorSum) {
    int n = 1000000;
    size_t size = n * sizeof(float);

    //Allocate memory on CPU
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Add values to h_a and h_b
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    //Transfer from CPU to GPU
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    //Execute kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = ((n/4) + threadsPerBlock - 1) / threadsPerBlock;
    vectorizedVectorSum<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,d_c,n);

    //Transfer results from GPU to CPU
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    EXPECT_FLOAT_EQ(h_c[0], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n/2], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n-1], 3.0f);
    EXPECT_FLOAT_EQ(h_c[12345], 3.0f);
    EXPECT_FLOAT_EQ(h_c[999999], 3.0f);

    free(h_a);
    free(h_b);
    free(h_c);
}



/**
 * Vectorized (float4) Grid Stride Vector Sum:
 * -------------------------------------------------------------------------------------
 * Calculations:
 * blockPerGrid = 24 * 6 = 144 blocks
 * threadsPerBlock: Common rule of thumb is multiples of 32 (32 thread per warp), results in 128, 256 or 512.
 *                  128 threadsPerBlock results in low occupancy (not utilizing all threads)
 *                  256 threadsPerBlock results in the highest occupancy without changing # of blocks
 *                  512 threadsPerBlock results in 3 blocks per SM, but would require changing # of blocks.
 * 
 * 
 * With this configuration:
 *  - 1,536 threads per SM
 *  - 49,152 registers per SM (32 registers per thread for this kernel, from NVIDIA Nsight)
 *  - 36,864 threads total on GPU 
 *  - 1,179,648 registers total on GPU
 * 
 * 
 *  Potential Improvement:
 *  - While we hit the max thread count, we can increase compute by increasing instruction per thread, specifically using ILP.
 *  - Assuming an ILP of two, the register per thread would reach ~32-40, totaling 61,440 registers per SM assuming the highest end, still below the 65,536 register max per SM.
 */
TEST(GridVectorizedVectorSumTest, GridVectorizedVectorSum) {
    int n = 1000000;
    size_t size = n * sizeof(float);

    //Allocate memory on CPU
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Add values to h_a and h_b
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    //Transfer from CPU to GPU
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    //Execute kernel
    gridVectorizedVectorSum<<<144,256>>>(d_a,d_b,d_c,n);

    //Transfer results from GPU to CPU
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    EXPECT_FLOAT_EQ(h_c[0], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n/2], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n-1], 3.0f);
    EXPECT_FLOAT_EQ(h_c[12345], 3.0f);
    EXPECT_FLOAT_EQ(h_c[999999], 3.0f);

    free(h_a);
    free(h_b);
    free(h_c);
}



/**
 * Instructional Level Parallelism, Vectorized (float4) Grid Stride Vector Sum 
 * -------------------------------------------------------------------------------------
 * Calculations:
 * blockPerGrid = 24 * 6 = 144 blocks
 * threadsPerBlock: Common rule of thumb is multiples of 32 (32 thread per warp), results in 128, 256 or 512.
 *                  128 threadsPerBlock results in low occupancy (not utilizing all threads)
 *                  256 threadsPerBlock results in the highest occupancy without changing # of blocks
 *                  512 threadsPerBlock results in 3 blocks per SM, but would require changing # of blocks.
 * 
 *                  - From my current understanding, more threads per block means more threads are able to communicate when it comes to shared memory.
 *                    Bigger blocks means less granularity where while we have less scheduling overhead/block create overhead, but can result in less blocks to fill when one completes early.
 * 
 * With this configuration:
 *  - 1536 threads per SM
 *  - 55,296 registers per SM (36 registers per thread for this kernel, from NVIDIA Nsight)
 *  - 36,864 threads total for GPU
 *  - 1,327,104 registers total for GPU
 */
TEST(ILPVectorizedGridVectorSumTest, ILPVectorizedGridVectorSum) {
    int n = 1000000;
    size_t size = n * sizeof(float);

    //Allocate memory on CPU
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Add values to h_a and h_b
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    //Transfer from CPU to GPU
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    //Execute kernel
    //ILPVectorizedGridVectorSum<<<144,256>>>(d_a,d_b,d_c,n);

    //Transfer results from GPU to CPU
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    EXPECT_FLOAT_EQ(h_c[0], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n/2], 3.0f);
    EXPECT_FLOAT_EQ(h_c[n-1], 3.0f);
    EXPECT_FLOAT_EQ(h_c[12345], 3.0f);
    EXPECT_FLOAT_EQ(h_c[999999], 3.0f);

    free(h_a);
    free(h_b);
    free(h_c);
}

/**
 * -------------------------------------------------------------------------------------------------
 *                                      SUMMARY OF LEARNING
 * -------------------------------------------------------------------------------------------------
 * 
 * GOAL : Learn CUDA fundamentals by progessively learning techniques to add onto vector sum kernel, treating it as a sandbox.
 * 
 * -------------------------------------------------------------------------------------------------
 *                     TEST_SIZE: All kernels used a test size of n = 1,000,000
 * 
 * Kernel                      | Registers/Thread | Time 
 * ----------------------------|------------------|----------------------
 * Naive                       | 16               |  33.44 µs (61,081 cycles)
 * Grid Stride                 | 26               |  49.70 µs (90,810 cycles)
 * Vectorized (float4)         | 22               |  38.85 µs (71,055 cycles)
 * Grid Stride + Vectorized    | 32               |  43.94 µs (80,263 cycles)
 * Grid Stride + Vec + ILP=2   | 36               |  36.58 µs (66,811 cycles)
 * 
 * 
 * Observations:
 * 
 *  - Naive: The naive performed the best, likely because the number of threads > the number of elements. The naive method shines in this edge case but suffers significantly when in an threads < n scenario where an incomplete result
 *           would be produced.
 * 
 *  - Grid Stride: Grid stride showed both an increase in register pressure by 10 compared to the naive, and a significant increase in the number of cycles to processs n, likely because of the 
 *                 instructional overhead of the for loop and stride variable. The register increase is also likely due to the instructional overhead of the for loop and stride variable. 
 *                 While the naive required kernel required 3907 blocks @ 256 threads each, the use of grid stride allowed for a fixed 144 blocks @ 256 threads, making the grid stride hardware-aware.
 *  
 *                 This grid size was determined based off the GPU's (RTX 4060) hardware where each of the 24 SM's had a maximum block limit of 6 which resulted in a grid size 24 * 6 = 144. 
 *                 Thread count per block was determined using values 128, 256 and 512 (divisble by warp size of 32) where 256 resulted in the closest theoretical thread occupancy without
 *                 passing maximum thread per SM count. While Grid Stride was slower than the naive (+29,729 cycles), the kernel is hardware-aware and will work with any n size and GPU.
 * 
 * 
 *  - Vectorized (float4): Vectorization showed an increase in register pressure by 6 compared to the naive. Vectorization shines by allowing a single thread to process many values at once, with the key difference
 *                         being vectorization allows the kernel to make a single instructional call to the global memory to recieve four values, rather than 4 separate instructional calls for the same number of values. It is also key
 *                         to note a single memory transaction will cause either a 32, 64 or 128 byte transfer from global memory, vectorization takes advantage of this by ensuring all values recieved are useful rather than garbage data.
 *                         While vectorizationn performed slightly poorly than the naive version (+9,974 cycles), I am certain vectorization would outperform the naive when it comes to n*4, where the naive would most likely not have enough
 *                         threads (dependent on hardware thread count) to compute an accurate result.
 * 
 * - Grid Stride + Vectorized: Grid Stride and Vectorization combined in a single kernel resulted in an increase register pressure of 16 compared to the naive. Grid Stride and Vectorization together would likely shine when 
 *                             threads < n significantly, and when needing to increase memory throughput by ensuring cache line transfers have useful data. While Grid Stride and Vectorization together performed poorly (+19,182 cycles), 
 *                             I am certain this kernel would outperform the naive when dealing with significant n values. It is also key to note, when n size is low/medium, using these techniques creates unneeded instructional overhead, 
 *                             resulting in a slower performance compared to the naive as seen here.
 * 
 * Grid Stride + Vectorized + ILP=2 (GS-V-ILP2): The kernel combined in a single kernel resulted in an increase register pressure of 20 compared to the naive. GS-V-ILP2 togther would likely shine when
 *                                   threads < n significantly, and the use of ILP allows for latency hiding where two memory loads are issued per thread, allowing both loads to be enroute while the first is being calculated. 
 *                                   While GS-V-ILP2 performed poorly compared to the naive (+5,730 cycles), this kernel performed better than earlier kernels likely due to decreased iteration count (ILP), re-use of threads (grid stride) and
 *                                   instructional memory call efficiency (4 floats per load).
 * 
 * ------- Note -----------
 * Memory throughput is left out for n = 1,000,000 as the total data size (12 MB) is too small to saturate the memory bus. The kernel launch overhead also plays a significant role in µs at this scale.
 * 
 * 
 * -------------------------------------------------------------------------------------------------
 *                     TEST_SIZE: All kernels used a test size of n = 100,000,000 
 * 
 * Kernel                      | Registers/Thread | Time 
 * ----------------------------|------------------|----------------------
 * Naive                       | 16               |  6.17 ms (9,179,890 cycles)
 * Grid Stride                 | 26               |  7.10 ms (9,607,890 cycles)
 * Vectorized (float4)         | 22               |  5.06 ms (9,264,820 cycles)
 * Grid Stride + Vectorized    | 32               |  5.10 ms (9,324,337 cycles)
 * Grid Stride + Vec + ILP=2   | 36               |  5.11 ms (9,343,578 cycles)     
 * 
 * Observations:
 *  - After comparing all the kernels to the vectorized (float4) kernel (best performance) the biggest difference between these kernels was the increased memory throughput. This suggests the kernel is memory bound.
 *      - Naive: 201.26 GB/s
 *      - Grid Stride: 177.77 GB/s
 *      - Vectorized (float4): 249.49 GB/s
 *      - Grid Stride + Vectorized: 246.86 GB/s
 *      - Grid Stride + Vec + ILP=2: 246.63 GB/s
 * 
 *  - The roofline model confirms each kernel is ~0.08 FLOP/byte, placing them near the diagonal memory roof rather than the horizontal compute roof.
 * 
 *  - Another, much simpler way, I've learned is to analyze the mathematical formula/algorithm itself. Where for vector addition C = A + B, we have the following:
 *      - Two 4 byte reads (A and B)
 *      - One 4 byte write (C)
 *      - One floating point operation (add)
 *    This results in one FLOP per 12 bytes, resulting in a theoretical FLOP/byte of 0.083.
 *   
 *      
 *  - Naive: The naive kernel launches ~390k blocks @ 256 threads, and each thread issues a separate load instruction per element.
 *  
 *  - Grid Stride: The grid stride kernel successfully processes all n elements while using only a fixed 144 blocks @ 256 threads. The increased latency from the naive is likely due 
 *                 to the launch parameter of 144 blocks @ 256 threads not because of the technique itself.
 * 
 *  - Vectorization: Vectorization plays a significant role in optimizing this kernel. This is because vectorization allows each thread to receive 4 float values with just a single load
 *                   instruction. This minimizes the scheduling overhead and achieves the highest memory throughput of 249 GB/s, 94.59% of peak.
 *                  
 * 
 *  - Grid Stride + Vectorization: While vectorization optimizes the kernels memory throughput, grid stride provides an unnecessary overhead when n = 100,000,000, but for larger n's grid stride would allow the kerne to scale efficiently.
 * 
 *  - Grid Stride + Vec + ILP=2: While vectorization optimizes the kernels memory throughput, both ILP and grid stride add unnecessary overhead when n = 100,000,000. For larger n values, grid stride would allow the kernel to scale, and while
 *                               ILP helps with latency hiding, for this memory bound kernel it would most likely not help even at larger n values.
 * 
 * 
 * 
 *  * --------------------------------------------------------------------------------------------------------
 *                     TEST_SIZE: BOTH kernels used a test size of n = 100,000,000
 *                TRIAL COUNT: THREE
 *                           
 * 
 * Kernel                      | Registers/Thread | Time                         | Launch configuration
 * ----------------------------|------------------|------------------------------|----------------------------
 * Naive                       | 16               |  6.13 ms (9,633,646 cycles)  |  <<<389625,256>>> //Required number of blocks to reach 100,000,000 threads
 * Grid Stride                 | 26               |  6.04 ms (9,557,458 cycles)  |  <<<389625,256>>> //Required number of blocks to reach 100,000,000 threads
 *                             |                  |                              |      
 * Naive                       | 16               |  6.38 ms (9,658,163 cycles)  |  <<<389625,256>>> //Required number of blocks to reach 100,000,000 threads
 * Grid Stride                 | 26               |  6.50 ms (9,759,083 cycles)  |  <<<389625,256>>> //Required number of blocks to reach 100,000,000 threads
 *                             |                  |                              |      
 * Naive                       | 16               |  6.11 ms (9,486,617 cycles)  |  <<<389625,256>>> //Required number of blocks to reach 100,000,000 threads
 * Grid Stride                 | 26               |  6.07 ms (9,520,425 cycles)  |  <<<389625,256>>> //Required number of blocks to reach 100,000,000 threads
 * 
 *  - Naive: The naive kernal maintains its same performance as the n = 100,000,000 test prior.
 *  - Grid Stride: The grid stride kernel performs similar to the naive kernel when launch conditions are the same. This supports the earlier hypothesis of the increased latency of grid stride in the prior experiment was due to launch configuration
 *                 and not because of the technique itself. 
 * 
 *                 Throughout the trials, some kernels showed higher cycle counts but lower times. This is due to SM clock frequency variation between runs (e.g., 1.59 GHz vs 1.65 GHz). The GPU dynamically adjusts frequency 
 *                 based on thermal and power conditions. The number of cycles performed by each kernel is independent from its SM clock frequency. The following memory throughput values confirm both kernels perform at a similar level.
 * 
 *                 Trial One Memory Throughput:
 *                  - Naive: 207.67 Gbyte/s
 *                  - Grid:  209.65 Gbyte/s
 *                 Trial Two Memory Throughput:
 *                  - Naive: 199.22 Gbyte/s
 *                  - Grid:  198.06 Gbyte/s
 *                 Trial Three Memory Throughput:
 *                  - Naive: 206.92 Gbyte/s
 *                  - Grid:  208.95 Gbyte/s
 *                 
 *                 
 * 
 *                   
 */