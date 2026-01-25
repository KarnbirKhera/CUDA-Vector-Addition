#include <iostream>
#include <cuda_runtime.h>
/**
 * Naive Vector Sum Kernel:
 * --------------------------------------------------
 * Mechanism: Maps 1 thread to one vector sum, in a 1 thread to 1 element ratio.
 * Memory Analysis: Maintains memory coalescing as warp threads access neighboring memory. Thread is retired when local sum is computed.
*  Use Case: Small datasets where one thread per element provides full parallelism without loop overhead.
 * Constraints: 
 *  - Limited by Grid Size: If threads < n, the kernel will produce an incomplete sum as not enough threads in grid (determined at kernel launch <<<blocksPerGrid,threadsPerBlock>>>) 
 *                          to calculate sum of all elements.
 */
__global__ void vectorSum(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //Local Thread ID
    if(i < n) {
        C[i] = A[i] + B[i];
    }
}
/**
 * Grid Stride Vector Sum Kernel:
 * --------------------------------------------------
 * Mechanism: Threads are persistent and reused in grid to compute complete sum of n, regardless of n size (scalability).
 * Memory Analysis: Maintains memory coalescing as warp threads access neighboring memory. The use of stride allows for re-use of threads across n. Less thread initialization overhead due to thread reuse (unlike naive version)
 * Use Case: 
 *  - Kernel needs to run on multiple different GPUs with different constraints (hardware-aware)
 *  - Unpredictable n size
 * Constraints: 
*   - Warp Divergence: Tail warp encounters divergence and idle threads when some threads do not satisfy i < n, results in minimal performance impact.
 */
__global__ void gridStrideVectorSum(float* A, float* B, float* C, int n)  {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x; //Local Thread ID
    int stride = blockDim.x * gridDim.x;

    for(int i = globalId; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}
/**
 * Vectorized (float4) Vector Sum Kernel:
 * --------------------------------------------------
 * Mechanism: reinterpets float* (4 bytes) input as float4* (16 bytes) to compute the sum of 4 floats per thread. Uses else block to process tail elements if n is not divisible by 4.
 * Memory Analysis: Effectively reduces memory overhead by calling for 4 floats at once, rather than 4 separate calls to global memory. The use of float4 allows every call to global memory to fully saturate
 * L1 cache line (128 bytes), increasing transactional density.
 * using float4 allows the 128 to be used fully, increasing transaction density.
 * Use Case: 
 *  - Medium datasets where threads > n/4
 *  - Useful for memory bound kernels
 * Constraints:
 *  - Large datasets (threads < n/4): If the number of elements is four times more than the number of threads available in the grid, the kernel will produce an incomplete sum.
 *  - Memory Alignment: Requires memory alignment where starting address of float* must be 16 byte aligned allowing a proper float4* reinterpret_cast
 */
__global__ void vectorizedVectorSum(float* A, float* B, float* C, int n)  {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x; //Local Thread ID
    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    
    int n4 = n / 4;
    if(globalId < n4) {
        float4 a = A4[globalId];
        float4 b = B4[globalId];
        C4[globalId] = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
    }

    if (globalId == 0) {
        int start = n4 * 4;
        for (int i = start; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
}


/**
 * Vectorized (float4) Grid Stride Vector Sum:
 * --------------------------------------------------
 * Mechanism: Combines float4 reinterpret_cast with grid stride loop. Allows one thread to process 4 floats at once, and can process efficiently if thread < n.
 * Memory Analysis: 
 *  - Vectorization: Increases memory throughput by calling 4 floats using a single instructional command, rather than 4 separate instructional calls, 
 *                   decreasing the instructional overhead and allowing for other operations to take its place.
 *  - Grid Stride: Allows the re-use of threads to process all n elements (scalable). 
 * Use case: 
 *  - Large n size, as well as unpredictable n size (Grid Stride)
 *  - Useful for memory bound kernels
 * Constraints:
 *  - Vectorization requires tail end logic to process elements if n is not divisible by 4, very small overhead.
 *  - Memory Alignment: Requires memory alignment where starting address of float* must be 16 byte aligned allowing a proper float4* reinterpret_cast
 */
__global__ void gridVectorizedVectorSum(float* A, float* B, float* C, int n)  {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x; //Local Thread ID
    int stride = blockDim.x * gridDim.x; //Stride, units of work per thread

    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    int n4 = n/4; 

    for(int vector = globalId; vector < n4; vector += stride) {
        float4 a = A4[vector];
        float4 b = B4[vector];
        C4[vector] = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
    }

    if (globalId == 0) {
        int start = n4 * 4;
        for (int i = start; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

/**
 * Instructional Level Parallelism, Vectorized (float4) Grid Stride Vector Sum 
 * Mechanism: ILP allows for consecutive load calls to global memory to overlap to hide latency. Vectorization allows for more memory throughput, while grid stride allows the kernel to be
 *            hardware-aware. 
 * Memory Analysis: 
 *  - ILP: ILP allows for consecutive load calls to global memory allowing overlap to hide latency.
 *  - Vectorization: Increases memory throughput by calling 4 floats using a single instructional command, rather than 4 separate instructional calls, 
 *                   decreasing the instructional overhead and allowing for other operations to take its place.
 *  - Grid Stride: Allows the re-use of threads to process all n elements (scalable). 
 * Use case: 
 *  - Memory bound kernels at any scale where bandwidth saturation is needed
 *  - Datasets that require both performance and hardware compatibility
 * Constraints:
 *  - Vectorization requires tail end logic to process elements if n is not divisible by 4
 *  - Memory Alignment: Requires memory alignment where starting address of float* must be 16 byte aligned allowing a proper float4* reinterpret_cast
 *  - ILP: Increases register pressure, can reduce occupancy introducing additional overhead hiding benefits of latency hiding. Increased register pressure can also cause register
 *         spill into local memory, slowing performance significantly.
 */
__global__ void ILP2VectorizedGridVectorSum(float* A, float* B, float* C, int n)  {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int ILP = 2;

    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    int vector = globalId;
    int n4 = n/4;

    for(; vector + stride < n4; vector += stride * ILP) {
        int vector2 = vector + stride;

        float4 a = A4[vector];
        float4 b = B4[vector];

        float4 a2 = A4[vector2];
        float4 b2 = B4[vector2];

        C4[vector] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

        C4[vector2] = make_float4(a2.x + b2.x, a2.y + b2.y, a2.z + b2.z, a2.w + b2.w);
    }


    for (; vector < n4; vector += stride) {
        float4 a = A4[vector];
        float4 b = B4[vector];
        C4[vector] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }


    for (int idx = n4 * 4 + globalId; idx < n; idx += stride) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void ILP4VectorizedGridVectorSum(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int ILP = 4;

    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    int vector = i;
    int n4 = n/4;

    for(; vector + stride * 3 < n4; vector += stride * ILP) {
        int vector2 = vector + stride;
        int vector3 = vector + stride * 2;
        int vector4 = vector + stride * 3;

        
        float4 a = A4[vector];
        float4 b = B4[vector];

        float4 a2 = A4[vector2];
        float4 b2 = B4[vector2];
        
        float4 a3 = A4[vector3];
        float4 b3 = B4[vector3];

        float4 a4 = A4[vector4];
        float4 b4 = B4[vector4];


        C4[vector] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

        C4[vector2] = make_float4(a2.x + b2.x, a2.y + b2.y, a2.z + b2.z, a2.w + b2.w); 

        C4[vector3] = make_float4(a3.x + b3.x, a3.y + b3.y, a3.z + b3.z, a3.w + b3.w); 

        C4[vector4] = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w); 
    }

    for(; vector < n4; vector += stride) {
        float4 a = A4[vector];
        float4 b = B4[vector];

        C4[vector] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    int start = n4 * 4 + i;
    for(int z = start; z < n; z += stride) {
        C[z] = A[z] + B[z];
    }
}





