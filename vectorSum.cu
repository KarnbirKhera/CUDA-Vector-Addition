#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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




/*

*
*
*               EXPERIMENT KERNELS BELOW
*
*
*
*/



/*
            TESTING
*/

__global__ void vectorSumWriteOnly(float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = 1.0f;
    }
}

__global__ void vectorSumReadOnly(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        float result = A[i] + B[i];
        // Discard result - prevent compiler from optimizing away reads
        if(result < -999999.0f) {
            C[0] = result;
        }
    }
}

/*
    CACHE HINTING TEST
*/

__global__ void vectorAddPTX_CacheStreamed(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        float a, b, c;

        // Load float A
        // "Load.GlobalMemory.CachedStreaming.Float32 Output, Input (dereferenced)" : "writeOnlyFloat"(Output) : "64bitPointer"(Input)
        asm("ld.global.cs.f32 %0, [%1];" : "=f"(a) : "l"(&A[i]));

        // Load float B
        // "Load.GlobalMemory.CachedStreaming.Float32 Output, Input (dereferenced)" : "writeOnlyFloat"(Output) : "64bitPointer"(Input)
        asm("ld.global.cs.f32 %0, [%1];" : "=f"(b) : "l"(&B[i]));

        // Calculate A + B
        // "Add.Float32 Output, Input1, Input2" : "writeOnlyFloat"(Output) : "float"(input1), "float"(input2);
        asm("add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));

        //Store float C
        // "Store.GlobalMemory.CachedStreaming.Float32, Input1, Input2" :: 64bitPointer(input1), "float"(input2)
        // In PTX, input means data needed before the command functions, and output means if a value has been modified, which 
        // in this case it has not.
        asm("st.global.cs.f32 [%0], %1;" :: "l"(&C[i]), "f"(c));
    }
}


__global__ void vectorAddPTX_LastUse(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        float a, b, c;

        // Load float A
        // "Load.GlobalMemory.CachedStreaming.Float32 Output, Input (dereferenced)" : "writeOnlyFloat"(Output) : "64bitPointer"(Input)
        asm("ld.global.lu.f32 %0, [%1];" : "=f"(a) : "l"(&A[i]));

        // Load float B
        // "Load.GlobalMemory.CachedStreaming.Float32 Output, Input (dereferenced)" : "writeOnlyFloat"(Output) : "64bitPointer"(Input)
        asm("ld.global.lu.f32 %0, [%1];" : "=f"(b) : "l"(&B[i]));

        // Calculate A + B
        // "Add.Float32 Output, Input1, Input2" : "writeOnlyFloat"(Output) : "float"(input1), "float"(input2);
        asm("add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));

        //Store float C
        // "Store.GlobalMemory.CachedStreaming.Float32, Input1, Input2" :: 64bitPointer(input1), "float"(input2)
        // In PTX, input means data needed before the command functions, and output means if a value has been modified, which 
        // in this case it has not.
        asm("st.global.cs.f32 [%0], %1;" :: "l"(&C[i]), "f"(c));
    }
}


/*
    INSTRUCTION LEVEL PARALLELISM ISOLATED 
*/

__global__ void vectorSumILP(float* A, float* B, float* C, int n) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;

    int gridSize = blockDim.x * gridDim.x;

    int i1 = i0 + gridSize;
    int i2 = i0 + gridSize * 2;
    int i3 = i0 + gridSize * 3;
    
    if(i0 < n) {
        float a0 = A[i0];
        float a1 = A[i1];
        float a2 = A[i2];
        float a3 = A[i3];

        float b0 = B[i0];
        float b1 = B[i1];
        float b2 = B[i2];
        float b3 = B[i3];

        C[i0] = a0 + b0;
        C[i1] = a1 + b1;
        C[i2] = a2 + b2;
        C[i3] = a3 + b3;
    }
}



/*
    GRID STRIDE ILP=2
*/

__global__ void vectorSumGridILP2(float* A, float* B, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride * 2) {

        int i0 = i;
        int i1 = i + stride;

        // First ILP lane
        float a0 = A[i0];
        float b0 = B[i0];
        C[i0] = a0 + b0;


        // Second ILP lane
        float a1 = A[i1];
        float b1 = B[i1];
        C[i1] = a1 + b1;
    }
}



/*
    GRID STRIDE ILP=4
*/

__global__ void vectorSumGridILP4(float* A, float* B, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Increase the loop jump to stride * 4 to account for the 4 elements per thread
    for (int i = tid; i < n; i += stride * 4) {

        // Calculate four strided indices to maintain coalescing
        int i0 = i;
        int i1 = i + stride;
        int i2 = i + stride * 2;
        int i3 = i + stride * 3;

        // Load Phase: Issuing 4 independent loads to fill the memory pipe
        float a0 = A[i0];
        float a1 = A[i1];
        float a2 = A[i2];
        float a3 = A[i3];

        float b0 = B[i0];
        float b1 = B[i1];
        float b2 = B[i2];
        float b3 = B[i3];

        // Compute and Store Phase
        C[i0] = a0 + b0;
        C[i1] = a1 + b1;
        C[i2] = a2 + b2;
        C[i3] = a3 + b3;
    }
}


/*
    VECTORIZATION without padding
*/

__global__ void vectorSumVectorizedNoPadding(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    if(i < n) {

        float4 a = A4[i];
        float4 b = B4[i];

        C4[i] = make_float4(
            a.x + b.x, 
            a.y + b.y, 
            a.z + b.z, 
            a.w + b.w
        );
    }
} 


/*
    L2 CACHE EXPERIMENT, COALESCED VS UNCOALESCED 
*/

__global__ void writeOnlyCoalesced(float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = 1.0f;
    }
}

__global__ void writeOnlyUncoalesced(float* C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i < n) {
        C[i] = 1.0f;
    }
}


/*
 NAIVE FP8
*/


__global__ void vectorAddFP8(const float* A, const float* B, __nv_fp8_e4m3* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // 1. Pull 32-bit floats from DRAM
        float a_val = A[i];
        float b_val = B[i];

        // 2. Compute in FP32
        float res = a_val + b_val;

        // 3. Cast to FP8 and write 8-bit value to DRAM
        C[i] = __nv_fp8_e4m3(res);
    }
}