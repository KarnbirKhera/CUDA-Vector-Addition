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
    int i = blockIdx.x * blockDim.x + threadIdx.x; // [block] * [thread/block] + [thread] = [thread]
    if(i * 1 < n) { // element < element. While compiler implicitly multiplies i to allow thread to element comparison, for the sake of learning Dimensional Analysis, it has been helpful to explicitly write.
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
__global__ void gridStrideVectorSum(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // [block] * [thread/block] + [thread] = [thread]
    int idx = i * 1; // [thread] * 1[element/thread] = [element]
    int stride = gridDim.x * blockDim.x * 1; // [block/grid] * [thread/block] * 1[element/thread]= [element/grid]
    
    //While compiler implicitly includes * 1 for thread to element comparison, for learning purposes included explicitly.
    
    for(; idx < n; idx += stride) { // element < element; element += [element/grid].
        C[idx] = A[idx] + B[idx];
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
__global__ void vectorizedVectorSum(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // [block] * [thread/block] + [thread] = [thread]
    int idx = i * 4; // [thread] * 4[element/thread] = 4[element]

    int n_4 = (n/4) * 4; // floor division using integer truncation to match float4, avoids if(idx + 3 < n) check implicitly. Needs tail handling.

    if(idx < n_4) {
        float4 a = *reinterpret_cast<float4*>(A + idx);
        float4 b = *reinterpret_cast<float4*>(B + idx);
        float4 result; //Avoid reinterpret_cast early to avoid reading garbage values

        result.x = a.x + b.x;
        result.y = a.y + b.y;
        result.z = a.z + b.z;
        result.w = a.w + b.w;

        float4* c_ptr = reinterpret_cast<float4*>(C + idx);
        *c_ptr = result;
    }

    if(i == 0) { //Tail handling, only first thread needed.
        for(int z = n_4; z < n; z++) { // element = element; element < element; element++
            C[z] = A[z] + B[z];
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
__global__ void gridVectorizedVectorSum(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // [block] * [thread/block] + [thread] = [thread], localID
    int idx = i * 4; // [thread] * 4[element/thread] = 4[element], vectorID
    int stride = gridDim.x * blockDim.x * 4;//[block/grid] * [thread/block] * 4[element/thread] = 4[element/grid] 

    int n_4 = (n/4) * 4; // floor division using integer truncation to match float4, avoids if(idx + 3 < n) check implicitly. Needs tail handling.

    for(; idx < n_4; idx += stride) { // 4[element] < element; 4[element] += 4[element/grid]
        float4 a = *reinterpret_cast<float4*>(A + idx);
        float4 b = *reinterpret_cast<float4*>(B + idx);
        float4 result; //Avoid reinterpret_cast early to avoid reading garbage values

        result.x = a.x + b.x;
        result.y = a.y + b.y;
        result.z = a.z + b.z;
        result.w = a.w + b.w;

        float4* c = reinterpret_cast<float4*>(C + idx);
        *c = result;
    }

    if(i == 0) { //Thread 0 only, handle tail elements
        for(int z = n_4; z < n; z++) { // element = element; element < element; element++
            C[z] = A[z] + B[z];
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

 __global__ void ILPVectorizedGridVectorSum(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int ILP = 8;

    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    int n4 = n/4;

    for(int vector = i; vector < n4; vector += stride * ILP) {
        int vector2 = vector + blockDim.x;
        int vector3 = vector + blockDim.x * 2;
        int vector4 = vector + blockDim.x * 3;

        bool vec2Valid = vector2 < n4;
        bool vec3Valid = vector3 < n4;
        bool vec4Valid = vector4 < n4;

        float4 a, b, a2, b2, a3, b3, a4, b4;
        
        a = A4[vector];
        b = B4[vector];

        if(vec2Valid) {
            a2 = A4[vector2];
            b2 = B4[vector2];
        }

        if(vec3Valid) {
            a3 = A4[vector3];
            b3 = B4[vector3];
        }

        if(vec4Valid) {
            a4 = A4[vector4];
            b4 = B4[vector4];
        }



        C4[vector] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

        if(vec2Valid) {
           C4[vector2] = make_float4(a2.x + b2.x, a2.y + b2.y, a2.z + b2.z, a2.w + b2.w); 
        }

        if(vec3Valid) {
            C4[vector3] = make_float4(a3.x + b3.x, a3.y + b3.y, a3.z + b3.z, a3.w + b3.w); 
        }

        if(vec4Valid) {
            C4[vector4] = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w); 
        }
    }

    if(i == 0) {
        int start = n4 * 4;
        for(int z = start; z < n; z++) {
            C[z] = A[z] + B[z];
        }
    }
 }


