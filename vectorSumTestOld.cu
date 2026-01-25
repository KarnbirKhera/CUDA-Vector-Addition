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

struct KernelResult {
    const char* name;
    float mean;
    float stddev;
};

KernelResult results[7];

TEST(Small_N_Test, BenchmarkAllKernels_SMALL) {

//                       ----------------------------------------------------------------------
//                       ------------------------- TEST CONFIGURATION -------------------------
//                       ----------------------------------------------------------------------

    int n = 10000000;
    const int NUM_OF_TRIALS = 10;
    const int THREADS = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


//                      ----------------------------------------------------------------------
//                      -------------------------     ALLOCATION     -------------------------
//                      ----------------------------------------------------------------------

    //Allocate host memory
    size_t size = n * sizeof(float);
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Intialize values in host memory
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory and transfer values from Host to Device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b, size, cudaMemcpyHostToDevice);

    //CudaEvent timer to accurately time kernels rather than relying on variable wall clock times
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Variables re-used throughout runs
    float times[NUM_OF_TRIALS];
    float sum, mean, sum_sq_diff, stddev;
    int blocks;

//                      --------------------------------------------------------------------------
//                      -------------------------         TESTS          -------------------------
//                      --------------------------------------------------------------------------
    

//========================== NAIVE ==========================

    // Intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    // Allocate correct number of blocks required to ensure thread to element ratio is 1:1
    blocks = (n + THREADS - 1) / THREADS;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[0] = {"NAIVE", mean, stddev};

//========================== GRID STRIDE ==========================
    
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    //Find number of SMs in GPU, as well as the number of blocks each SM can run to calculate optimal grid stride block per grid count (Hardware-aware).
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        gridStrideVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        gridStrideVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[1] = {"GRID STRIDE", mean, stddev};

//========================== VECTORIZED ==========================

    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    // Allocate correct number of blocks required to ensure thread to element ratio is 1:4
    blocks = ((n + 3) / 4 + THREADS - 1) / THREADS; // (n + 3) for n < 4 edge case, divided by 4 to match vectorization

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorizedVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorizedVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[2] = {"VECTORIZED", mean, stddev};

//========================== VECTORIZED + GRID STRIDE ==========================

    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    //Find number of SMs in GPU, as well as the number of blocks each SM can run to calculate optimal grid stride block per grid count (Hardware-aware).
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        gridVectorizedVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        gridVectorizedVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[3] = {"VEC + GRID", mean, stddev};

//========================== VECTORIZED + GRID STRIDE + ILP=2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILPVectorizedGridVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        ILPVectorizedGridVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        ILPVectorizedGridVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[4] = {"VEC + GRID + ILP=2", mean, stddev};

//========================== ILP = 2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */

    //ILP = 2
    blocks = ((n + 1) / 2 + THREADS - 1) / THREADS;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        ILPVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        ILPVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[5] = {"ILP=2", mean, stddev};


//========================== Vectorization + ILP = 2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */

    //ILP = 2, Vectorization float4
    blocks = ((n + 7) / 8 + THREADS - 1) / THREADS; 

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorizedILPVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorizedILPVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[6] = {"Vectorization + ILP=2", mean, stddev};


//                      --------------------------------------------------------------------------
//                      -------------------------         RESULTS        -------------------------
//                      --------------------------------------------------------------------------
    printf("\n================= RESULTS (10M elements) =========================\n");

    printf("%-20s %.4f ms  (+/- %.4f)\n", results[0].name, results[0].mean, results[0].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[1].name, results[1].mean, results[1].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[2].name, results[2].mean, results[2].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[3].name, results[3].mean, results[3].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[4].name, results[4].mean, results[4].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[5].name, results[5].mean, results[5].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[6].name, results[6].mean, results[6].stddev);
    printf(" ");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



TEST(Medium_N_Test, BenchmarkAllKernels_Medium) {

//                       ----------------------------------------------------------------------
//                       ------------------------- TEST CONFIGURATION -------------------------
//                       ----------------------------------------------------------------------

    int n = 100000000;
    const int NUM_OF_TRIALS = 10;
    const int THREADS = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


//                      ----------------------------------------------------------------------
//                      -------------------------     ALLOCATION     -------------------------
//                      ----------------------------------------------------------------------

    //Allocate host memory
    size_t size = n * sizeof(float);
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Intialize values in host memory
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory and transfer values from Host to Device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b, size, cudaMemcpyHostToDevice);

    //CudaEvent timer to accurately time kernels rather than relying on variable wall clock times
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Variables re-used throughout runs
    float times[NUM_OF_TRIALS];
    float sum, mean, sum_sq_diff, stddev;
    int blocks;

//                      --------------------------------------------------------------------------
//                      -------------------------         TESTS          -------------------------
//                      --------------------------------------------------------------------------
    

//========================== NAIVE ==========================

    // Intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    // Allocate correct number of blocks required to ensure thread to element ratio is 1:1
    blocks = (n + THREADS - 1) / THREADS;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[0] = {"NAIVE", mean, stddev};

//========================== GRID STRIDE ==========================
    
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    //Find number of SMs in GPU, as well as the number of blocks each SM can run to calculate optimal grid stride block per grid count (Hardware-aware).
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        gridStrideVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        gridStrideVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[1] = {"GRID STRIDE", mean, stddev};

//========================== VECTORIZED ==========================

    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    // Allocate correct number of blocks required to ensure thread to element ratio is 1:4
    blocks = ((n + 3) / 4 + THREADS - 1) / THREADS; // (n + 3) for n < 4 edge case, divided by 4 to match vectorization

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorizedVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorizedVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[2] = {"VECTORIZED", mean, stddev};

//========================== VECTORIZED + GRID STRIDE ==========================

    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    //Find number of SMs in GPU, as well as the number of blocks each SM can run to calculate optimal grid stride block per grid count (Hardware-aware).
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        gridVectorizedVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        gridVectorizedVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[3] = {"VEC + GRID", mean, stddev};

//========================== VECTORIZED + GRID STRIDE + ILP=2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILPVectorizedGridVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        ILPVectorizedGridVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        ILPVectorizedGridVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[4] = {"VEC + GRID + ILP=2", mean, stddev};

//========================== ILP = 2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */

    //ILP = 2
    blocks = ((n + 1) / 2 + THREADS - 1) / THREADS;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        ILPVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        ILPVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[5] = {"ILP=2", mean, stddev};


//========================== Vectorization + ILP = 2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */

    //ILP = 2, Vectorization float4
    blocks = ((n + 7) / 8 + THREADS - 1) / THREADS; 

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorizedILPVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorizedILPVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[6] = {"Vectorization + ILP=2", mean, stddev};


//                      --------------------------------------------------------------------------
//                      -------------------------         RESULTS        -------------------------
//                      --------------------------------------------------------------------------
    printf("\n================= RESULTS (100M elements) =========================\n");

    printf("%-20s %.4f ms  (+/- %.4f)\n", results[0].name, results[0].mean, results[0].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[1].name, results[1].mean, results[1].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[2].name, results[2].mean, results[2].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[3].name, results[3].mean, results[3].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[4].name, results[4].mean, results[4].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[5].name, results[5].mean, results[5].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[6].name, results[6].mean, results[6].stddev);
    printf(" ");


    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}




TEST(Large_N_Test, BenchmarkAllKernels_Large) {

//                       ----------------------------------------------------------------------
//                       ------------------------- TEST CONFIGURATION -------------------------
//                       ----------------------------------------------------------------------

    int n = 200000000;
    const int NUM_OF_TRIALS = 10;
    const int THREADS = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


//                      ----------------------------------------------------------------------
//                      -------------------------     ALLOCATION     -------------------------
//                      ----------------------------------------------------------------------

    //Allocate host memory
    size_t size = n * sizeof(float);
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    //Intialize values in host memory
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Allocate memory and transfer values from Host to Device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b, size, cudaMemcpyHostToDevice);

    //CudaEvent timer to accurately time kernels rather than relying on variable wall clock times
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Variables re-used throughout runs
    float times[NUM_OF_TRIALS];
    float sum, mean, sum_sq_diff, stddev;
    int blocks;

//                      --------------------------------------------------------------------------
//                      -------------------------         TESTS          -------------------------
//                      --------------------------------------------------------------------------
    

//========================== NAIVE ==========================

    // Intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    // Allocate correct number of blocks required to ensure thread to element ratio is 1:1
    blocks = (n + THREADS - 1) / THREADS;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[0] = {"NAIVE", mean, stddev};

//========================== GRID STRIDE ==========================
    
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    //Find number of SMs in GPU, as well as the number of blocks each SM can run to calculate optimal grid stride block per grid count (Hardware-aware).
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        gridStrideVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        gridStrideVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[1] = {"GRID STRIDE", mean, stddev};

//========================== VECTORIZED ==========================

    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    // Allocate correct number of blocks required to ensure thread to element ratio is 1:4
    blocks = ((n + 3) / 4 + THREADS - 1) / THREADS; // (n + 3) for n < 4 edge case, divided by 4 to match vectorization

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorizedVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorizedVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);


    results[2] = {"VECTORIZED", mean, stddev};

//========================== VECTORIZED + GRID STRIDE ==========================

    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    //Find number of SMs in GPU, as well as the number of blocks each SM can run to calculate optimal grid stride block per grid count (Hardware-aware).
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        gridVectorizedVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        gridVectorizedVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[3] = {"VEC + GRID", mean, stddev};

//========================== VECTORIZED + GRID STRIDE + ILP=2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILPVectorizedGridVectorSum, THREADS, 0);
    blocks = prop.multiProcessorCount * blocks_per_sm;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        ILPVectorizedGridVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        ILPVectorizedGridVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[4] = {"VEC + GRID + ILP=2", mean, stddev};



//========================== ILP = 2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */

    //ILP = 2
    blocks = ((n + 1) / 2 + THREADS - 1) / THREADS;

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        ILPVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        ILPVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[5] = {"ILP=2", mean, stddev};


//========================== Vectorization + ILP = 2 ==========================
    // Re-intialize required variables
    sum = 0;
    sum_sq_diff = 0;

    /* 
    * Calculate optimal block count for full GPU occupancy.
    * cudaOccupancyMaxActiveBlocksPerMultiprocessor accounts for
    * register pressure, thread limits, and shared memory (not used here).
    */

    //ILP = 2, Vectorization float4
    blocks = ((n + 7) / 8 + THREADS - 1) / THREADS; 

    // Run kernel before testing to avoid inaccurate cold start numbers
    for(int i = 0; i < 2; i++) {
        vectorizedILPVectorSum<<<blocks, THREADS>>>(d_a,d_b,d_c,n);
    }
    cudaDeviceSynchronize();

    // Run and time trials
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        cudaEventRecord(start);
        vectorizedILPVectorSum<<<blocks,THREADS>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Post trial calculations
    // Mean
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        sum += times[i];
    }
    mean = sum / NUM_OF_TRIALS;

    //Standard deviation
    for(int i = 0; i < NUM_OF_TRIALS; i++) {
        float diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    stddev = sqrt(sum_sq_diff/NUM_OF_TRIALS);

    results[6] = {"Vectorization + ILP=2", mean, stddev};


//                      --------------------------------------------------------------------------
//                      -------------------------         RESULTS        -------------------------
//                      --------------------------------------------------------------------------
    printf("\n================= RESULTS (200M elements) =========================\n");

    printf("%-20s %.4f ms  (+/- %.4f)\n", results[0].name, results[0].mean, results[0].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[1].name, results[1].mean, results[1].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[2].name, results[2].mean, results[2].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[3].name, results[3].mean, results[3].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[4].name, results[4].mean, results[4].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[5].name, results[5].mean, results[5].stddev);
    printf("%-20s %.4f ms  (+/- %.4f)\n", results[6].name, results[6].mean, results[6].stddev);
    printf(" ");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


TEST(Correctness, AllKernels) {
    std::vector<int> test_sizes = {1, 3, 4, 5, 255, 256, 257, 1024, 10000};
    const int THREADS = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for (int n : test_sizes) {
        size_t size = n * sizeof(float);

        // Allocate host memory
        float* h_a = (float*)malloc(size);
        float* h_b = (float*)malloc(size);
        float* h_c = (float*)malloc(size);

        // Initialize values
        for (int i = 0; i < n; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
        }

        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        // Lambda to test a kernel
        auto testKernel = [&](const char* name, auto launchKernel) {
            cudaMemset(d_c, 0, size);
            launchKernel();
            cudaDeviceSynchronize();
            ASSERT_EQ(cudaGetLastError(), cudaSuccess) << name << " failed at n=" << n;

            cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
            EXPECT_FLOAT_EQ(h_c[0], 3.0f) << name << " failed at n=" << n << " idx=0";
            EXPECT_FLOAT_EQ(h_c[n-1], 3.0f) << name << " failed at n=" << n << " idx=n-1";
            if (n > 2) {
                for (int i = 0; i < n; i++) { EXPECT_FLOAT_EQ(h_c[i], 3.0f) << name << " failed at n=" << n << " idx=" << i; }
            }
        };

        int blocks = (n + THREADS - 1) / THREADS;
        int blocks_per_sm;

        // Naive
        testKernel("Naive", [&]() {
            vectorSum<<<blocks, THREADS>>>(d_a, d_b, d_c, n);
        });

        // Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
        testKernel("GridStride", [&]() {
            gridStrideVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

        // Vectorized
        testKernel("Vectorized", [&]() {
            int vec_blocks = ((n + 3) / 4 + THREADS - 1) / THREADS;
            vectorizedVectorSum<<<vec_blocks, THREADS>>>(d_a, d_b, d_c, n);
        });

        // Vectorized + Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
        testKernel("VectorizedGridStride", [&]() {
            gridVectorizedVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

        // ILP + Vectorized + Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILPVectorizedGridVectorSum, THREADS, 0);
        testKernel("VectorizedGridILP=2", [&]() {
            ILPVectorizedGridVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

        testKernel("ILP=2", [&]() {
            int ilp_blocks = ((n + 1) / 2 + THREADS - 1) / THREADS;
            ILPVectorSum<<<ilp_blocks, THREADS>>>(d_a, d_b, d_c, n);
        });

        testKernel("VectorizedILP=2", [&]() {
            int ilp_blocks = ((n + 3) / 8 + THREADS - 1) / THREADS;
            vectorizedILPVectorSum<<<ilp_blocks, THREADS>>>(d_a, d_b, d_c, n);
        });

        // Cleanup
        free(h_a);
        free(h_b);
        free(h_c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
}

TEST(Profile, SingleRun) {
    int n = 200000000;
    size_t size = n * sizeof(float);
    const int THREADS = 256;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *d_a, *d_b, *d_c;
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    int blocks_per_sm;
    
    // Naive - 1 thread : 1 element
    int naive_blocks = (n + THREADS - 1) / THREADS;
    vectorSum<<<naive_blocks, THREADS>>>(d_a, d_b, d_c, n);
    
    // Grid Stride - hardware-aware block count
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
    int grid_blocks = prop.multiProcessorCount * blocks_per_sm;
    gridStrideVectorSum<<<grid_blocks, THREADS>>>(d_a, d_b, d_c, n);
    
    // Vectorized - 1 thread : 4 elements
    int vec_blocks = ((n + 3) / 4 + THREADS - 1) / THREADS;
    vectorizedVectorSum<<<vec_blocks, THREADS>>>(d_a, d_b, d_c, n);

    
    // Vectorized + Grid Stride - hardware-aware
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
    int gridvec_blocks = prop.multiProcessorCount * blocks_per_sm;
    gridVectorizedVectorSum<<<gridvec_blocks, THREADS>>>(d_a, d_b, d_c, n);
    
    // ILP + Vectorized + Grid Stride - hardware-aware
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILPVectorizedGridVectorSum, THREADS, 0);
    int gridvecilp_blocks = prop.multiProcessorCount * blocks_per_sm;
    ILPVectorizedGridVectorSum<<<gridvecilp_blocks, THREADS>>>(d_a, d_b, d_c, n);

    // ILP=2 - 1 thread : 2 elements
    int ilp_blocks = ((n + 1) / 2 + THREADS - 1) / THREADS;
    ILPVectorSum<<<ilp_blocks, THREADS>>>(d_a, d_b, d_c, n);

    // Vectorized + ILP=2 - 1 thread : 8 elements
    int vec_ilp_blocks = ((n + 7) / 8 + THREADS - 1) / THREADS;
    vectorizedILPVectorSum<<<vec_ilp_blocks, THREADS>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}