#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include "vectorSumHeader.cuh"

/*
* On current GPU (RTX 4060):
*  - 24 SMs
*  - 6 max resident blocks per SM
*  - 1,536 max resident threads per SM
*  - 65,536 max registers per SM
*  - 36,864 max threads for GPU
*  - 1,572,864 max registers for GPU
*/

struct KernelResult {
    const char* name;
    int n;
    float time_mean;
    float time_stddev;
    float throughput_mean;  // GB/s
    float throughput_stddev;
    float efficiency;       // % of peak bandwidth
    float efficiency_stddev;
};

const float PEAK_BANDWIDTH_GBS = 272.0f;  // RTX 4060 theoretical peak

std::vector<KernelResult> allResults;  // Store all results for CSV export

// Helper function to run benchmark for a kernel
template<typename KernelFunc>
KernelResult runBenchmark(const char* name, KernelFunc launchKernel, int n, int numTrials) {
    float times[numTrials];
    float throughputs[numTrials];
    
    // Total bytes: 2 reads + 1 write = 3 * n * sizeof(float)
    size_t total_bytes = (size_t)n * 3 * sizeof(float);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        launchKernel();
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Timed runs
    for (int i = 0; i < numTrials; i++) {
        cudaEventRecord(start);
        launchKernel();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
        
        // Calculate throughput for this run (ms to seconds, bytes to GB)
        float time_seconds = times[i] / 1000.0f;
        throughputs[i] = (total_bytes / time_seconds) / 1e9f;
    }
    
    // Calculate time mean
    float time_sum = 0;
    for (int i = 0; i < numTrials; i++) {
        time_sum += times[i];
    }
    float time_mean = time_sum / numTrials;
    
    // Calculate time stddev
    float time_sum_sq_diff = 0;
    for (int i = 0; i < numTrials; i++) {
        float diff = times[i] - time_mean;
        time_sum_sq_diff += diff * diff;
    }
    float time_stddev = sqrt(time_sum_sq_diff / numTrials);
    
    // Calculate throughput mean
    float throughput_sum = 0;
    for (int i = 0; i < numTrials; i++) {
        throughput_sum += throughputs[i];
    }
    float throughput_mean = throughput_sum / numTrials;
    
    // Calculate throughput stddev
    float throughput_sum_sq_diff = 0;
    for (int i = 0; i < numTrials; i++) {
        float diff = throughputs[i] - throughput_mean;
        throughput_sum_sq_diff += diff * diff;
    }
    float throughput_stddev = sqrt(throughput_sum_sq_diff / numTrials);
    
    // Calculate efficiency and its stddev
    float efficiency = (throughput_mean / PEAK_BANDWIDTH_GBS) * 100.0f;
    float efficiency_stddev = (throughput_stddev / PEAK_BANDWIDTH_GBS) * 100.0f;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return {name, n, time_mean, time_stddev, throughput_mean, throughput_stddev, efficiency, efficiency_stddev};
}

void runAllBenchmarks(int n, const char* sizeLabel) {
    const int NUM_OF_TRIALS = 10;
    const int THREADS = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Allocation
    size_t size = n * sizeof(float);
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blocks_per_sm;
    KernelResult results[6];

    // 1. NAIVE
    int naive_blocks = (n + THREADS - 1) / THREADS;
    results[0] = runBenchmark("Naive", [&]() {
        vectorSum<<<naive_blocks, THREADS>>>(d_a, d_b, d_c, n);
    }, n, NUM_OF_TRIALS);

    // 2. GRID STRIDE
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
    int grid_blocks = prop.multiProcessorCount * blocks_per_sm;
    results[1] = runBenchmark("Grid Stride", [&]() {
        gridStrideVectorSum<<<grid_blocks, THREADS>>>(d_a, d_b, d_c, n);
    }, n, NUM_OF_TRIALS);

    // 3. VECTORIZED
    int vec_blocks = ((n + 3) / 4 + THREADS - 1) / THREADS;
    results[2] = runBenchmark("Vectorized", [&]() {
        vectorizedVectorSum<<<vec_blocks, THREADS>>>(d_a, d_b, d_c, n);
    }, n, NUM_OF_TRIALS);

    // 4. VEC + GRID
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
    int gridvec_blocks = prop.multiProcessorCount * blocks_per_sm;
    results[3] = runBenchmark("Vec + Grid", [&]() {
        gridVectorizedVectorSum<<<gridvec_blocks, THREADS>>>(d_a, d_b, d_c, n);
    }, n, NUM_OF_TRIALS);

    // 5. VEC + GRID + ILP=2
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILP2VectorizedGridVectorSum, THREADS, 0);
    int ilp2_blocks = prop.multiProcessorCount * blocks_per_sm;
    results[4] = runBenchmark("Vec + Grid + ILP=2", [&]() {
        ILP2VectorizedGridVectorSum<<<ilp2_blocks, THREADS>>>(d_a, d_b, d_c, n);
    }, n, NUM_OF_TRIALS);

    // 6. VEC + GRID + ILP=4
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILP4VectorizedGridVectorSum, THREADS, 0);
    int ilp4_blocks = prop.multiProcessorCount * blocks_per_sm;
    results[5] = runBenchmark("Vec + Grid + ILP=4", [&]() {
        ILP4VectorizedGridVectorSum<<<ilp4_blocks, THREADS>>>(d_a, d_b, d_c, n);
    }, n, NUM_OF_TRIALS);

    // Print results
    printf("\n==================== RESULTS (%s) ====================\n", sizeLabel);
    printf("%-22s %-20s %-22s %-18s\n", "Kernel", "Time (ms)", "Throughput (GB/s)", "Efficiency");
    printf("------------------------------------------------------------------------------------\n");
    for (int i = 0; i < 6; i++) {
        printf("%-22s %7.4f +/- %-7.4f %7.2f +/- %-7.2f %6.2f +/- %-5.2f%%\n", 
               results[i].name, 
               results[i].time_mean, 
               results[i].time_stddev,
               results[i].throughput_mean,
               results[i].throughput_stddev,
               results[i].efficiency,
               results[i].efficiency_stddev);
        
        // Add to global results for CSV
        allResults.push_back(results[i]);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void writeCSV(const std::string& filename) {
    std::ofstream csv(filename);
    csv << "Kernel,N,Time_ms,Time_stddev,Throughput_GBs,Throughput_stddev,Efficiency,Efficiency_stddev\n";
    
    for (const auto& r : allResults) {
        csv << r.name << ","
            << r.n << ","
            << r.time_mean << "," << r.time_stddev << ","
            << r.throughput_mean << "," << r.throughput_stddev << ","
            << r.efficiency << "," << r.efficiency_stddev << "\n";
    }
    csv.close();
    printf("\nResults written to %s\n", filename.c_str());
}


TEST(Small_N_Test, BenchmarkAllKernels_SMALL) {
    runAllBenchmarks(10000000, "10M elements");
}

TEST(SmallMedium_N_Test, BenchmarkAllKernels_SmallMedium) {
    runAllBenchmarks(50000000, "50M elements");
}

TEST(Medium_N_Test, BenchmarkAllKernels_Medium) {
    runAllBenchmarks(100000000, "100M elements");
}

TEST(MediumLarge, BenchmarkAllKernels_MediumLarge) {
    runAllBenchmarks(150000000, "150M elements");
}

TEST(Large_N_Test, BenchmarkAllKernels_Large) {
    runAllBenchmarks(200000000, "200M elements");
}

TEST(Extreme_N_Test, BenchmarkAllKernels_Extreme) {
    runAllBenchmarks(300000000, "300M elements");
}


TEST(Correctness, AllKernels) {
    std::vector<int> test_sizes = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 255, 256, 257, 1024, 10000};
    const int THREADS = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for (int n : test_sizes) {
        size_t size = n * sizeof(float);

        float* h_a = (float*)malloc(size);
        float* h_b = (float*)malloc(size);
        float* h_c = (float*)malloc(size);

        for (int i = 0; i < n; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
        }

        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        auto testKernel = [&](const char* name, auto launchKernel) {
            cudaMemset(d_c, 0, size);
            launchKernel();
            cudaDeviceSynchronize();
            
            cudaError_t err = cudaGetLastError();
            ASSERT_EQ(err, cudaSuccess) << name << " failed at n=" << n << ": " << cudaGetErrorString(err);

            cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++) {
                EXPECT_FLOAT_EQ(h_c[i], 3.0f) << name << " failed at n=" << n << " idx=" << i;
            }
        };

        int blocks_per_sm;

        // 1. Naive
        int naive_blocks = (n + THREADS - 1) / THREADS;
        testKernel("Naive", [&]() {
            vectorSum<<<naive_blocks, THREADS>>>(d_a, d_b, d_c, n);
        });

        // 2. Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
        testKernel("GridStride", [&]() {
            gridStrideVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

        // 3. Vectorized
        int vec_blocks = ((n + 3) / 4 + THREADS - 1) / THREADS;
        testKernel("Vectorized", [&]() {
            vectorizedVectorSum<<<vec_blocks, THREADS>>>(d_a, d_b, d_c, n);
        });

        // 4. Vectorized + Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
        testKernel("VecGridStride", [&]() {
            gridVectorizedVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

        // 5. ILP=2 + Vectorized + Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILP2VectorizedGridVectorSum, THREADS, 0);
        testKernel("VecGridILP2", [&]() {
            ILP2VectorizedGridVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

        // 6. ILP=4 + Vectorized + Grid Stride
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILP4VectorizedGridVectorSum, THREADS, 0);
        testKernel("VecGridILP4", [&]() {
            ILP4VectorizedGridVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
        });

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
    
    // 1. Naive
    int naive_blocks = (n + THREADS - 1) / THREADS;
    vectorSum<<<naive_blocks, THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // 2. Grid Stride
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridStrideVectorSum, THREADS, 0);
    gridStrideVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // 3. Vectorized
    int vec_blocks = ((n + 3) / 4 + THREADS - 1) / THREADS;
    vectorizedVectorSum<<<vec_blocks, THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // 4. Vectorized + Grid Stride
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, gridVectorizedVectorSum, THREADS, 0);
    gridVectorizedVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // 5. ILP=2 + Vectorized + Grid Stride
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILP2VectorizedGridVectorSum, THREADS, 0);
    ILP2VectorizedGridVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // 6. ILP=4 + Vectorized + Grid Stride
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ILP4VectorizedGridVectorSum, THREADS, 0);
    ILP4VectorizedGridVectorSum<<<prop.multiProcessorCount * blocks_per_sm, THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}



TEST(ExportCSV, WriteResults) {
    writeCSV("benchmark_results.csv");
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}