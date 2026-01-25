#include "vectorSumHeader.cuh"
#include <iostream>

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    //Allocate memory on CPU
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    //Populate with data
    for(int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    //Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //Move data from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //Execute kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 144; // RTX 4060 has 24 SM's, while 128 blocks optimial as it is divisible by 32 threads (warp), 144 allows each SM to have 6 blocks each resulting in all SMs finishing at approximately the same time.
                             //Nsight confirms 144 block
    gridVectorizedVectorSum<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    //Move result from GPU to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << "Result of first element: " << h_C[0] << " (Expected: 3.0)" << std::endl;

    //Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}