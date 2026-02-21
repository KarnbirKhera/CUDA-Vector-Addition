#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

__global__ void vectorSum(float* A, float* B, float* C, int n);
__global__ void gridStrideVectorSum(float* A, float* B, float* C, int n);
__global__ void vectorizedVectorSum(float* A, float* B, float* C, int n);
__global__ void gridVectorizedVectorSum(float* A, float* B, float* C, int n);
__global__ void ILP2VectorizedGridVectorSum(float* A, float* B, float* C, int n);
__global__ void ILP4VectorizedGridVectorSum(float* A, float* B, float* C, int n);


__global__ void vectorSumWriteOnly(float* C, int n);
__global__ void vectorSumReadOnly(float* A, float* B, float* C, int n);

__global__ void vectorAddPTX_CacheStreamed(float* A, float* B, float* C, int n);
__global__ void vectorAddPTX_LastUse(float* A, float* B, float* C, int n);

__global__ void vectorSumILP(float* A, float* B, float* C, int n);
__global__ void vectorSumGridILP2(float* A, float* B, float* C, int n);
__global__ void vectorSumGridILP4(float* A, float* B, float* C, int n);


__global__ void vectorSumVectorizedNoPadding(float* A, float* B, float* C, int n);

__global__ void writeOnlyUncoalesced(float* C, int n);
__global__ void writeOnlyCoalesced(float* C, int n);

__global__ void vectorAddFP8(const float* A, const float* B, __nv_fp8_e4m3* C, int n);