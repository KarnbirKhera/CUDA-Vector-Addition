#pragma once

__global__ void vectorSum(float* A, float* B, float* C, int n);
__global__ void gridStrideVectorSum(float* A, float* B, float* C, int n);
__global__ void vectorizedVectorSum(float* A, float* B, float* C, int n);
__global__ void gridVectorizedVectorSum(float* A, float* B, float* C, int n);
__global__ void ILP2VectorizedGridVectorSum(float* A, float* B, float* C, int n);
__global__ void ILP4VectorizedGridVectorSum(float* A, float* B, float* C, int n);
