#include <iostream>
#include <numeric>
#include <stdlib.h>

#define N 2
#define BLOCK_DIM 32


__device__ int internalProduct(int *vectorA, int *vectorB, int *vectorResult);
__global__ void addVector(int *vectorA, int *vectorB, int *vectorResult, int vectorSize);
__global__ void productMatrix(int *matrixA, int *matrixB, int *productAB, int rows, int columns);

int main() {
	int matrixA[N][N] = {
			{1, 2},
			{3, 4}
	},
	matrixB[N][1] = {
			{2},
			{2}
	},
	productAB[N][N];
	int *dev_a, *dev_b, *dev_productAB, *dev_test;

	printf("sizeof: %d", sizeof(productAB));

	int size = N * N * sizeof(int), test;

	// initialize a and b with real values (NOT SHOWN)
	cudaMalloc((void**) &dev_a, size);
	cudaMalloc((void**) &dev_b, size);
	cudaMalloc((void**) &dev_productAB, size);
	cudaMalloc((void**) &dev_test, sizeof(int));

	cudaMemcpy(dev_a, matrixA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, matrixB, size, cudaMemcpyHostToDevice);

//	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
//	dim3 dimGrid((int) ceil(N / dimBlock.x), (int) ceil(N / dimBlock.y));

	dim3 grid(65536, 1);
	dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

	productMatrix<<<grid, block>>>(dev_a, dev_b, dev_productAB, N, N);

	cudaMemcpy(productAB, dev_productAB, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_test, sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nMatrix productAB:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", productAB[i][j]);
		}
		printf("\n");
	}

	printf("Teste: %d", test);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_productAB);
}
__global__ void productMatrix(int *matrixA, int *matrixB, int *productAB, int rows, int columns) {
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
//	unsigned int index = (y * gridDim.x * blockDim.x) + x;
	int k, sum = 0;

//	if (index < N) {
//		productAB[index] = matrixA[index] * matrixB[index];
////		test[0] = index;
//	}
//
//	if (index >= N && index < N * N) {
//		productAB[index] = matrixA[index] + matrixB[index];
////		test[0] = index;
//	}
//
//	if (index == 67108864 - 1) {
//		*width = index;
//
//	}

	if (x < columns) {
		if (y < rows) {

			for (k = 0; k < columns; k++) {
				sum += matrixA[y * columns + k] * matrixB[k * rows + x];
			}

			productAB[y * columns + x] = sum;
		}
	}

}

__device__ int internalProduct(int *vectorA, int *vectorB, int *vectorResult) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	int result, index;
	index = row + (col * N);

	memset(vectorResult,0 ,sizeof(int) * N);

	if (row < N) {
		if (col < N) {
			result = index;
		}
	}

//	for (int i = 0; i < N; ++i) {
//		result += vectorResult[i];
//
//	}

	memset(vectorResult,0 ,sizeof(int) * N * N);

	return result;
}

__global__ void addVector(int *vectorA, int *vectorB, int *vectorResult, int vectorSize) {
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index = (y * gridDim.x * blockDim.x) + x;

	if (index < vectorSize) {
		vectorResult[index] = vectorA[index] + vectorB[index];
	}

}
