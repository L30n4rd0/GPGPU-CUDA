/*
 ============================================================================
 Name        : ComutacaoMatrizes.cu
 Author      : Leonardo
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>

#define N 2

__device__ int **product(int **matrixA, int **matrixB){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int matrixResult[N][N];

//	memset(C ,0 ,sizeof(int) * N * N);

	if(blockId < N && threadId < N){

		for (int k = 0; k < N; k++) {

//			C[xid * N + yid] += A[xid * N + k] * B[k * N + yid];

		}

	}

	return matrixResult;

}

__device__ int compare(int **matrixA, int **matrixB){

	int result = 1;

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	if(blockId < N && threadId < N){

		if( matrixA[blockId][threadId] != matrixB[blockId][threadId] ) {
			result = 0;

		}

	}

	return result;

}

__global__ void commutative (int **matrixA, int **matrixB, int *result) {

	int ab[N][N], ba[N][N];

	ab = product(matrixA, matrixB);

	ba = product(matrixB, matrixA);

	result[0] = compare(ab, ba);

}

int main(void)
{
	int matrixA[N][N] = {{1, -1}, {0, 2}};
	int matrixB[N][N] = {{1, -2}, {0, 3}};
	int result = 0;
	int **dev_matrixA, **dev_matrixB, *dev_result;

	cudaMalloc((void**) &dev_matrixA, N * N * sizeof(int));
	cudaMalloc((void**) &dev_matrixB, N * N * sizeof(int));
	cudaMalloc((void**) &dev_result, sizeof(int));


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", matrixA[i][j]);
		}
		printf("\n");
	}

	cudaMemcpy(dev_matrixA, matrixA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrixB, matrixB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	commutative<<<N, N>>>(dev_matrixA, dev_matrixB, dev_result);

	cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Result: %d", result);

	return 0;
}
