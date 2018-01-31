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

__global__ void commutative (int **matrixA, int **matrixB, int *result);
__device__ int **allocateDeviceMatrix(int rows, int cols);
__device__ void freeDeviceMatrix(int **dev_matrix, int rows);
__device__ int **product(int **matrixA, int **matrixB);
__device__ int compare(int **matrixA, int **matrixB);
int **allocateMatrix(int rows, int cols);
void freeMatrix(int **matrix, int rows);

int main(void)
{
	int matrixA[N][N] = {{1, -1}, {0, 2}};
	int matrixB[N][N] = {{1, -2}, {0, 3}};
	int result = 0;
	int **dev_matrixA, **dev_matrixB, *dev_result;

	dev_matrixA = allocateMatrix(N, N);
	dev_matrixB = allocateMatrix(N, N);

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

	freeMatrix(dev_matrixA, N);
	freeMatrix(dev_matrixB, N);
	cudaFree(dev_result);

	return 0;
}

__device__ int **product(int **matrixA, int **matrixB){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int **matrixResult = allocateDeviceMatrix(N, N);

//	memset(C ,0 ,sizeof(int) * N * N);

	if(blockId < N && threadId < N){

		matrixResult[blockId][threadId] = matrixA[blockId][threadId] + matrixB[blockId][threadId];

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

//	int **ab, **ba;

//	ab = product(matrixA, matrixB);
//
//	ba = product(matrixB, matrixA);

//	result[0] = compare(ab, ba);

	result[0] = 11;

}

__device__ int **allocateDeviceMatrix(int rows, int cols) {

    // allocate rows, each row is a pointer to int
    int **dev_matrix;
    cudaMalloc((void**) &dev_matrix, rows * sizeof(int *));

    // for each row allocate cols ints
    int row;
    for (row = 0; row < rows; row++) {
    	cudaMalloc((void**) &dev_matrix[row], cols * sizeof(int));
    }

    return dev_matrix;
}

__device__ void freeDeviceMatrix(int **dev_matrix, int rows) {
	// you must supply the number of rows

    int row;

//    // first free each row
//    for (row = 0; row < rows; row++) {
//    	cudaFree(dev_matrix[row]);
//    }
//
//    // Eventually free the memory of the pointers to the rows
//    cudaFree(dev_matrix);
}

int **allocateMatrix(int rows, int cols) {

	// allocate rows, each row is a pointer to int
	int **dev_matrix;
	cudaMalloc((void**) &dev_matrix, rows * sizeof(int *));

	// for each row allocate cols ints
	int row;
	for (row = 0; row < rows; row++) {
		cudaMalloc((void**) &dev_matrix[row], cols * sizeof(int));
	}

	return dev_matrix;
}

void freeMatrix(int **matrix, int rows) {
	// you must supply the number of rows

    int row;

    // first free each row
    for (row = 0; row < rows; row++) {
    	cudaFree(matrix[row]);
    }

    // Eventually free the memory of the pointers to the rows
    cudaFree(matrix);
}
