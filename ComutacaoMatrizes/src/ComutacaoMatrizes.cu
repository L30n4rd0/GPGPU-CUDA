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

__global__ void commutative (int **matrixA, int **matrixB, int **productAB, int **productBA, int *result);
__device__ int **allocateDeviceMatrix(int rows, int cols);
__device__ void freeDeviceMatrix(int **dev_matrix, int rows);
__device__ int **product(int **matrixA, int **matrixB);
__device__ int compare(int **matrixA, int **matrixB);
void allocateMatrixDev(int **matrix, int rows, int cols);
void freeMatrixDev(int **matrix, int rows);
void allocateMatrix(int **matrix, int rows, int cols);
void freeMatrix(int **matrix, int rows);
int **getTransposedMatrix(int **matrix);
int *getSerializedMatrix(int matrix[N][N]);

int main(void)
{
	int matrixA[N][N] = {
			{1, -1},
			{0, 2}
	};
	int matrixB[N][N] = {
			{1, -2},
			{0, 3}
	};

	int matrixTemp[N][N];
//	allocateMatrix(matrixTemp, N, N);
	printf("\npassou 2");

//	int *serealizedMatrixA = getSerializedMatrix(matrixA);
//	int *serealizedMatrixB = getSerializedMatrix(matrixB);
//
//	int row;
//	for (row = 0; row < N * N; row++) {
//		printf("%d", serealizedMatrixA[row]);
//	}

	int result = 0;
	int **dev_matrixA, **dev_matrixB, **dev_ab, **dev_ba, *dev_result;

//	allocateMatrixDev(dev_matrixA, N, N);
//	allocateMatrixDev(dev_matrixB, N, N);
//	allocateMatrixDev(dev_ab, N, N);
//	allocateMatrixDev(dev_ba, N, N);

	cudaMalloc((void**) &dev_result, sizeof(int));
	cudaMalloc((void**) &dev_matrixA, N * N * sizeof(int));
	cudaMalloc((void**) &dev_matrixB, N * N * sizeof(int));
	cudaMalloc((void**) &dev_ab, N * N * sizeof(int));
	cudaMalloc((void**) &dev_ba, N * N * sizeof(int));


	// for each row allocate cols ints
//	int row;
//	for (row = 0; row < N; row++) {
//		cudaMalloc((void**) dev_matrix[row], N * sizeof(int));
//	}
//
////
	cudaMemcpy(dev_matrixA, matrixA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrixB, matrixB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	commutative<<<N, N>>>(dev_matrixA, dev_matrixB, dev_ab, dev_ba, dev_result);

	cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixTemp, dev_ab, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nMatrix temp:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", matrixTemp[i][j]);
		}
		printf("\n");
	}

	printf("\nResult: %d", result);

//	freeMatrix(matrixTemp, N);
//	freeMatrixDev(dev_matrixA, N);
//	freeMatrixDev(dev_matrixB, N);
//	freeMatrixDev(dev_ab, N);
//	freeMatrixDev(dev_ba, N);
//	cudaFree(dev_result);

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

__global__ void commutative (int **matrixA, int **matrixB, int **productAB, int **productBA, int *result) {

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	if(blockId < N) {
		if (threadId < N) {
			productAB[blockId][threadId] = matrixA[blockId][threadId] * matrixB[blockId][threadId];
		}
	}

//	int **ab, **ba;

//	ab = product(matrixA, matrixB);
//
//	ba = product(matrixB, matrixA);

//	result[0] = compare(ab, ba);

//	result[0] = 11;
	result[0] = matrixA[0][0];

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

//    int row;

//    // first free each row
//    for (row = 0; row < rows; row++) {
//    	cudaFree(dev_matrix[row]);
//    }
//
//    // Eventually free the memory of the pointers to the rows
//    cudaFree(dev_matrix);
}

int *getSerializedMatrix(int matrix[N][N]) {

	int *matrixResult = (int *) malloc(N * N * sizeof(int *));
	int cont = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrixResult[cont] = matrix[i][j];
			// printf("Matrix-%d: %d\n", cont, matrixResult[cont]);
			cont++;
		}
	}

	return matrixResult;

}

int **getTransposedMatrix(int **matrix) {



}

void allocateMatrix(int **matrix, int rows, int cols) {

    // allocate rows, each row is a pointer to int
    matrix = (int **) malloc(rows * sizeof(int *));

    // for each row allocate cols ints
    int row;
    for (row = 0; row < rows; row++) {
        matrix[row] = (int *) malloc(cols * sizeof(int));
    }

    matrix[0][0] = 34;

    printf("\nValor34: %d", matrix[0][0]);

}

void allocateMatrixDev(int **matrix, int rows, int cols) {

	// allocate rows, each row is a pointer to int
	cudaMalloc((void**) &matrix, rows * sizeof(int *));

	// for each row allocate cols ints
	int row;
	for (row = 0; row < rows; row++) {
		printf("dasdasda");
		cudaMalloc((void**) &matrix[row], cols * sizeof(int));
	}

}

void freeMatrixDev(int **matrix, int rows) {
	// you must supply the number of rows

    int row;

    // first free each row
    for (row = 0; row < rows; row++) {
    	cudaFree(matrix[row]);
    }

    // Eventually free the memory of the pointers to the rows
    cudaFree(matrix);
}

void freeMatrix(int **matrix, int rows) {
	// you must supply the number of rows

    int row;

    // first free each row
    for (row = 0; row < rows; row++) {
         free(matrix[row]);
    }

    // Eventually free the memory of the pointers to the rows
    free(matrix);
 }
