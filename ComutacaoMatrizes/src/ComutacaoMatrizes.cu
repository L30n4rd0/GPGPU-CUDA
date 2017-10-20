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

__global__ void commutative (int **matrixA, int **matrixB, int *result) {

	int ab[N][N], ba[N][N];

	int tid = blockIdx.x;
	int tid2 = blockIdx.y;

	if (tid < N) {
		if (tid2) {

			ab[tid][tid2] =

		}

	}

	result[0] = matrixA[0][0] + matrixB[0][0];

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

	commutative<<<(N * N), 1>>>(dev_matrixA, dev_matrixB, dev_result);

	cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Result: %d", result);

	return 0;
}
