/*
 ============================================================================
 Name        : Comutacao.cu
 Author      : Luciano
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 2

__global__ void product(int *A, int *B, int *C){

	int xid = threadIdx.x;
	int yid = threadIdx.y;

	memset(C ,0 ,sizeof(int) * N * N);

	if(xid < N && yid < N){

		for (int k = 0; k < N; k++) {

			C[xid * N + yid] += A[xid * N + k] * B[k * N + yid];

		}

	}

}

__global__ void compare(int *A, int *B, int *result){

	*result = 1;

	int xid = threadIdx.x;
	int yid = threadIdx.y;

	if(xid < N && yid < N){

		if( A[xid * N + yid] != B[xid * N + yid] ){

			*result = 0;

		}

	}


}

int main(void)
{

	int A[N][N], B[N][N], C[N][N], D[N][N];
	int *dev_a, *dev_b, *dev_c, *dev_d, *result;
	int commute;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; ++j) {
			A[i][j] = i + j;
			B[i][j] = i * j;
		}
	}

	cudaMalloc((void**)&dev_a, N * N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * N * sizeof(int));
	cudaMalloc((void**)&dev_d, N * N * sizeof(int));
	cudaMalloc((void**)&result, sizeof(int));

	cudaMemcpy(dev_a, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(N,N);

	product<<<1,threadsPerBlock>>>(dev_a, dev_b, dev_c);
	product<<<1,threadsPerBlock>>>(dev_b, dev_a, dev_d);
	compare<<<1 , threadsPerBlock >>>(dev_c ,dev_d ,result );

	cudaMemcpy(C, dev_c, N * N * (sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(D, dev_d, N * N * (sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(&commute, result, (sizeof(int)), cudaMemcpyDeviceToHost);

	if (commute == 1) {
		printf("Sim");
	} else {
		printf("Não");
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
