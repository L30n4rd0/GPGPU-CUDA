/*
 ============================================================================
 Name        : ExemploSlide3.cu
 Author      : Leonardo
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

#define N 10

__global__ void add(int *a, int *b, int *c);

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

// allocate the memory on the GPU
	cudaMalloc((void**) &dev_a, sizeof(a));
	cudaMalloc((void**) &dev_b, sizeof(b));
	cudaMalloc((void**) &dev_c, sizeof(c));

// fill the arrays 'a' and 'b' on the CPU
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N, 1>>> (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;

}

__global__ void add(int *a, int *b, int *c) {
	int tid = blockIdx.x;
// handle the data at this index
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}