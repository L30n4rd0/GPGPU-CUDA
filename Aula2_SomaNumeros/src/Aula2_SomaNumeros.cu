/*
 ============================================================================
 Name        : ExemploSlide2.cu
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

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void add(int a, int b, int *c) {
	c[0] = a + b;
}

int main(void) {
	int c;
	int *dev_c;

	cudaMalloc((void**) &dev_c, sizeof(int));

	add<<<1, 1>>>(2, 7, dev_c);

	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("2 + 7 = %d\n", c);

	cudaFree(dev_c);

	return 0;
}
