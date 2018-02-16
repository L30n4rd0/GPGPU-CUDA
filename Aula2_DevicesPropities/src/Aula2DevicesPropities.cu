/*
 ============================================================================
 Name        : GetPropitiesDevices.cu
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
#include <cuda.h>

int main(void) {
	cudaDeviceProp prop;

	int count;

	cudaGetDeviceCount(&count);

	printf("Quantidade de GPU: %d", count);

	cudaGetDeviceProperties(&prop, 0);

//	for (int i = 0; i < count; ++i) {
//		cudaGetDeviceProperties(&prop, i);
//
//	}

	printf("\nnome: %s", prop.name);
	printf("\ntotalGlobalMem: %zu", prop.totalGlobalMem);
	printf("\nsharedMemPerBlock: %zu", prop.sharedMemPerBlock);
	printf("\nsharedMemPerMultiprocessor: %zu", prop.sharedMemPerMultiprocessor);
	printf("\nconcurrentKernels: %d", prop.concurrentKernels);
	printf("\nmultiProcessorCount: %d", prop.multiProcessorCount);
	printf("\nregsPerBlock: %d", prop.regsPerBlock);
	printf("\nwarpSize: %d", prop.warpSize);
	printf("\nmaxThreadsPerBlock: %d", prop.maxThreadsPerBlock);
	printf("\nmaxThreadsDim: %d", prop.maxThreadsDim);
	printf("\nmaxGridSize: %d", prop.maxGridSize);
	printf("\nmajor: %d", prop.major);
	printf("\nminor: %d", prop.minor);

	return 0;
}
