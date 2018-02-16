#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define bloco 16
#define nelem 4096

float ran1(long *idum);
__global__ void prodVecCuda(float * a, float * b, float * c);

int main(void) {
	int i, j;
	unsigned int nThreads, tamanho_bytes;
	long int semente;
	float *a, *b, *c;
	float *ga, *gb, *gc;
	float time;
	/* estrutura CUDA que permite armazenar tempo */
	cudaEvent_t start, stop;
	/* aloca espaco na RAM */
	a = (float *) malloc(nelem * nelem * sizeof(float));
	b = (float *) malloc(nelem * nelem * sizeof(float));
	c = (float *) malloc(nelem * nelem * sizeof(float));
	semente = -3890103L;
	/* Gera numeros aleatorios  e preenche as matrizes*/
	for (i = 0; i < nelem; i++) {
		for (j = 0; j < nelem; j++) {
			a[i * nelem + j] = ran1(&semente);
		}
	}
	for (i = 0; i < nelem; i++) {
		for (j = 0; j < nelem; j++) {
			b[i * nelem + j] = ran1(&semente);
		}
	}
	for (i = 0; i < nelem; i++) {
		for (j = 0; j < nelem; j++) {
			c[i * nelem + j] = 0.0;
		}
	}
	/* define numero de threads por bloco */
	dim3 nThreadsPorBloco(bloco, bloco);
	/* define numero de blocos */
	i = nelem / bloco;
	dim3 nBlocos(i, i);
	nThreads = nelem * nelem;
	/* alocando memoria global da GPU */
	tamanho_bytes = nThreads * sizeof(float);
	cudaMalloc((void **) &ga, tamanho_bytes);
	cudaMalloc((void **) &gb, tamanho_bytes);
	cudaMalloc((void **) &gc, tamanho_bytes);
	/* Copia dados da RAM para a memoria global da GPU */
	cudaMemcpy(ga, a, tamanho_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gb, b, tamanho_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gc, c, tamanho_bytes, cudaMemcpyHostToDevice);
	/* Inicia o cronometro e registra o tempo */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	prodVecCuda<<<nBlocos, nThreadsPorBloco>>>(ga, gb, gc);
	/* Para o cronometro e registra o tempo */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	/* Copia o resultado da memoria global da GPU ah RAM */
	cudaMemcpy(c, gc, tamanho_bytes, cudaMemcpyDeviceToHost);
	/* Imprime um dado do meio da matriz resultado */
	i = nelem / 2;
	j = nelem / 2;
	printf("%04.2f\n", c[i * nelem + j]);
	/* Imprime o tempo de execucao */
	time = time / 1000.0;
	printf("O tempo de execucao foi: %f segundos\n", time);
	free(a);
	free(b);
	free(c);
	cudaFree(ga);
	cudaFree(gb);
	cudaFree(gc);
	return 0;
}
/*   Kernel CUDA */
__global__ void prodVecCuda(float * a, float * b, float * c) {
	__shared__ float as[bloco][bloco];
	__shared__ float bs[bloco][bloco];
	int bx, by, tx, ty, clin, ccol, k, m;
	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	clin = by * bloco + ty;
	ccol = bx * bloco + tx;
	float soma = 0;
	for (m = 0; m < nelem / bloco; ++m) {
		as[ty][tx] = a[clin * nelem + (m * bloco + tx)];
		bs[ty][tx] = b[(m * bloco + ty) * nelem + ccol];
		__syncthreads();
		for (k = 0; k < bloco; ++k) {
			soma += as[ty][k] * bs[k][tx];
			__syncthreads();
		}
	}
	c[clin * nelem + ccol] = soma;
}
/* Numeros aleatorios gerados na CPU */
/* O SDK da Nvidia prove um gerador */

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
float ran1(long *idum) {
	int j;
	long k;
	static long iy = 0;
	static long iv[NTAB];
	float temp;
	if (*idum <= 0 || !iy) {
		if (-(*idum) < 1)
			*idum = 1;
		else
			*idum = -(*idum);
		for (j = NTAB + 7; j >= 0; j--) {
			k = (*idum) / IQ;
			*idum = IA * (*idum - k * IQ) - IR * k;
			if (*idum < 0)
				*idum += IM;
			if (j < NTAB)
				iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0)
		*idum += IM;
	j = iy / NDIV;
	iy = iv[j];
	iv[j] = *idum;
	if ((temp = AM * iy) > RNMX)
		return RNMX;
	else
		return temp;
}

#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
