/*
 ============================================================================
 Name        : TestC.c
 Author      : Leonardo
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#define N 2

int* matrix() {

	int matrixA[N][N] = {{1, -1}, {0, 2}};

	return matrixA;

}

int main(void) {
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	int test[N][N] = matrix();

	printf("Valor: %d", test[0][0]);


	return EXIT_SUCCESS;
}
