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

int **allocateMatrix(int rows, int cols);
void freeMatrix(int **matrix, int rows);
int **getFilledMatrix(int rows, int cols);

int main(void) {

	int a = 2, *p = &a;
	int **test = getFilledMatrix(N, N);

	printf("\na: %d", a);
	printf("\n&a: %d", &a);
	printf("\np: %d", p);
	printf("\n&p: %d", &p);
	printf("\n*p: %d", *p);

	printf("\n## MATRIZ ##\n");
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			printf("%d ", test[row][col]);
		}
		printf("\n");
	}

	freeMatrix(test, N);

	return EXIT_SUCCESS;
}

int **getFilledMatrix(int rows, int cols) {

	int **matrix = allocateMatrix(rows, cols);
	
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			matrix[row][col] = col + 1;
		}
		
	}

	return matrix;

}

int **allocateMatrix(int rows, int cols) {

    // allocate rows, each row is a pointer to int
    int **matrix = (int **)malloc(rows * sizeof(int *));

    // for each row allocate cols ints
    int row;
    for (row = 0; row < rows; row++) {
        matrix[row] = (int *)malloc(cols * sizeof(int));
    }

    matrix[0][0] = 34;

    printf("\nValor34: %d", matrix[0][0]);

    return matrix;
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
