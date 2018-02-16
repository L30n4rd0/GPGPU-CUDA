/*
 ============================================================================
 Name        : TesteC.c
 Author      : Leonardo
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void kernel (void) {

}

int main(void) {

	kernel <<<1, 1>>>();

	printf("!!!Hello World!!!"); /* prints !!!Hello World!!! */
	return EXIT_SUCCESS;
}
