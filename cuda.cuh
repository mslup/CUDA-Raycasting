#include "framework.h"

#include <stdio.h>

__global__ void doThingsKernel(int pixelsCount, unsigned int* image, int width, int height)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k >= pixelsCount)
		return;

	int y = k / width;
	int x = k % width;

	GLuint res = 0x000000ff | ((int)(y * 255.0 / height) << 8);
	image[k] = res;
}