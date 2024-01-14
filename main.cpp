#include "framework.h"

#include "cuda.cuh"

void callKernels(int blocks_per_grid, int max_threads, unsigned int *cudaImage, int pixelsCount,
    int width, int height)
{
    doThingsKernel << <blocks_per_grid, max_threads >> > (pixelsCount, cudaImage, width, height);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

int main()
{
    Application app;
    app.run();
    return 0;
}