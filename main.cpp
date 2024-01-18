#include "framework.hpp"
#include "application.hpp"
#include "cuda.cuh"

__host__ void callKernels(dim3 blocks_per_grid, dim3 max_threads, cudaArguments args)
{
    rayTrace << <blocks_per_grid, max_threads >> > (args);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

int main()
{
    Application app;
    app.run();
    return 0;
}