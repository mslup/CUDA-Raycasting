#ifndef _CUDA_CUH
#define _CUDA_CUH

#include "framework.hpp"
#include "scene.hpp"
#include <stdio.h>

//struct Scene;

struct cudaArguments {
    unsigned int* cudaImage;
    const int width;
    const int height;
    const struct Scene scene;
    const glm::vec3 rayOrigin;
    const glm::vec3* rayDirections;
    const glm::vec3 cameraPos;
    const glm::mat4 inverseProjMatrix;
    const glm::mat4 inverseViewMatrix;
};

__host__ void callKernels(dim3 blocks_per_grid, dim3 max_threads,
    cudaArguments args);

__global__ void rayTrace(cudaArguments args);

#endif