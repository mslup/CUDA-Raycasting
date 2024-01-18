#pragma once

#include "framework.hpp"
#include "scene.hpp"
#include <stdio.h>

struct cudaArguments {
    unsigned int* cudaImage;
    const int width;
    const int height;
    const class Scene scene;
    const glm::vec3 rayOrigin;
    const glm::vec3* rayDirections;
    const glm::vec3 cameraPos;
    const glm::mat4 inverseProjMatrix;
    const glm::mat4 inverseViewMatrix;
    bool shadows;
};

__host__ void callKernels(dim3 blocks_per_grid, dim3 max_threads,
    cudaArguments args);

__global__ void rayTrace(cudaArguments args);

