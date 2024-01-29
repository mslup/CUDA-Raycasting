# GPU-Accelerated Ray Tracing Simulation

## Overview

This C++ project, developed for the course on Graphic Processors in Computational Applications at the Faculty of Mathematics and Information Science at WUT, implements a ray tracing simulation using CUDA Toolkit 12.2. The project is compiled in Visual Studio 2022 and optimized for Release mode. Heavily inspired by TheCherno's [series on ray tracing](https://www.youtube.com/watch?v=gfW1Fhd9u9Q&list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl&ab_channel=TheCherno) in C++.

By default, the simulation runs on GPU, offering an option in the Menu to switch to CPU. The scene includes 1000 spheres and 10 light sources.

## Ray Tracing Mechanism

For each pixel, a ray is cast from the camera, and each ray corresponds to a separate thread. The simulation checks for sphere intersections, applying the [Phong lighting model](https://en.wikipedia.org/wiki/Phong_reflection_model) to calculate pixel colors.

In the shadow-enabled version (selectable in the menu), rays are reflected to all light sources. New rays are created, checking for additional sphere intersections on the path from the light source to the reflection point.

## Camera Controls

Camera controls are outlined in the Menu under the Camera Controls section. The project incorporates projection and view matrices along with texture resizing for dynamic window size changes and automatic resolution adjustments

## Adjustable Parameters

### Environment
- **Sky color:** Color of a pixel where the ray doesn't hit any sphere.
- **Ambient light color:** Color of ambient light.
- **Sphere material:** Parameters for the Phong lighting model (diffuse, specular, shininess, ambient).

### Light Sources
- **Attenuation:** Light strength, determining how light intensity changes with increasing distance from the object.
- **Light positions**
- **Light colors**
- **Disable lights**

<!--
## How to Run

1. Ensure you have Visual Studio 2022 installed.
2. Install CUDA Toolkit 12.2.
3. Clone the repository.
4. Open the project in Visual Studio.
5. Select the Release configuration.
6. Build and run the project.
-->

## Demonstration
![Zrzut ekranu 2024-01-17 223353](https://github.com/mslup/CUDA-Raycasting/assets/132074948/35ed03d9-d91a-4663-b1d6-f08011ed93c4)
![Zrzut ekranu 2024-01-17 224856](https://github.com/mslup/CUDA-Raycasting/assets/132074948/a65f3c16-9ca5-4277-924b-6703cb4557ad)
![Zrzut ekranu 2024-01-17 184450](https://github.com/mslup/CUDA-Raycasting/assets/132074948/9fb3cbce-bd8a-47f8-84cc-220369b1f3c1)



## Todo
- write from CUDA to texture buffer directly
- further optimalization (kd-tree...)
- physically based rendering
