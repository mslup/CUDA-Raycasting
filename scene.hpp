#pragma once

#include "framework.hpp"

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/constants.hpp>

#ifndef GPU_ERRCHK
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif

class Scene
{
public:
	glm::vec3* spherePositions;
	glm::vec3* sphereAlbedos;
	float* sphereRadii;

	glm::vec3* lightPositions;
	glm::vec3* lightColors;
	const float lightRadius = 0.2f;
	bool* lightBools;

	// todo: idk, dry, maybe do sth like SceneGPU and SceneCPU?
	glm::vec3* cudaSpherePositions;
	glm::vec3* cudaSphereAlbedos;
	float* cudaSphereRadii;

	glm::vec3* cudaLightPositions;
	glm::vec3* cudaLightColors;
	bool* cudaLightBools;

	const int sphereCount = 1000;
	const int lightCount = 10;

	glm::vec3 ambientColor{
		1.0f, 1.0f, 1.0f
	};
	glm::vec3 skyColor{
		0.25f, 0.25f, 0.25f };

	float kDiffuse = 0.9f;
	float kSpecular = 0.4f;
	float kAmbient = 0.2f;
	float kShininess = 40;

	float linearAtt = 0.0f;
	float quadraticAtt = 0.032f;

	const float worldBorder = 30;

	void create()
	{
		static std::vector<glm::vec3> colorPalette;
		static std::vector<glm::vec3> lightColorPalette;
		{
			//colorPalette.push_back(glm::vec3(0.5, 0.5, 0.5));
			colorPalette.push_back(glm::vec3(155.0f / 255.0f, 34.0f / 255.0f, 38.0f / 255.0f));
			colorPalette.push_back(glm::vec3(0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f));
			colorPalette.push_back(glm::vec3(0.0f / 255.0f, 95.0f / 255.0f, 115.0f / 255.0f));
			colorPalette.push_back(glm::vec3(10.0f / 255.0f, 147.0f / 255.0f, 150.0f / 255.0f));
			colorPalette.push_back(glm::vec3(148.0f / 255.0f, 210.0f / 255.0f, 189.0f / 255.0f));
			colorPalette.push_back(glm::vec3(233.0f / 255.0f, 216.0f / 255.0f, 166.0f / 255.0f));
			colorPalette.push_back(glm::vec3(238.0f / 255.0f, 155.0f / 255.0f, 0.0f / 255.0f));
			colorPalette.push_back(glm::vec3(202.0f / 255.0f, 103.0f / 255.0f, 2.0f / 255.0f));
			colorPalette.push_back(glm::vec3(187.0f / 255.0f, 62.0f / 255.0f, 3.0f / 255.0f));
			colorPalette.push_back(glm::vec3(174.0f / 255.0f, 32.0f / 255.0f, 18.0f / 255.0f));

			//colorPalette.push_back(glm::vec3(128.0f / 255.0f, 93.0f / 255.0f, 147.0f / 255.0f));
			//colorPalette.push_back(glm::vec3(244.0f / 255.0f, 159.0f / 255.0f, 188.0f / 255.0f));
			//colorPalette.push_back(glm::vec3(255.0f / 255.0f, 211.0f / 255.0f, 186.0f / 255.0f));
			//colorPalette.push_back(glm::vec3(158.0f / 255.0f, 189.0f / 255.0f, 110.0f / 255.0f));
			//colorPalette.push_back(glm::vec3(22.0f / 255.0f, 152.0f / 255.0f, 115.0f / 255.0f));

			lightColorPalette.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
			lightColorPalette.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
			lightColorPalette.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
			lightColorPalette.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
			lightColorPalette.push_back(glm::vec3(1.0f, 0.0f, 1.0f));
			lightColorPalette.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
			lightColorPalette.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
			lightColorPalette.push_back(glm::vec3(0.5f, 1.0f, 0.2f));
			lightColorPalette.push_back(glm::vec3(0.0f, 0.5f, 0.1f));
			lightColorPalette.push_back(glm::vec3(0.3f, 0.1f, 0.8f));
		}

		spherePositions = new glm::vec3[sphereCount];
		sphereRadii = new float[sphereCount];
		sphereAlbedos = new glm::vec3[sphereCount];

		lightPositions = new glm::vec3[lightCount];
		lightColors = new glm::vec3[lightCount];
		lightBools = new bool[lightCount];

		srand(time(NULL));
		for (int i = 0; i < sphereCount; i++)
		{
			// generate uniformly distributed points in a ball
			glm::vec3 direction = glm::normalize(glm::gaussRand(glm::vec3(0.0f), glm::vec3(1.0f)));
			float radius = glm::pow(glm::linearRand(0.0f, 1.0f), 1.0f / 3.0f) * worldBorder;
			spherePositions[i] = radius * direction;
			sphereRadii[i] = 0.2f * (i % 5 + 1);
			sphereAlbedos[i] = colorPalette[i % colorPalette.size()];
		}

		for (int i = 0; i < lightCount; i++)
		{
			glm::vec3 direction = glm::normalize(glm::gaussRand(glm::vec3(0.0f), glm::vec3(1.0f)));
			float radius = glm::pow(glm::linearRand(0.0f, 1.0f), 1.0f / 3.0f) * worldBorder;
			lightPositions[i] = radius * direction;
			lightColors[i] = lightColorPalette[i % lightColorPalette.size()];
			lightBools[i] = false;
		}

		size_t sphereVec3ArrSize = sphereCount * sizeof(glm::vec3);
		size_t sphereFloatArrSize = sphereCount * sizeof(float);
		size_t lightVec3ArrSize = lightCount * sizeof(glm::vec3);
		size_t lightBoolArrSize = lightCount * sizeof(bool);

		gpuErrchk(cudaMalloc(&cudaSpherePositions, sphereVec3ArrSize));
		gpuErrchk(cudaMalloc(&cudaSphereAlbedos, sphereVec3ArrSize));
		gpuErrchk(cudaMalloc(&cudaSphereRadii, sphereFloatArrSize));
		gpuErrchk(cudaMalloc(&cudaLightPositions, lightVec3ArrSize));
		gpuErrchk(cudaMalloc(&cudaLightColors, lightVec3ArrSize));
		gpuErrchk(cudaMalloc(&cudaLightBools, lightBoolArrSize));

		gpuErrchk(cudaMemcpy(cudaSpherePositions, spherePositions, sphereVec3ArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaSphereAlbedos, sphereAlbedos, sphereVec3ArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaSphereRadii, sphereRadii, sphereFloatArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaLightPositions, lightPositions, lightVec3ArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaLightColors, lightColors, lightVec3ArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaLightBools, lightBools, lightBoolArrSize, cudaMemcpyHostToDevice));
	}

	void free()
	{
		delete[] spherePositions;
		delete[] sphereRadii;
		delete[] sphereAlbedos;
		delete[] lightPositions;
		delete[] lightColors;
		delete[] lightBools;

		gpuErrchk(cudaFree(cudaSpherePositions));
		gpuErrchk(cudaFree(cudaSphereAlbedos));
		gpuErrchk(cudaFree(cudaSphereRadii));
		gpuErrchk(cudaFree(cudaLightPositions));
		gpuErrchk(cudaFree(cudaLightColors));
		gpuErrchk(cudaFree(cudaLightBools));
	}

	void updateCuda()
	{
		size_t lightVec3ArrSize = lightCount * sizeof(glm::vec3);
		size_t lightBoolArrSize = lightCount * sizeof(bool);

		gpuErrchk(cudaMemcpy(cudaLightPositions, lightPositions, lightVec3ArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaLightColors, lightColors, lightVec3ArrSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(cudaLightBools, lightBools, lightBoolArrSize, cudaMemcpyHostToDevice));
	}

};