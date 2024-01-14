#include <vector>
#include <glm/glm.hpp>

#pragma once

struct Sphere
{
	glm::vec3 center;
	float radius;
	glm::vec3 albedo;
};

struct Light
{
	glm::vec3 position;
	//const float radius = 0.2f;
	glm::vec3 color;
};

struct Scene
{
	//std::vector<Sphere> spheres;
	//std::vector<Light> lights;

	glm::vec3* spherePositions;
	float* sphereRadii;
	glm::vec3* sphereAlbedos;

	glm::vec3* lightPositions;
	const float lightRadius = 0.2f;
	glm::vec3* lightColors;

	const int sphereCount = 15;
	const int lightCount = 10;

	void create()
	{
		static std::vector<glm::vec3> colorPalette;
		static std::vector<glm::vec3> lightColorPalette;
		{
			colorPalette.push_back(glm::vec3(0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f));
			colorPalette.push_back(glm::vec3(0.0f / 255.0f, 95.0f / 255.0f, 115.0f / 255.0f));
			colorPalette.push_back(glm::vec3(10.0f / 255.0f, 147.0f / 255.0f, 150.0f / 255.0f));
			colorPalette.push_back(glm::vec3(148.0f / 255.0f, 210.0f / 255.0f, 189.0f / 255.0f));
			colorPalette.push_back(glm::vec3(233.0f / 255.0f, 216.0f / 255.0f, 166.0f / 255.0f));
			colorPalette.push_back(glm::vec3(238.0f / 255.0f, 155.0f / 255.0f, 0.0f / 255.0f));
			colorPalette.push_back(glm::vec3(202.0f / 255.0f, 103.0f / 255.0f, 2.0f / 255.0f));
			colorPalette.push_back(glm::vec3(187.0f / 255.0f, 62.0f / 255.0f, 3.0f / 255.0f));
			colorPalette.push_back(glm::vec3(174.0f / 255.0f, 32.0f / 255.0f, 18.0f / 255.0f));
			colorPalette.push_back(glm::vec3(155.0f / 255.0f, 34.0f / 255.0f, 38.0f / 255.0f));

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

		//srand(time(NULL));
		for (int i = 0; i < sphereCount; i++)
		{
			spherePositions[i] = glm::vec3(
				(float)rand() / RAND_MAX * 10.0f - 5.0f,
				(float)rand() / RAND_MAX * 10.0f - 5.0f,
				(float)rand() / RAND_MAX * 10.0f - 5.0f);
			sphereRadii[i] = 0.2f * (i % 5 + 1);
			sphereAlbedos[i] = colorPalette[i % colorPalette.size()];
		}

		for (int i = 0; i < lightCount; i++)
		{
			lightPositions[i] = glm::vec3(
					5.0f * (float)rand() / RAND_MAX - 2.5f,
					5.0f * (float)rand() / RAND_MAX - 2.5f,
					5.0f * (float)rand() / RAND_MAX - 2.5f);
			lightColors[i] = lightColorPalette[i % lightColorPalette.size()];
		}
	}

};