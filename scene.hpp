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
	std::vector<Sphere> spheres;
	std::vector<Light> lights;

	void create()
	{
		std::vector<glm::vec3> colors;

		//colors.push_back(glm::vec3(0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f));
		colors.push_back(glm::vec3(0.0f / 255.0f, 95.0f / 255.0f, 115.0f / 255.0f));
		colors.push_back(glm::vec3(10.0f / 255.0f, 147.0f / 255.0f, 150.0f / 255.0f));
		colors.push_back(glm::vec3(148.0f / 255.0f, 210.0f / 255.0f, 189.0f / 255.0f));
		colors.push_back(glm::vec3(233.0f / 255.0f, 216.0f / 255.0f, 166.0f / 255.0f));
		colors.push_back(glm::vec3(238.0f / 255.0f, 155.0f / 255.0f, 0.0f / 255.0f));
		colors.push_back(glm::vec3(202.0f / 255.0f, 103.0f / 255.0f, 2.0f / 255.0f));
		colors.push_back(glm::vec3(187.0f / 255.0f, 62.0f / 255.0f, 3.0f / 255.0f));
		colors.push_back(glm::vec3(174.0f / 255.0f, 32.0f / 255.0f, 18.0f / 255.0f));
		colors.push_back(glm::vec3(155.0f / 255.0f, 34.0f / 255.0f, 38.0f / 255.0f));

		//srand(time(NULL));
		int n = 15;
		switch (sphereMode)
		{
		case NEAR_CENTER:
			spheres.push_back(Sphere{
				glm::vec3(-1.0f, 0.0f, 0.0f),
				0.5f,
				glm::vec3(0.8f, 0.3f, 1.0f)
				});
			spheres.push_back(Sphere{
				glm::vec3(2.0f, 0.0f, 0.0f),
				1.5f,
				glm::vec3(0.8f, 0.3f, 1.0f)
				});
			spheres.push_back(Sphere{
				glm::vec3(-3.0f, 0.0f, 0.0f),
				1.5f,
				glm::vec3(0.8f, 0.3f, 1.0f)
				});
			spheres.push_back(Sphere{
				glm::vec3(4.0f, 0.0f, 0.0f),
				0.5f,
				glm::vec3(0.8f, 0.3f, 1.0f)
				});
			break;

		case GRID:
			for (int i = 0; i < n; i++)
			{
				spheres.push_back(Sphere{
					-glm::vec3(i % 3 + 1, 1.0f, i % 5 + 1),
						0.2f * (i % 5 + 1),
						colors[(i % colors.size())]
					});
			}
			break;
		case RANDOM:
			for (int i = 0; i < n; i++)
			{
				spheres.push_back(Sphere{
					glm::vec3((float)rand() / RAND_MAX * 10.0f - 5.0f,
							 (float)rand() / RAND_MAX * 10.0f - 5.0f,
							 (float)rand() / RAND_MAX * 10.0f - 5.0f),
						0.2f * (i % 5 + 1),
						colors[(i % colors.size())]
					});
			}
			break;
		case LINE:
			for (int i = 0; i < n; i++)
			{
				spheres.push_back(Sphere{
					glm::vec3(i + 1, 1.0f, 0.0f),
						0.2f * (i % 5 + 1),
						colors[(i % colors.size())]
					});
			}
			break;
		default:
			break;
		}

		std::vector<glm::vec3> lightColors;

		lightColors.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
		lightColors.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
		lightColors.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
		lightColors.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		lightColors.push_back(glm::vec3(1.0f, 0.0f, 1.0f));
		lightColors.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
		lightColors.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
		lightColors.push_back(glm::vec3(0.5f, 1.0f, 0.2f));
		lightColors.push_back(glm::vec3(0.0f, 0.5f, 0.1f));
		lightColors.push_back(glm::vec3(0.3f, 0.1f, 0.8f));

		n = 10;
		for (int i = 0; i < 10; i++)
		{
			lights.push_back(Light{
					glm::vec3(5.0f * (float)rand() / RAND_MAX - 2.5f,
							  5.0f * (float)rand() / RAND_MAX - 2.5f,
							  5.0f * (float)rand() / RAND_MAX - 2.5f),
					lightColors[i % lightColors.size()]
				});
		}
	}

private:
	enum sphereModes { GRID, RANDOM, LINE, NEAR_CENTER };
	sphereModes sphereMode = RANDOM;
};