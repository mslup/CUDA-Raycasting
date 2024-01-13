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
};