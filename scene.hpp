#include <vector>
#include <glm/glm.hpp>

#pragma once

struct Sphere
{
	glm::vec3 center;
	float radius;
	glm::vec3 albedo;
};

struct Scene
{
	std::vector<Sphere> spheres;
};