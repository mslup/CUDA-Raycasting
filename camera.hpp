// https://github.com/TheCherno/RayTracing/blob/master/RayTracing/src/Camera.h

#pragma once

#include "framework.h"

class Camera
{
public:
	Camera(int width, int height, float left = -1.0f, float right = 1.0f, float bottom = -1.0f, float top = 1.0f);

	std::vector<glm::vec3>& getOrthographicRayOrigins();
	std::vector<glm::vec3>& getRayDirections();

	void onResize(int width, int height);
	void onUpdate(int key, float deltaTime);

	glm::vec3 position;

	void calculateRayDirections();


private:
	float speed;

	float left, right, bottom, top;
	int viewportWidth, viewportHeight;

	void calculateProjMatrix();
	void calculateViewMatrix();

	glm::mat4 projMatrix;
	glm::mat4 viewMatrix;
	glm::mat4 inverseProjMatrix;
	glm::mat4 inverseViewMatrix;

	std::vector<glm::vec3> rayOrigins;
	std::vector<glm::vec3> rayDirections;
};