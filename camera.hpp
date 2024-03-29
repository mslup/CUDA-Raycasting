// https://github.com/TheCherno/RayTracing/blob/master/RayTracing/src/Camera.h

#pragma once

#include "framework.h"

class Camera
{
public:
	Camera(int width, int height, float fov = glm::radians(70.0), 
		float nearPlane = 0.01, float farPlane = 100);
		//float left = -1.0f, float right = 1.0f, float bottom = -1.0f, float top = 1.0f);

	std::vector<glm::vec3>& getOrthographicRayOrigins();
	glm::vec3& getRayOrigin() { return position; }
	std::vector<glm::vec3>& getRayDirections();

	void onResize(int width, int height);
	void onUpdate(int key, float deltaTime);
	void onMouseUpdate(glm::vec2 offset, float deltaTime);

	glm::vec3 position;

	void calculateRayDirections();

private:
	float speed, rotationSpeed;

	glm::vec3 forwardDirection, rightDirection, upDirection;
	const glm::vec3 worldUpDirection = glm::vec3(0.0f, 1.0f, 0.0f);

	float fov, nearPlane, farPlane;

	//float left, right, bottom, top;
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
