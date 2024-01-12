#include "framework.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

Camera::Camera(int width, int height, float left, float right, float bottom, float top)
{
	viewportWidth = width;
	viewportHeight = height;

	this->left = left;
	this->right = right;
	this->bottom = bottom;
	this->top = top;

	position = glm::vec3(0.0f, 0.0f, -3.0f);

	calculateProjMatrix();
	calculateViewMatrix();

	calculateRayDirections(1);
}

const std::vector<glm::vec3>& Camera::getOrthographicRayOrigins()
{
	return rayOrigins;
}

const std::vector<glm::vec3>& Camera::getRayDirections()
{
	return rayDirections;
}

void Camera::calculateRayDirections(float deltaTime)
{
	rayOrigins.resize(viewportWidth * viewportHeight);
	rayDirections.resize(viewportWidth * viewportHeight);

	calculateViewMatrix();

	//std::cout << glm::to_string(viewMatrix) << std::endl;

	for (int i = 0; i < viewportWidth; ++i)
	{
		for (int j = 0; j < viewportHeight; ++j)
		{
			glm::vec2 coord{
				(float)i / viewportHeight,
				(float)j / viewportWidth
			};

			coord = coord * 2.0f - 1.0f;

			glm::vec4 origin = inverseProjMatrix * glm::vec4(coord.x, coord.y, -2.0f, 1.0f);
			glm::vec3 rayOrigin = glm::vec3(inverseViewMatrix * origin);//glm::vec4(glm::normalize(glm::vec3(origin) / origin.w), 1));

			glm::vec4 target = inverseProjMatrix * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
			glm::vec3 rayDirection = glm::vec3(inverseViewMatrix * glm::vec4(glm::vec3(target), 0)); // World space
			rayDirection = glm::normalize(rayDirection);

			rayDirection = glm::normalize(rayDirection);

			rayOrigins[i * viewportWidth + j] = rayOrigin;
			rayDirections[i * viewportWidth + j] = rayDirection;
		}
	}
	std::cout << glm::to_string(rayOrigins[0]) << std::endl;
	std::cout << glm::to_string(rayDirections[0]) << std::endl<<std::endl;
}

void Camera::calculateProjMatrix()
{
	projMatrix = glm::ortho(left, right, bottom, top);
	inverseProjMatrix = glm::inverse(projMatrix);
}

void Camera::calculateViewMatrix()
{
	viewMatrix = glm::mat4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		-position.x, -position.y, -position.z, 1.0f
	);
	inverseViewMatrix = glm::inverse(viewMatrix); //todo
}
