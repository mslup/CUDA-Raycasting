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

	speed = 0.5f;

	calculateProjMatrix();
	calculateViewMatrix();

	calculateRayDirections();
}

void Camera::onResize(int width, int height)
{
	if (viewportWidth == width && viewportHeight == height)
		return;

	viewportWidth = width;
	viewportHeight = height;

	float aspectRatio = (float)width / (float)height;
	bottom = -aspectRatio;
	top = aspectRatio;

	calculateProjMatrix();
}

void Camera::onUpdate(int key, float deltaTime)
{
	switch (key)
	{
	case GLFW_KEY_D:
		position.x += speed * deltaTime;
		break;
	case GLFW_KEY_A:
		position.x -= speed * deltaTime;
		break;
	case GLFW_KEY_SPACE:
		position.y += speed * deltaTime;
		break;
	case GLFW_KEY_LEFT_SHIFT:
		position.y -= speed * deltaTime;
		break;
	case GLFW_KEY_S:
		position.z -= speed * deltaTime;
		break;
	case GLFW_KEY_W:
		position.z += speed * deltaTime;
		break;
	}
}

std::vector<glm::vec3>& Camera::getOrthographicRayOrigins()
{
	return rayOrigins;
}

std::vector<glm::vec3>& Camera::getRayDirections()
{
	return rayDirections;
}

void Camera::calculateRayDirections()
{
	rayOrigins.resize(viewportWidth * viewportHeight);
	rayDirections.resize(viewportWidth * viewportHeight);

	calculateViewMatrix();

	for (int i = 0; i < viewportHeight; ++i)
	{
		for (int j = 0; j < viewportWidth; ++j)
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
