#include "camera.hpp"

#include <execution>
#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

Camera::Camera(int width, int height,
	std::vector<GLuint>& viewportHorizontalIter,
	std::vector<GLuint>& viewportVerticalIter,
	float fov, float nearPlane, float farPlane)
	: viewportHorizontalIter{ viewportHorizontalIter },
	viewportVerticalIter{ viewportVerticalIter }
{
	viewportWidth = width;
	viewportHeight = height;

	this->fov = fov;
	this->nearPlane = nearPlane;
	this->farPlane = farPlane;

	position = glm::vec3(0.0f, 0.0f, 30.0f);

	forwardDirection = glm::vec3(0.0f, 0.0f, -1.0f);
	rightDirection = glm::normalize(glm::cross(forwardDirection, worldUpDirection));
	upDirection = glm::normalize(glm::cross(rightDirection, forwardDirection));

	rayDirections = new glm::vec3[width * height];

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

	delete[] rayDirections;
	rayDirections = new glm::vec3[width * height];

	calculateProjMatrix();
}

void Camera::onKeyboardUpdate(int key, float deltaTime)
{
	float rotationSpeed = speed / 3;

	switch (key)
	{
	case GLFW_KEY_D:
		position += rightDirection * speed * deltaTime;
		break;
	case GLFW_KEY_A:
		position -= rightDirection * speed * deltaTime;
		break;
	case GLFW_KEY_SPACE:
		position += upDirection * speed * deltaTime;
		break;
	case GLFW_KEY_LEFT_SHIFT:
		position -= upDirection * speed * deltaTime;
		break;
	case GLFW_KEY_S:
		position -= forwardDirection * speed * deltaTime;
		break;
	case GLFW_KEY_W:
		position += forwardDirection * speed * deltaTime;
		break;
	case GLFW_KEY_Q:
		onMouseUpdate(glm::vec2(-rotationSpeed, 0), deltaTime);
		break;
	case GLFW_KEY_E:
		onMouseUpdate(glm::vec2(rotationSpeed, 0), deltaTime);
		break;
	case GLFW_KEY_1:
		onMouseUpdate(glm::vec2(0, rotationSpeed), deltaTime);
		break;
	case GLFW_KEY_3:
		onMouseUpdate(glm::vec2(0, -rotationSpeed), deltaTime);
		break;
	}

	calculateViewMatrix();
	calculateRayDirections();
}

void Camera::onMouseUpdate(glm::vec2 offset, float deltaTime)
{
	float pitchDelta = offset.y * rotationSpeed;
	float yawDelta = offset.x * rotationSpeed;

	glm::quat q = glm::normalize(glm::cross(glm::angleAxis(pitchDelta, rightDirection),
		glm::angleAxis(-yawDelta, worldUpDirection)));

	forwardDirection = glm::normalize(glm::rotate(q, forwardDirection));
	rightDirection = glm::normalize(glm::cross(forwardDirection, worldUpDirection));
	upDirection = glm::normalize(glm::cross(rightDirection, forwardDirection));

	calculateViewMatrix();
	calculateRayDirections();
}

std::vector<glm::vec3>& Camera::getOrthographicRayOrigins()
{
	return rayOrigins;
}

glm::vec3* Camera::getRayDirections()
{
	return rayDirections;
}

void Camera::calculateRayDirections()
{
	calculateViewMatrix();

	std::for_each(std::execution::par, viewportVerticalIter.begin(), viewportVerticalIter.end(),
		[this](uint32_t i)
		{
			std::for_each(std::execution::par, viewportHorizontalIter.begin(), viewportHorizontalIter.end(),
			[this, i](uint32_t j)
				{
					glm::vec2 coord{
						(float)j / viewportWidth,
						(float)i / viewportHeight
					};

					coord = coord * 2.0f - 1.0f;

					glm::vec4 target = inverseProjMatrix * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
					glm::vec3 rayDirection = glm::vec3(inverseViewMatrix * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0));
					rayDirections[i * viewportWidth + j] = rayDirection;
				});
		});
}

void Camera::calculateProjMatrix()
{
	projMatrix = glm::perspective(fov, (float)viewportWidth / (float)viewportHeight, nearPlane, farPlane);
	inverseProjMatrix = glm::inverse(projMatrix);
}

void Camera::calculateViewMatrix()
{
	viewMatrix = glm::lookAt(position, position + forwardDirection, worldUpDirection);
	inverseViewMatrix = glm::inverse(viewMatrix);
}
