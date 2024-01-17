#include "camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

Camera::Camera(int width, int height, float fov, float nearPlane, float farPlane)
{
	viewportWidth = width;
	viewportHeight = height;

	this->fov = fov;
	this->nearPlane = nearPlane;
	this->farPlane = farPlane;

	/*this->left = left;
	this->right = right;
	this->bottom = bottom;
	this->top = top;*/

	position = glm::vec3(0.0f, 0.0f, 6.0f);

	speed = 3.0f;
	rotationSpeed = 0.005f;

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

	//float aspectRatio = (float)width / (float)height;
	//bottom = -aspectRatio;
	//top = aspectRatio;

	calculateProjMatrix();
}

void Camera::onUpdate(int key, float deltaTime)
{
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
	}

	calculateViewMatrix();
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
	//rayOrigins.resize(viewportWidth * viewportHeight);
	//rayDirections.resize(viewportWidth * viewportHeight);

	calculateViewMatrix();

	for (int i = 0; i < viewportHeight; ++i)
	{
		for (int j = 0; j < viewportWidth; ++j)
		{
			glm::vec2 coord{
				(float)j / viewportWidth,
				(float)i / viewportHeight
			};

			coord = coord * 2.0f - 1.0f;

			//glm::vec4 origin = inverseProjMatrix * glm::vec4(coord.x, coord.y, -2.0f, 1.0f);
			//glm::vec3 rayOrigin = glm::vec3(inverseViewMatrix * origin);//glm::vec4(glm::normalize(glm::vec3(origin) / origin.w), 1));

			glm::vec4 target = inverseProjMatrix * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
			glm::vec3 rayDirection = glm::vec3(inverseViewMatrix * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0));
			//rayDirection = glm::normalize(rayDirection);

			/*rayOrigins[i * viewportWidth + j] = rayOrigin;*/
			rayDirections[i * viewportWidth + j] = rayDirection;
		}
	}
}

void Camera::calculateProjMatrix()
{
	/*projMatrix = glm::ortho(left, right, bottom, top);*/
	projMatrix = glm::perspective(fov, (float)viewportWidth / (float)viewportHeight, nearPlane, farPlane);
	inverseProjMatrix = glm::inverse(projMatrix);
}

void Camera::calculateViewMatrix()
{
	/*viewMatrix = glm::mat4(
		glm::vec4(rightDirection, 0.0f),
		glm::vec4(upDirection, 0.0f),
		glm::vec4(forwardDirection, 0.0f),
		glm::vec4(position, 1.0)
	);*/
	viewMatrix = glm::lookAt(position, position + forwardDirection, worldUpDirection);
	inverseViewMatrix = glm::inverse(viewMatrix);
	
	/*inverseViewMatrix = glm::mat4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		position.x, position.y, position.z, 1.0f
	);*/
}
