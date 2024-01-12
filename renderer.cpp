#include "framework.h"

#include <execution>
#include <algorithm>
#include <glm/gtx/string_cast.hpp>

Renderer::Renderer(int width, int height)
{
	imageData = nullptr;
	resize(width, height);

	camera = new Camera(width, height);
}

Renderer::~Renderer()
{
	if (imageData != nullptr)
		delete[] imageData;
}

void Renderer::resize(int width, int height)
{
	this->width = width;
	this->height = height;

	if (imageData != nullptr)
		delete [] imageData;

	imageData = new GLuint[width * height + 1];
}

void Renderer::render(float deltaTime)
{
	camera->calculateRayDirections(deltaTime);

	//radius = glm::sin(glfwGetTime()) / 2.0f + 0.5f;

	srand(time(NULL));

	std::vector<GLuint> horizontalIter;
	std::vector<GLuint> verticalIter;

	horizontalIter.resize(width);
	verticalIter.resize(height);
	for (uint32_t i = 0; i < width; i++)
		horizontalIter[i] = i;
	for (uint32_t i = 0; i < height; i++)
		verticalIter[i] = i;

	//std::cout << glm::to_string(camera->getOrthographicRayOrigins()[0]) << std::endl;

	std::for_each(std::execution::par, verticalIter.begin(), verticalIter.end(),
		[this, deltaTime, horizontalIter](uint32_t y)
		{
			std::for_each(std::execution::par, horizontalIter.begin(), horizontalIter.end(),
			[this, y, deltaTime](uint32_t x)
				{
					imageData[x + y * width] = rayGen(y, x, deltaTime);
				});
		});

	/*for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			imageData[i * width + j] = rayGen(i, j, deltaTime);
		}
	}*/
}

GLuint* Renderer::getImage()
{
	return imageData;
}

GLuint Renderer::toRGBA(glm::vec4& color)
{
	unsigned char r = color.r * 255.0f;
	unsigned char g = color.g * 255.0f;
	unsigned char b = color.b * 255.0f;
	unsigned char a = color.a * 255.0f;

	return (r << 24) | (g << 16) | (b << 8) | a;
}

GLuint Renderer::rayGen(int i, int j, float deltaTime)
{
	glm::vec3 rayOrigin = camera->getOrthographicRayOrigins()[i * width + j];
	glm::vec3 rayDirection = camera->getRayDirections()[i * width + j];

	

	//std::cout << radius << std::endl;

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;

	float delta = b * b - 4.0f * a * c;
	if (delta < 0)
		return 0xb8a9a9ff;

	float t = (-b - glm::sqrt(delta)) / (2.0f * a);

	//glm::vec3 hitPoint = rayOrigin + rayDirection * t;
	glm::vec3 normal = glm::normalize(rayOrigin + rayDirection * t);

	glm::vec3 lightDir = glm::normalize(glm::vec3(-1, 0, -1));
	float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

	glm::vec3 sphereColor(1, 0, 1);
	sphereColor *= lightIntensity;
		
	return toRGBA(glm::vec4(sphereColor, 1.0f));

}