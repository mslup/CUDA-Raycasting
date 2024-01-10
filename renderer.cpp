#include "framework.h"

Renderer::Renderer(int width, int height)
{
	imageData = nullptr;
	resize(width, height);
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

void Renderer::render()
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			glm::vec2 coord = glm::vec2{
				2.0f * (1.0f * i / height) - 1.0f,
				2.0f * (1.0f * j / width) - 1.0f
			};

			imageData[i * width + j] = rayGen(coord);
		}
	}
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

GLuint Renderer::rayGen(glm::vec2& coord)
{
	float x = coord.x;
	float y = coord.y;
	float z = 2.0f;

	glm::vec3 rayOrigin(x, y, z); // x = y = 0 for perspective
	glm::vec3 rayDirection(x, y, -1.0f);
	rayDirection = glm::normalize(rayDirection);

	float radius = 0.5f;

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;

	float delta = b * b - 4.0f * a * c;
	if (delta < 0)
		return 0xb8a9a9ff;

	float t = (-b - glm::sqrt(delta)) / (2.0f * a);

	//glm::vec3 hitPoint = rayOrigin + rayDirection * t;
	glm::vec3 normal = glm::normalize(rayOrigin + rayDirection * t);

	glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
	float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

	glm::vec3 sphereColor(1, 0, 1);
	sphereColor *= lightIntensity;
		
	return toRGBA(glm::vec4(sphereColor, 1.0f));

}