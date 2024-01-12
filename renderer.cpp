#include "framework.h"

#include <execution>
#include <algorithm>
#include <glm/gtx/string_cast.hpp>

Renderer::Renderer(int width, int height)
{
	imageData = nullptr;

	createScene();

	camera = new Camera(width, height);
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
		delete[] imageData;

	imageData = new GLuint[width * height + 1];
	camera->onResize(width, height);
}

void Renderer::render(float deltaTime)
{
	camera->calculateRayDirections();

	srand(time(NULL));

	std::vector<GLuint> horizontalIter;
	std::vector<GLuint> verticalIter;

	horizontalIter.resize(width);
	verticalIter.resize(height);
	for (uint32_t i = 0; i < width; i++)
		horizontalIter[i] = i;
	for (uint32_t i = 0; i < height; i++)
		verticalIter[i] = i;

	std::for_each(std::execution::par, verticalIter.begin(), verticalIter.end(),
		[this, deltaTime, horizontalIter](uint32_t i)
		{
			std::for_each(std::execution::par, horizontalIter.begin(), horizontalIter.end(),
			[this, i, deltaTime](uint32_t j)
				{
					imageData[i * width + j] = rayGen(i, j, deltaTime);
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

void Renderer::createScene()
{
	scene.spheres.push_back(Sphere{
		glm::vec3(0.0f, 0.0f, 0.0f),
		0.5f,
		glm::vec3(1.0f, 0.0f, 0.0f)
		});
	scene.spheres.push_back(Sphere{
		glm::vec3(0.0f, 0.0f, -1.0f),
		0.25f,
		glm::vec3(0.0f, 0.0f, 1.0f)
		});
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

	int hitSphereIndex = -1;
	Sphere closestSphere;

	float minT = FLT_MAX;

	for (int k = 0; k < scene.spheres.size(); k++)
	{
		Sphere &sphere = scene.spheres[k];

		glm::vec3 origin = rayOrigin - sphere.center;
		glm::vec3 direction = rayDirection;

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(origin, direction);
		float c = glm::dot(origin, origin) 
			- sphere.radius * sphere.radius;

		float delta = b * b - 4.0f * a * c;
		if (delta < 0)
			continue;

		float t = (-b - glm::sqrt(delta)) / (2.0f * a);

		if (t> 0 && t < minT)
		{
			minT = t;
			hitSphereIndex = k;
			closestSphere = sphere;
		}
	}

	if (hitSphereIndex == -1)
		return 0xb8a9a9ff;

	//glm::vec3 hitPoint = rayOrigin + rayDirection * t;

	glm::vec3 origin = rayOrigin - closestSphere.center;
	glm::vec3 position = origin + rayDirection * minT;
	glm::vec3 normal = glm::normalize(position);

	glm::vec3 lightDir = glm::normalize(glm::vec3(0, 0, -1));
	float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

	glm::vec3 sphereColor = closestSphere.albedo;

	return toRGBA(glm::vec4(sphereColor * lightIntensity, 1.0f));

}

void Renderer::processKeyboard(int key, float deltaTime)
{
	camera->onUpdate(key, deltaTime);
}

void Renderer::processMouse(glm::vec2 offset, float deltaTime)
{
	camera->onMouseUpdate(offset, deltaTime);
}