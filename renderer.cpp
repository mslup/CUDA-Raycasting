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
	std::vector<glm::vec3> colors;

	colors.push_back(glm::vec3(0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f));
	colors.push_back(glm::vec3(0.0f / 255.0f, 95.0f / 255.0f, 115.0f / 255.0f));
	colors.push_back(glm::vec3(10.0f / 255.0f, 147.0f / 255.0f, 150.0f / 255.0f));
	colors.push_back(glm::vec3(148.0f / 255.0f, 210.0f / 255.0f, 189.0f / 255.0f));
	colors.push_back(glm::vec3(233.0f / 255.0f, 216.0f / 255.0f, 166.0f / 255.0f));
	colors.push_back(glm::vec3(238.0f / 255.0f, 155.0f / 255.0f, 0.0f / 255.0f));
	colors.push_back(glm::vec3(202.0f / 255.0f, 103.0f / 255.0f, 2.0f / 255.0f));
	colors.push_back(glm::vec3(187.0f / 255.0f, 62.0f / 255.0f, 3.0f / 255.0f));
	colors.push_back(glm::vec3(174.0f / 255.0f, 32.0f / 255.0f, 18.0f / 255.0f));
	colors.push_back(glm::vec3(155.0f / 255.0f, 34.0f / 255.0f, 38.0f / 255.0f));

	/*scene.spheres.push_back(Sphere{
		glm::vec3(0.0f, 0.0f, 0.0f),
		0.5f,
		glm::vec3(0.8f, 0.3f, 1.0f)
		});*/

	//srand(time(NULL));

	int n = 30;

	for (int i = 0; i < n; i++)
	{
		scene.spheres.push_back(Sphere{
			glm::vec3((float)rand() / RAND_MAX * 10.0f - 5.0f,
					  (float)rand() / RAND_MAX * 10.0f - 5.0f,
					  (float)rand() / RAND_MAX * 10.0f - 5.0f),
			0.5f * (i % 3),
			colors[(i % colors.size())]
			});
	}
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
	glm::vec3 rayOrigin = camera->getRayOrigin(); //camera->getOrthographicRayOrigins()[i * width + j];
	glm::vec3 rayDirection = camera->getRayDirections()[i * width + j];




	int hitSphereIndex = -1;
	Sphere closestSphere;

	float minT = FLT_MAX;

	for (int k = 0; k < scene.spheres.size(); k++)
	{
		Sphere& sphere = scene.spheres[k];

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

		if (t > 0 && t < minT)
		{
			minT = t;
			hitSphereIndex = k;
			closestSphere = sphere;
		}
	}

	if (hitSphereIndex == -1)
		return toRGBA(glm::vec4(skyColor, 1.0f));

	glm::vec3 hitPoint = rayOrigin + rayDirection * minT;

	glm::vec3 origin = rayOrigin - closestSphere.center;
	glm::vec3 position = origin + rayDirection * minT;
	glm::vec3 normal = glm::normalize(position);

	float kDiffuse = 0.0f;
	float kSpecular = 0.7f;
	float kAmbient = 0.5f;
	float kShininess = 50;

	glm::vec3 lightPosition = glm::vec3(
		0.0f, 
		0.0f, 
		0.0f
	);
	glm::vec3 lightDir = glm::normalize(lightPosition - hitPoint);
	glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
	float cosNL = glm::max(0.0f, glm::dot(lightDir, normal));
	glm::vec3 reflectionVector = glm::normalize(2.0f * cosNL * normal - lightDir);
	glm::vec3 eyeVector = glm::normalize(camera->position - hitPoint);
	float cosVR = glm::max(0.0f, glm::dot(reflectionVector, eyeVector));

	//float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);
	glm::vec3 color = kAmbient * ambientColor + 
		kDiffuse * cosNL * lightColor +
		kSpecular * glm::pow(cosVR, kShininess) * lightColor;
	color *= closestSphere.albedo;

	return toRGBA(glm::vec4(glm::clamp(color, 0.0f, 1.0f), 1.0f));

}

void Renderer::processKeyboard(int key, float deltaTime)
{
	camera->onUpdate(key, deltaTime);
}

void Renderer::processMouse(glm::vec2 offset, float deltaTime)
{
	camera->onMouseUpdate(offset, deltaTime);
}