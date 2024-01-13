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

void Renderer::createScene()
{
	std::vector<glm::vec3> colors;

	//colors.push_back(glm::vec3(0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f));
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

	int n = 15;

	for (int i = 0; i < n; i++)
	{
		scene.spheres.push_back(Sphere{
			glm::vec3(i % 3 + 1, 0.0f, i % 5 + 1)
				/*glm::vec3((float)rand() / RAND_MAX * 10.0f - 5.0f,
						  (float)rand() / RAND_MAX * 10.0f - 5.0f,
						  (float)rand() / RAND_MAX * 10.0f - 5.0f)*/
			,
				0.1f * ((i + 1) % 5),
				colors[(i % colors.size())]
			});
	}
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
					imageData[i * width + j] = toRGBA(rayGen(i, j, deltaTime));
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

glm::vec4 Renderer::rayGen(int i, int j, float deltaTime)
{
	Ray ray;

	ray.origin = camera->getRayOrigin(); //camera->getOrthographicRayOrigins()[i * width + j];
	ray.direction = camera->getRayDirections()[i * width + j];

	HitPayload payload = traceRay(ray);

	// no sphere detected
	if (payload.hitDistance < 0)
		return glm::vec4(skyColor, 1.0f);

	const Sphere& sphere = scene.spheres[payload.sphereIndex];
	glm::vec4 color = glm::vec4(kAmbient * ambientColor * sphere.albedo, 1.0f);

	// for light : lights
	{
		Light light{ {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };

		Ray rayToLight;
		rayToLight.origin = payload.hitPoint + payload.normal * 0.01f;
		rayToLight.direction = glm::normalize(light.position - payload.hitPoint);

		HitPayload payloadToLight = traceRay(rayToLight);

		if (payloadToLight.hitDistance < 0)
			color += phong(payload, light);
	}

	return glm::clamp(color, 0.0f, 1.0f);
}

glm::vec4 Renderer::phong(HitPayload payload, Light light)
{
	glm::vec3 lightPosition = glm::vec3(
		0.0f,//2.5f * (float)glm::sin(glfwGetTime()),
		0.0f,//2.5f * (float)glm::cos(glfwGetTime()),
		0.0f //2.5f * (float)glm::sin(glfwGetTime())
	);
	glm::vec3 lightDir = glm::normalize(lightPosition - payload.hitPoint);
	glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
	float cosNL = glm::max(0.0f, glm::dot(lightDir, payload.normal));
	glm::vec3 reflectionVector = glm::normalize(2.0f * cosNL * payload.normal - lightDir);
	glm::vec3 eyeVector = glm::normalize(camera->position - payload.hitPoint);
	float cosVR = glm::max(0.0f, glm::dot(reflectionVector, eyeVector));

	glm::vec3 color =
		kDiffuse * cosNL * lightColor +
		kSpecular * glm::pow(cosVR, kShininess) * lightColor;

	Sphere& sphere = scene.spheres[payload.sphereIndex];
	color *= sphere.albedo;

	return glm::vec4(color, 1.0f);
}

Renderer::HitPayload Renderer::traceRay(const Ray& ray)
{
	int hitSphereIndex = -1;
	Sphere closestSphere;
	float hitDistance = FLT_MAX;

	for (int k = 0; k < scene.spheres.size(); k++)
	{
		Sphere& sphere = scene.spheres[k];

		glm::vec3 origin = ray.origin - sphere.center;
		glm::vec3 direction = ray.direction;

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(origin, direction);
		float c = glm::dot(origin, origin)
			- sphere.radius * sphere.radius;

		float delta = b * b - 4.0f * a * c;
		if (delta < 0)
			continue;

		float t = (-b - glm::sqrt(delta)) / (2.0f * a);

		if (t > 0 && t < hitDistance)
		{
			hitDistance = t;
			hitSphereIndex = k;
			closestSphere = sphere;
		}
	}

	if (hitSphereIndex == -1)
		return miss(ray);

	return closestHit(ray, hitSphereIndex, hitDistance);
}

Renderer::HitPayload Renderer::miss(const Ray& ray)
{
	HitPayload payload;
	payload.hitDistance = -1.0f;
	return payload;
}

Renderer::HitPayload Renderer::closestHit(const Ray& ray, int sphereIndex, float hitDistance)
{
	HitPayload payload;

	payload.hitDistance = hitDistance;
	payload.sphereIndex = sphereIndex;

	Sphere& sphere = scene.spheres[sphereIndex];

	payload.hitPoint = ray.origin + ray.direction * hitDistance;
	payload.normal = glm::normalize(payload.hitPoint - sphere.center);

	return payload;
}

void Renderer::processKeyboard(int key, float deltaTime)
{
	camera->onUpdate(key, deltaTime);
}

void Renderer::processMouse(glm::vec2 offset, float deltaTime)
{
	camera->onMouseUpdate(offset, deltaTime);
}