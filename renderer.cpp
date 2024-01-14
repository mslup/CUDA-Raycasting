#include "framework.h"

#include <execution>
#include <algorithm>
#include <glm/gtx/string_cast.hpp>

Renderer::Renderer(int width, int height)
{
	imageData = nullptr;

	scene.create();

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

	static float time = glfwGetTime();

	scene.lights[0].position = glm::vec3(
		//(glfwGetTime() - time), 0.0f, 0.0f
		2.5f * glm::sin(glfwGetTime()),
		2.5f * glm::cos(glfwGetTime()),
		1.5f * glm::sin(glfwGetTime())
	);//light.position;

	std::for_each(std::execution::par, verticalIter.begin(), verticalIter.end(),
		[this, deltaTime, horizontalIter](uint32_t i)
		{
			std::for_each(std::execution::par, horizontalIter.begin(), horizontalIter.end(),
			[this, i, deltaTime](uint32_t j)
				{
					imageData[i * width + j] = toRGBA(rayGen(i, j, deltaTime));
				});
		});
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

	HitPayload payload = traceRayFromPixel(ray);

	// no sphere detected
	if (payload.hitDistance < 0)
		return glm::vec4(skyColor, 1.0f);

	// light source hit
	if (payload.hitDistance == 0)
		return glm::vec4(scene.lights[payload.sphereIndex].color, 1.0f);

	const Sphere& sphere = scene.spheres[payload.sphereIndex];
	glm::vec4 color = glm::vec4(kAmbient * ambientColor * sphere.albedo, 1.0f);

	for (int i = 0; i < scene.lights.size(); i++)
	{
		const Light& light = scene.lights[i];

		Ray rayToLight;

		// cast ray a bit away from the sphere so that the ray doesn't hit it
		rayToLight.origin = payload.hitPoint + payload.normal * 1e-4f;
		float distanceToLight = glm::length(light.position - payload.hitPoint);
		rayToLight.direction = glm::normalize(light.position - payload.hitPoint);

		HitPayload payloadToLight = traceRayFromHitpoint(rayToLight, distanceToLight);

		// no sphere hit on path to light
		if (payloadToLight.hitDistance < 0)
			color += phong(payload, light);
	}

	return glm::clamp(color, 0.0f, 1.0f);
}

glm::vec4 Renderer::phong(HitPayload payload, Light light)
{
	glm::vec3 lightDir = glm::normalize(light.position - payload.hitPoint);
	glm::vec3 lightColor = light.color;
	float cosNL = glm::max(0.0f, glm::dot(lightDir, payload.normal));
	glm::vec3 reflectionVector = glm::reflect(-lightDir, payload.normal);
	glm::vec3 eyeVector = glm::normalize(camera->position - payload.hitPoint);
	float cosVR = glm::max(0.0f, glm::dot(reflectionVector, eyeVector));

	glm::vec3 color =
		kDiffuse * cosNL * lightColor +
		kSpecular * glm::pow(cosVR, kShininess) * lightColor;

	Sphere& sphere = scene.spheres[payload.sphereIndex];
	color *= sphere.albedo;

	return glm::vec4(color, 1.0f);
}

Renderer::HitPayload Renderer::traceRayFromPixel(const Ray& ray)
{
	int hitSphereIndex = -1;
	int hitLightIndex = -1;
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
		}
	}

	for (int k = 0; k < scene.lights.size(); k++)
	{
		Light& light = scene.lights[k];

		glm::vec3 origin = ray.origin - light.position;
		glm::vec3 direction = ray.direction;

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(origin, direction);
		float c = glm::dot(origin, origin)
			- 0.1f * 0.1f;

		float delta = b * b - 4.0f * a * c;
		if (delta < 0)
			continue;

		float t = (-b - glm::sqrt(delta)) / (2.0f * a);

		if (t > 0 && t < hitDistance)
		{
			hitDistance = t;
			hitSphereIndex = -2;
			hitLightIndex = k;
		}
	}

	if (hitSphereIndex == -1)
		return miss(ray);

	if (hitSphereIndex == -2)
		return lightHit(ray, hitLightIndex);

	return closestHit(ray, hitSphereIndex, hitDistance);
}

Renderer::HitPayload Renderer::traceRayFromHitpoint(const Ray& ray, float diff)
{
	int hitSphereIndex = -1;
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

		if (t > 0 && t < diff && t < hitDistance)
		{
			hitDistance = t;
			hitSphereIndex = k;
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

Renderer::HitPayload Renderer::lightHit(const Ray& ray, int lightIndex)
{
	HitPayload payload;
	payload.hitDistance = 0.0f;
	payload.sphereIndex = lightIndex;
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