#include "renderer.hpp"
#include "cuda.cuh"

#include <execution>
#include <algorithm>
#include <glm/gtx/string_cast.hpp>

Renderer::Renderer(int width, int height)
{
	imageData = nullptr;

	scene.create();

	gpuErrchk(cudaMalloc(&cudaImage, width * height * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc(&cudaRayDirections, width * height * sizeof(glm::vec3)));

	camera = new Camera(width, height, viewportHorizontalIter, viewportVerticalIter);
	resize(width, height);
}

Renderer::~Renderer()
{
	scene.free();

	if (imageData != nullptr)
		delete[] imageData;

	gpuErrchk(cudaFree(cudaImage));
	gpuErrchk(cudaFree(cudaRayDirections));
}

void Renderer::resize(int width, int height)
{
	this->width = width;
	this->height = height;

	if (imageData != nullptr)
	{
		delete[] imageData;
		gpuErrchk(cudaFree(cudaImage));
	}

	gpuErrchk(cudaFree(cudaRayDirections));

	imageData = new unsigned int[width * height];
	gpuErrchk(cudaMalloc(&cudaImage, width * height * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc(&cudaRayDirections, width * height * sizeof(glm::vec3)));

	viewportHorizontalIter.resize(width);
	viewportVerticalIter.resize(height);
	for (uint32_t i = 0; i < width; i++)
		viewportHorizontalIter[i] = i;
	for (uint32_t i = 0; i < height; i++)
		viewportVerticalIter[i] = i;

	camera->onResize(width, height);
}

void Renderer::notifyCamera()
{
	camera->calculateRayDirections();
}

void Renderer::renderCPU()
{
	std::for_each(std::execution::par, viewportVerticalIter.begin(), viewportVerticalIter.end(),
		[this](uint32_t i)
		{
			std::for_each(std::execution::par, viewportHorizontalIter.begin(), viewportHorizontalIter.end(),
			[this, i](uint32_t j)
				{
					imageData[i * width + j] = toRGBA(rayGen(i, j));
				});
		});
}

void Renderer::renderGPU(bool shadows)
{
	int pixelsCount = width * height;
	int size = pixelsCount * sizeof(unsigned int);
	size_t vecSize = pixelsCount * sizeof(glm::vec3);

	int tx = 32;
	int ty = 32;

	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);
	
	cudaArguments args{
		cudaImage, width, height, scene, 
		camera->getRayOrigin(),
		cudaRayDirections,
		camera->getRayOrigin(),
		camera->getInverseProjMatrix(),
		camera->getInverseViewMatrix(),
		shadows
	};

	callKernels(blocks, threads, args);

	gpuErrchk(cudaMemcpy(imageData, cudaImage, size, cudaMemcpyDeviceToHost));
}

GLuint* Renderer::getImage()
{
	return imageData;
}

glm::vec4 Renderer::rayGen(int i, int j)
{
	Ray ray;
	ray.origin = camera->getRayOrigin(); 
	ray.direction = camera->getRayDirections()[i * width + j];

	HitPayload payload = traceRayFromPixel(ray);
	int idx = payload.objectIndex;

	// no sphere detected
	if (payload.hitDistance < 0)
		return glm::vec4(scene.params.skyColor, 1.0f);

	// light source hit
	if (payload.hitDistance == 0)
		return glm::vec4(scene.lightColors[idx], 1.0f);

	glm::vec4 color = glm::vec4(scene.params.kAmbient 
		* scene.params.ambientColor 
		* scene.sphereAlbedos[idx], 1.0f);

	// cast rays from hitpoint to light sources
	for (int lightIdx = 0; lightIdx < scene.lightCount; lightIdx++)
	{
		float distanceToLight = glm::length(scene.lightPositions[lightIdx] - payload.hitPoint);
		//if (!scene.lightBools[lightIdx])
		//	continue;

		//Ray rayToLight;

		//// cast ray a bit away from the sphere so that the ray doesn't hit it
		//rayToLight.origin = payload.hitPoint + payload.normal * 1e-4f;
		//rayToLight.direction = glm::normalize(scene.lightPositions[lightIdx] - payload.hitPoint);

		//HitPayload payloadToLight = traceRayFromHitpoint(rayToLight, distanceToLight);

		//// no sphere hit on path to light
		//if (payloadToLight.hitDistance < 0)
			color += phong(payload, lightIdx, distanceToLight);
	}

	return glm::clamp(color, 0.0f, 1.0f);
}

glm::vec4 Renderer::phong(HitPayload payload, int lightIndex, float d)
{
	glm::vec3 lightDir = glm::normalize(scene.lightPositions[lightIndex] - payload.hitPoint);
	glm::vec3 lightColor = scene.lightColors[lightIndex];
	float cosNL = glm::max(0.0f, glm::dot(lightDir, payload.normal));
	glm::vec3 reflectionVector = glm::reflect(-lightDir, payload.normal);
	glm::vec3 eyeVector = glm::normalize(camera->position - payload.hitPoint);
	float cosVR = glm::max(0.0f, glm::dot(reflectionVector, eyeVector));

	glm::vec3 color =
		scene.params.kDiffuse * cosNL * lightColor +
		scene.params.kSpecular * glm::pow(cosVR, scene.params.kShininess) * lightColor;

	float attenuation = 1.0f / (1.0f + d * (scene.params.linearAtt + scene.params.quadraticAtt * d));
	color *= attenuation * scene.sphereAlbedos[payload.objectIndex];

	return glm::vec4(color, 1.0f);
}

// todo: merge two traceray functions
HitPayload Renderer::traceRayFromPixel(const Ray& ray)
{
	int hitSphereIndex = -1;
	int hitLightIndex = -1;
	float hitDistance = FLT_MAX;

	for (int k = 0; k < scene.sphereCount; k++)
	{
		glm::vec3 origin = ray.origin - scene.spherePositions[k];
		glm::vec3 direction = ray.direction;

		float radius = scene.sphereRadii[k];

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(origin, direction);
		float c = glm::dot(origin, origin)
			- radius * radius;

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

	for (int k = 0; k < scene.lightCount; k++)
	{
		glm::vec3 origin = ray.origin - scene.lightPositions[k];
		glm::vec3 direction = ray.direction;

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(origin, direction);
		float c = glm::dot(origin, origin)
			- scene.lightRadius * scene.lightRadius;

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

HitPayload Renderer::traceRayFromHitpoint(const Ray& ray, float diff)
{
	int hitSphereIndex = -1;
	float hitDistance = FLT_MAX;

	for (int k = 0; k < scene.sphereCount; k++)
	{
		glm::vec3 origin = ray.origin - scene.spherePositions[k];
		glm::vec3 direction = ray.direction;

		float radius = scene.sphereRadii[k];

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(origin, direction);
		float c = glm::dot(origin, origin)
			- radius * radius;

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

HitPayload Renderer::miss(const Ray& ray)
{
	HitPayload payload;
	payload.hitDistance = -1.0f;
	return payload;
}

HitPayload Renderer::lightHit(const Ray& ray, int lightIndex)
{
	HitPayload payload;
	payload.hitDistance = 0.0f;
	payload.objectIndex = lightIndex;
	return payload;
}

HitPayload Renderer::closestHit(const Ray& ray, int sphereIndex, float hitDistance)
{
	HitPayload payload;

	payload.hitDistance = hitDistance;
	payload.objectIndex = sphereIndex;

	glm::vec3 sphereCenter = scene.spherePositions[sphereIndex];

	payload.hitPoint = ray.origin + ray.direction * hitDistance;
	payload.normal = glm::normalize(payload.hitPoint - sphereCenter);

	return payload;
}

void Renderer::processKeyboard(int key, float deltaTime)
{
	camera->onKeyboardUpdate(key, deltaTime);
}

void Renderer::processMouse(glm::vec2 offset, float deltaTime)
{
	camera->onMouseUpdate(offset, deltaTime);
}

unsigned int Renderer::toRGBA(glm::vec4& color) {
	unsigned char r = color.r * 255.0f;
	unsigned char g = color.g * 255.0f;
	unsigned char b = color.b * 255.0f;
	unsigned char a = color.a * 255.0f;

	return (r << 24) | (g << 16) | (b << 8) | a;
}