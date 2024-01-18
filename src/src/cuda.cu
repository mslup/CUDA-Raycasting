#include "cuda.cuh"

#define PRINT_VEC3(vec) printf("%.2f, %.2f, %.2f\n", vec.x, vec.y, vec.z);
#define PRINT_VEC4(vec) printf("%.2f, %.2f, %.2f, %.2f\n", vec.x, vec.y, vec.z, vec.w);

struct Ray;
struct HitPayload;

__device__ unsigned int toRGBA(glm::vec4& color)
{
	unsigned char r = color.r * 255.0f;
	unsigned char g = color.g * 255.0f;
	unsigned char b = color.b * 255.0f;
	unsigned char a = color.a * 255.0f;

	return (r << 24) | (g << 16) | (b << 8) | a;
}

__device__ HitPayload miss(const Ray& ray)
{
	HitPayload payload;
	payload.hitDistance = -1.0f;
	return payload;
}

__device__ HitPayload lightHit(const Ray& ray, int lightIndex)
{
	HitPayload payload;
	payload.hitDistance = 0.0f;
	payload.objectIndex = lightIndex;
	return payload;
}

__device__ HitPayload closestHit(const Ray& ray, int sphereIndex, float hitDistance, Scene scene)
{
	HitPayload payload;

	payload.hitDistance = hitDistance;
	payload.objectIndex = sphereIndex;

	glm::vec3 sphereCenter = scene.cudaSpherePositions[sphereIndex];

	payload.hitPoint = ray.origin + ray.direction * hitDistance;
	payload.normal = glm::normalize(payload.hitPoint - sphereCenter);

	return payload;
}

__device__ HitPayload traceRayFromPixel(const Ray& ray, Scene scene)
{
	int hitSphereIndex = -1;
	int hitLightIndex = -1;
	float hitDistance = FLT_MAX;

	for (int k = 0; k < scene.sphereCount; k++)
	{
		glm::vec3 origin = ray.origin - scene.cudaSpherePositions[k];
		glm::vec3 direction = ray.direction;

		float radius = scene.cudaSphereRadii[k];

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
		if (!scene.cudaLightBools[k])
			continue;

		glm::vec3 origin = ray.origin - scene.cudaLightPositions[k];
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

	return closestHit(ray, hitSphereIndex, hitDistance, scene);
}

__device__ HitPayload traceRayFromHitpoint(Ray& ray, float diff, Scene scene)
{
	int hitSphereIndex = -1;
	float hitDistance = FLT_MAX;

	for (int k = 0; k < scene.sphereCount; k++)
	{
		glm::vec3 origin = ray.origin - scene.cudaSpherePositions[k];
		glm::vec3 direction = ray.direction;

		float radius = scene.cudaSphereRadii[k];

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

	return closestHit(ray, hitSphereIndex, hitDistance, scene);
}

__device__ glm::vec4 phong(HitPayload payload, int lightIndex, Scene scene, glm::vec3 cameraPos, float d)
{
	glm::vec3 lightDir = glm::normalize(scene.cudaLightPositions[lightIndex] - payload.hitPoint);
	glm::vec3 lightColor = scene.cudaLightColors[lightIndex];
	float cosNL = glm::max(0.0f, glm::dot(lightDir, payload.normal));
	glm::vec3 reflectionVector = glm::reflect(-lightDir, payload.normal);
	glm::vec3 eyeVector = glm::normalize(cameraPos - payload.hitPoint);
	float cosVR = glm::max(0.0f, glm::dot(reflectionVector, eyeVector));

	glm::vec3 color =
		scene.params.kDiffuse * cosNL * lightColor +
		scene.params.kSpecular * glm::pow(cosVR, scene.params.kShininess) * lightColor;
	
	float attenuation = 1.0f / (1.0f + d * (scene.params.linearAtt + scene.params.quadraticAtt * d));
	color *= attenuation * scene.cudaSphereAlbedos[payload.objectIndex];

	return glm::vec4(color, 1.0f);
}

__device__ glm::vec4 rayGen(int i, int j, glm::vec3 origin,
	glm::vec3 direction, Scene scene, bool shadows)
{
	Ray ray;

	ray.origin = origin;
	ray.direction = direction;

	HitPayload payload = traceRayFromPixel(ray, scene);
	int idx = payload.objectIndex;

	// no sphere detected
	if (payload.hitDistance < 0)
		return glm::vec4(scene.params.skyColor, 1.0f);

	// light source hit
	if (payload.hitDistance == 0)
		return glm::vec4(scene.cudaLightColors[idx], 1.0f);

	glm::vec4 color = glm::vec4(scene.params.kAmbient * scene.params.ambientColor * scene.cudaSphereAlbedos[idx], 1.0f);

	// cast rays from hitpoint to light sources
	for (int lightIdx = 0; lightIdx < scene.lightCount; lightIdx++)
	{
		if (!scene.cudaLightBools[lightIdx])
			continue;

		float distanceToLight = glm::length(scene.cudaLightPositions[lightIdx] - payload.hitPoint);

		if (shadows)
		{
			Ray rayToLight;

			// cast ray a bit away from the sphere so that the ray doesn't hit it
			rayToLight.origin = payload.hitPoint + payload.normal * 1e-3f;
			rayToLight.direction = glm::normalize(scene.cudaLightPositions[lightIdx] - payload.hitPoint);

			HitPayload payloadToLight = traceRayFromHitpoint(rayToLight, distanceToLight, scene);

			// no sphere hit on path to light
			if (payloadToLight.hitDistance < 0)
				color += phong(payload, lightIdx, scene, origin, distanceToLight);
		}
		else
			color += phong(payload, lightIdx, scene, origin, distanceToLight);

	}

	return glm::clamp(color, 0.0f, 1.0f);
}

__device__ glm::vec3 getRayDirection(int i, int j, cudaArguments args)
{
	glm::vec2 coord{
		(float)j / args.width,
		(float)i / args.height
	};

	coord = coord * 2.0f - 1.0f;

	glm::vec4 target = args.inverseProjMatrix * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
	glm::vec3 rayDirection = glm::vec3(args.inverseViewMatrix * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0));

	return rayDirection;
}

__global__ void rayTrace(cudaArguments args)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= args.width || y >= args.height)
		return;

	int k = x + y * args.width;

	GLuint res = toRGBA(rayGen(y, x,
		args.rayOrigin, getRayDirection(y, x, args), 
		args.scene, args.shadows));

	args.cudaImage[k] = res;
}