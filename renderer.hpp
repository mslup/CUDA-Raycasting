#pragma once

#include "framework.hpp"

class Application;
class Camera;
struct Scene;

class Renderer
{
public:
	Renderer(int width, int height);
	~Renderer();

	void resize(int width, int height);
	void processKeyboard(int key, float dTime);
	void processMouse(glm::vec2 offset, float dTime);

	// todo: this should be another class's responsibility
	void update(float deltaTime);
	void renderCPU();
	void renderGPU(bool shadows);
	GLuint* getImage();
	int width, height;

	Scene scene;
private:
	Camera* camera;

	std::vector<GLuint> viewportHorizontalIter;
	std::vector<GLuint> viewportVerticalIter;

	unsigned int* imageData;
	unsigned int* cudaImage;
	glm::vec3* cudaRayDirections;

	glm::vec4 rayGen(int i, int j);
	// cast a ray and get hit information
	HitPayload traceRayFromPixel(const Ray& ray);
	HitPayload traceRayFromHitpoint(const Ray& ray, float diff);
	HitPayload lightHit(const Ray& ray, int lightIndex);
	HitPayload closestHit(const Ray& ray, int sphereIndex, float hitDistance);
	HitPayload miss(const Ray& ray);

	glm::vec4 phong(HitPayload payload, int lightIndex);

	unsigned int toRGBA(glm::vec4&);
};