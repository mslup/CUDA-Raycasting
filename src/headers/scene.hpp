#pragma once

#include "framework.hpp"

#include <vector>
#include <string>
#include <glm/gtc/random.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

struct SceneParams {
	glm::vec3 ambientColor{ 1.0f, 1.0f, 1.0f };
	glm::vec3 skyColor{	0.25f, 0.25f, 0.25f };

	float kDiffuse = 0.9f;
	float kSpecular = 0.4f;
	float kAmbient = 0.2f;
	float kShininess = 40;

	const float linearAttValues[13] = {
		0.7f, 0.35f, 0.22f, 0.14f, 0.09f, 0.07f, 0.045f, 0.027f, 0.022f, 0.014f, 0.007f, 0.0014f };
	const float quadraticAttValues[13] = {
		1.8f, 0.44f, 0.20f, 0.07f, 0.032f, 0.017f, 0.0075f, 0.0028f, 0.0019f, 0.0007f, 0.0002f, 0.000007f };

	const int startAttIdx = 5;

	float linearAtt = linearAttValues[startAttIdx];
	float quadraticAtt = quadraticAttValues[startAttIdx];

	const float worldBorder = 20;
};

class Scene
{
public:
	glm::vec3* spherePositions;
	glm::vec3* sphereAlbedos;
	float* sphereRadii;

	glm::vec3* lightPositions;
	glm::vec3* lightColors;
	const float lightRadius = 0.2f;
	bool* lightBools;

	glm::vec3* cudaSpherePositions;
	glm::vec3* cudaSphereAlbedos;
	float* cudaSphereRadii;

	glm::vec3* cudaLightPositions;
	glm::vec3* cudaLightColors;
	bool* cudaLightBools;

	const int sphereCount = 1000;
	const int lightCount = 10;

	SceneParams params;

	void create();
	void free();
	void updateScene(float deltaTime);
	void updateCuda();
	void drawImGui();

private:
	bool dirty;
};

