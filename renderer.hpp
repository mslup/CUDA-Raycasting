#include "framework.h"

class Renderer
{
public:
	Renderer(int width, int height);
	~Renderer();

	void resize(int width, int height);
	void processKeyboard(int key, float dTime);
	void processMouse(glm::vec2 offset, float dTime);
	void render(float deltaTime);
	GLuint* getImage();
	int width, height;

private:
	struct HitPayload {
		float hitDistance;
		glm::vec3 hitPoint;
		glm::vec3 normal;
		int sphereIndex;
	};

	struct Ray {
		glm::vec3 origin;
		glm::vec3 direction;
	};

	Camera* camera;
	Scene scene;
	GLuint* imageData;

	void createScene();

	float kDiffuse = 0.4f;
	float kSpecular = 0.9f;
	float kAmbient = 0.2f;
	float kShininess = 10;

	const glm::vec3 ambientColor{
		1.0f, 1.0f, 1.0f
	};
	const glm::vec3 skyColor{ 
		0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f };

	GLuint toRGBA(glm::vec4&);

	glm::vec4 rayGen(int i, int j, float);

	// cast a ray and get hit information
	HitPayload traceRay(const Ray& ray);
	HitPayload closestHit(const Ray& ray, int sphereIndex, float hitDistance);
	HitPayload miss(const Ray& ray);

	glm::vec4 phong(HitPayload payload, Light light);
};