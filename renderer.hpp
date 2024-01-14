#include "framework.h"

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
	void render();
	GLuint* getImage();
	int width, height;

	Scene scene;
private:
	struct HitPayload {
		float hitDistance;
		glm::vec3 hitPoint;
		glm::vec3 normal;
		int objectIndex;
	};

	struct Ray {
		glm::vec3 origin;
		glm::vec3 direction;
	};

	Camera* camera;
	GLuint* imageData;

	float kDiffuse = 0.9f;
	float kSpecular = 0.4f;
	float kAmbient = 0.2f;
	float kShininess = 40;

	const glm::vec3 ambientColor{
		1.0f, 1.0f, 1.0f
	};
	const glm::vec3 skyColor{
		0.0f, 0.0f, 0.0f };

	GLuint toRGBA(glm::vec4&);

	glm::vec4 rayGen(int i, int j);
	// cast a ray and get hit information
	HitPayload traceRayFromPixel(const Ray& ray);
	HitPayload traceRayFromHitpoint(const Ray& ray, float diff);
	HitPayload lightHit(const Ray& ray, int lightIndex);
	HitPayload closestHit(const Ray& ray, int sphereIndex, float hitDistance);
	HitPayload miss(const Ray& ray);

	glm::vec4 phong(HitPayload payload, int lightIndex);
};