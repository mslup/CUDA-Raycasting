#include "framework.h"

class Renderer
{
public:
	Renderer(int width, int height);
	~Renderer();
	void resize(int width, int height);
	void render(float deltaTime);
	GLuint* getImage();
	int width, height;

	Camera* camera;
private:

	GLuint* imageData;

	GLuint toRGBA(glm::vec4&);
	GLuint rayGen(int i, int j, float);

	float radius = 0.2f;
};