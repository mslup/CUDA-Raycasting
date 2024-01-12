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
	Camera* camera;
	Scene scene;
	GLuint* imageData;

	void createScene();

	GLuint toRGBA(glm::vec4&);
	GLuint rayGen(int i, int j, float);

	float radius = 0.2f;
};