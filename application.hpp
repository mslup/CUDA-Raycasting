#include "framework.h"

class Window;

class Application
{
public:
	Application();
	~Application();

	constexpr static int WIDTH = 800;
	constexpr static int HEIGHT = 800;

	void run();
	void resize(int width, int height);

	void processKeyboard(int key);
	void processMouse(glm::vec2 offset);

private:
	Window* window;
	Shader* shader;
	Renderer* renderer;

	unsigned int texture;

	double deltaTime;
	bool freeCamera = true;

	void createTexture();
	void createBuffers();
	void imGuiFrame(int);
	void renderScene();

	void textureResizeStorage(int width, int height);
};