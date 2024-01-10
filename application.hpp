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


	//unsigned char texture[WIDTH][HEIGHT][3];

private:
	unsigned int* data;
	Window* window;
	Shader* shader;
	unsigned int texture;

	double deltaTime;

	const glm::vec3 backColor = glm::vec3(65, 55, 46);

	void createTexture();
	void createBuffers();
	void imGuiFrame(int);
	void renderScene();
};