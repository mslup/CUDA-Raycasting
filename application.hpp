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

private:

	Window* window;

	double deltaTime;

	const glm::vec3 backColor = glm::vec3(65, 55, 46);

	void imGuiFrame(int);
};