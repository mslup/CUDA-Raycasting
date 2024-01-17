#pragma once

#include "framework.hpp"

class Window;
class Shader;
class Renderer;

class Application
{
public:
	Application();
	~Application();

	constexpr static int WIDTH =  800;
	constexpr static int HEIGHT = 800;

	void run();
	void resize(int width, int height);

	void processKeyboard(int key);
	void processMouse(glm::vec2 offset);

	enum solutionModes {CPU, GPU, GPUshadows};
	solutionModes solutionMode = GPU;
private:
	Window* window;
	Shader* shader;
	Renderer* renderer;

	unsigned int texture;

	double deltaTime;
	bool freeCamera = false;

	void createTexture();
	void createBuffers();
	void imGuiFrame(int);
	void updateAndRenderScene();

	void textureResizeStorage(int width, int height);

	bool pause = false;

};