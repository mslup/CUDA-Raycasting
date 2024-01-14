#include "framework.h"

#include <glm/gtc/type_ptr.hpp>

Application::Application()
{
	window = new Window(this);
	shader = new Shader("vertex.glsl", "fragment.glsl");
	renderer = new Renderer(WIDTH, HEIGHT, this);

	deltaTime = 0;
}

Application::~Application()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
}

void Application::run()
{
	createBuffers();

	shader->use();
	glfwSwapInterval(0);

	int num_frames = 0;
	int fps = 0;
	double previousTime = glfwGetTime();
	double previousFpsTime = previousTime;

	if (freeCamera)
		glfwSetInputMode(window->wndptr, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	else
		glfwSetInputMode(window->wndptr, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

	static bool firstFrame = true;

	while (!glfwWindowShouldClose(window->wndptr))
	{
		window->processInput();

		double currentTime = glfwGetTime();
		deltaTime = currentTime - previousTime;
		deltaTime = glm::min(deltaTime, 1.0 / 60.0);
		previousTime = currentTime;

		num_frames++;
		if (currentTime - previousFpsTime >= 1.0)
		{
			fps = num_frames;
			num_frames = 0;
			previousFpsTime += 1.0;
		}

		glClearColor(0, 0, 0, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		updateAndRenderScene();

		imGuiFrame(fps);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window->wndptr);
		glfwPollEvents();

		//while (true);
	}
}

void Application::resize(int width, int height)
{
	textureResizeStorage(width, height);
	renderer->resize(width, height);
}

void Application::createBuffers()
{
	float vertices[] =
	{
		// positions		 // textures
		 1.0f,  1.0f, 0.0f,  1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,	 0.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,	 0.0f, 1.0f
	};
	unsigned int indices[] =
	{
		0, 1, 3,
		1, 2, 3
	};

	unsigned int VBO, VAO, EBO;

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	createTexture();
}

void Application::createTexture()
{
	glCreateTextures(GL_TEXTURE_2D, 1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	textureResizeStorage(WIDTH, HEIGHT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glActiveTexture(0);
}

void Application::updateAndRenderScene()
{
	renderer->update(deltaTime);
	renderer->render();
	glTextureSubImage2D(texture, 0, 0, 0,
		renderer->width, renderer->height, GL_RGBA,
		GL_UNSIGNED_INT_8_8_8_8, renderer->getImage());
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Application::imGuiFrame(int fps)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	if (!ImGui::Begin("Menu", NULL, 0))
	{
		ImGui::End();
		return;
	}

	ImGui::PushItemWidth(-ImGui::GetWindowWidth() * 0.45f);

	static const char* items[] = { "CPU", "GPU" };
	static int selectedItem = 1;

	if (ImGui::Combo("Solution", &selectedItem, items, IM_ARRAYSIZE(items)))
	{
		solutionMode = (solutionModes)selectedItem;
	}

	ImGui::Text("%d fps", fps);

	ImGui::DragFloat3("Light", glm::value_ptr(renderer->scene.lightPositions[0]), 0.01f, -2.5f, 10.0f);

	ImGui::End();
}

void Application::textureResizeStorage(int width, int height)
{
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height,
		0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, NULL);
}

void Application::processKeyboard(int key)
{
	renderer->processKeyboard(key, deltaTime);
}

void Application::processMouse(glm::vec2 offset)
{
	if (freeCamera)
		renderer->processMouse(offset, deltaTime);
}