#include "framework.h"

#include <stdio.h> 
#include <stdlib.h>
#include <time.h> 

Application::Application()
{
	window = new Window(this);
	shader = new Shader("vertex.glsl", "fragment.glsl");
	deltaTime = 0;

	data = new unsigned int[WIDTH * HEIGHT + 1];

}

Application::~Application()
{
	delete[] data;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();

}

void Application::run()
{
	createBuffers();

	glfwSwapInterval(0);

	int num_frames = 0;
	int fps = 0;
	double previousTime = glfwGetTime();
	double previousFpsTime = previousTime;

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

		imGuiFrame(fps);

		glClearColor(0,0,0, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		renderScene();
		//calculate and render shit

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window->wndptr);
		glfwPollEvents();
	}
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
	glTextureStorage2D(texture, 1, GL_RGBA8, WIDTH, HEIGHT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			//unsigned char r = i;//(unsigned char)(((1.0 * i) / HEIGHT) * 0xff);//rand() % 0xff;
			//unsigned char g = (unsigned char)(((1.0 * j) / WIDTH) * 0xff);//rand() % 0xff;
			//unsigned char b = 0; //rand() % 0xff;
			//unsigned char a = 0xff;
			//
			//data[i * WIDTH + j] = (r << 24) | (g << 16) | (b << 8) | a;
			data[i * WIDTH + j] = 0xff00ffff;
		}
	}

	for (int i = 50; i < 150; i++)
	{
		for (int j = 50; j < 150; j++)
		{
			data[i * WIDTH + j] = 0x000000ff;
		}
	}

	data[0] = 0xff;


	glTextureSubImage2D(texture, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, data);

	glBindTextureUnit(0, texture);
}

void Application::renderScene()
{
	shader->use();

	srand(time(NULL));

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			unsigned char r =rand() % 0xff;;// (unsigned char)((1.0 * i / HEIGHT) * 0xff);//rand() % 0xff;
			unsigned char g =rand() % 0xff;;// (unsigned char)((1.0 * j / WIDTH) * 0xff); //rand() % 0xff;
			unsigned char b = rand() % 0xff;
			unsigned char a = 0xff;

			data[i * WIDTH + j] = (r << 24) | (g << 16) | (b << 8) | a;

			//std::cout << (int)r << " " << (int)g << " " << (int)b << std::endl;
		}
	}
	glTextureSubImage2D(texture, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, data);

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
	//ImGui::PushItemWidth(ImGui::GetFontSize() * -6);

	ImGui::Text("%d fps", fps);

	ImGui::Image((void*)0, ImVec2(40, 40));

	ImGui::End();
}

