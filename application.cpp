#include "framework.h"

Application::Application()
{
	window = new Window(this);
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

		glClearColor(
			backColor.r / 255.0f,
			backColor.g / 255.0f,
			backColor.b / 255.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//calculate and render shit

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window->wndptr);
		glfwPollEvents();
	}
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

	ImGui::End();
}

