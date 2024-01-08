#include "framework.h"

Window::Window(Application *parent)
{
	app = parent;

	width = Application::WIDTH;
	height = Application::HEIGHT;

	lastX = width / 2;
	lastY = height / 2;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwWindowHint(GLFW_SAMPLES, 4);

	wndptr = glfwCreateWindow(width, height, "Shoal of fish", NULL, NULL);
	if (wndptr == NULL)
	{
		std::cout << "Failed to create a window\n";
		glfwTerminate();
	}
	glfwMakeContextCurrent(wndptr);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initalize GLAD" << std::endl;
	}

	glViewport(0, 0, width, height);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LESS);


	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui_ImplGlfw_InitForOpenGL(wndptr, true);
	ImGui_ImplOpenGL3_Init();

	glEnable(GL_MULTISAMPLE);
}


void Window::processInput()
{
	if (glfwGetKey(wndptr, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(wndptr, true);

}