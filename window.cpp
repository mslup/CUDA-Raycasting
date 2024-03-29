#include "framework.h"

Window::Window(Application *parent)
{
	app = parent;

	width = Application::WIDTH;
	height = Application::HEIGHT;

	lastX = width / 2;
	lastY = height / 2;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwWindowHint(GLFW_SAMPLES, 4);

	wndptr = glfwCreateWindow(width, height, "Raycasting of spheres", NULL, NULL);
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

	std::cout << reinterpret_cast<const char*>(glGetString(GL_VERSION)) << std::endl;

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

	//glEnable(GL_MULTISAMPLE);

	glfwSetWindowUserPointer(wndptr, this);
	glfwSetFramebufferSizeCallback(wndptr,
		[](GLFWwindow* window, int width, int height)
		{
			Window& wnd = *(Window*)glfwGetWindowUserPointer(window);
			wnd.width = width;
			wnd.height = height;

			wnd.app->resize(width, height);

			glViewport(0, 0, width, height);
		});

	glfwSetCursorPosCallback(wndptr,
		[](GLFWwindow* window, double xposIn, double yposIn)
		{
			Window& wnd = *(Window*)glfwGetWindowUserPointer(window);

			float xpos = static_cast<float>(xposIn);
			float ypos = static_cast<float>(yposIn);

			if (wnd.firstMouse)
			{
				wnd.lastX = xpos;
				wnd.lastY = ypos;
				wnd.firstMouse = false;
			}

			float xoffset = xpos - wnd.lastX;
			float yoffset = wnd.lastY - ypos; // reversed since y-coordinates go from bottom to top
			wnd.lastX = xpos;
			wnd.lastY = ypos;

			wnd.app->processMouse(glm::vec2(xoffset, yoffset));
		});
}


void Window::processInput()
{
	if (glfwGetKey(wndptr, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(wndptr, true);

	constexpr int len = 11;
	int cameraPosKeys[len] = {
		GLFW_KEY_W,
		GLFW_KEY_S,
		GLFW_KEY_A,
		GLFW_KEY_D,
		GLFW_KEY_SPACE,
		GLFW_KEY_LEFT_SHIFT,
		GLFW_KEY_Q,
		GLFW_KEY_E,
		GLFW_KEY_1,
		GLFW_KEY_3,
		GLFW_KEY_F
	};

	for (int i = 0; i < len; i++)
	{
		if (glfwGetKey(wndptr, cameraPosKeys[i]) == GLFW_PRESS)
			app->processKeyboard(cameraPosKeys[i]);
	}
}