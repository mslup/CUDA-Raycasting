#include "framework.h"

class Renderer
{
public:
	Renderer(int width, int height);
	~Renderer();
	void resize(int width, int height);
	void render();
	GLuint* getImage();
	int width, height;

private:
	GLuint* imageData;

	GLuint perPixel(int i, int j);
};