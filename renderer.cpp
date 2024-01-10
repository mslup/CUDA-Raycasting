#include "framework.h"

Renderer::Renderer(int width, int height)
{
	imageData = nullptr;
	resize(width, height);
}

Renderer::~Renderer()
{
	if (imageData != nullptr)
		delete[] imageData;
}

void Renderer::resize(int width, int height)
{
	this->width = width;
	this->height = height;

	if (imageData != nullptr)
		delete [] imageData;

	imageData = new GLuint[width * height + 1];
}

void Renderer::render()
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			imageData[i * width + j] = perPixel(i, j);
		}
	}
}

GLuint* Renderer::getImage()
{
	return imageData;
}

GLuint Renderer::perPixel(int i, int j)
{
	unsigned char r = (1.0 * i / height) * 255;
	unsigned char g = 0x00;
	unsigned char b = (1.0 * j / width) * 255;
	unsigned char a = 0xff;

	return (r << 24) | (g << 16) | (b << 8) | a;
}