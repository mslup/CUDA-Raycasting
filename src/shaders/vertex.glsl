#version 330 core

layout(location = 0) in vec3 aVertex;
layout(location = 1) in vec2 aTexCoord;

//out vec3 color;
out vec2 TexCoord;

void main()
{
	gl_Position = vec4(aVertex, 1.0f);
	TexCoord = aTexCoord;
}