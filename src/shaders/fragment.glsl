#version 330 core

//in vec3 color;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D Texture;

void main()
{
	FragColor = texture(Texture, TexCoord);//
	//FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
