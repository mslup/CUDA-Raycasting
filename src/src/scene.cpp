#include "scene.hpp"


void Scene::create()
{
	static std::vector<glm::vec3> colorPalette;
	static std::vector<glm::vec3> lightColorPalette;
	{
		//colorPalette.push_back(glm::vec3(1,1,1));
		colorPalette.push_back(glm::vec3(155.0f / 255.0f, 34.0f / 255.0f, 38.0f / 255.0f));
		colorPalette.push_back(glm::vec3(0.0f / 255.0f, 18.0f / 255.0f, 25.0f / 255.0f));
		colorPalette.push_back(glm::vec3(0.0f / 255.0f, 95.0f / 255.0f, 115.0f / 255.0f));
		colorPalette.push_back(glm::vec3(10.0f / 255.0f, 147.0f / 255.0f, 150.0f / 255.0f));
		colorPalette.push_back(glm::vec3(148.0f / 255.0f, 210.0f / 255.0f, 189.0f / 255.0f));
		colorPalette.push_back(glm::vec3(233.0f / 255.0f, 216.0f / 255.0f, 166.0f / 255.0f));
		colorPalette.push_back(glm::vec3(238.0f / 255.0f, 155.0f / 255.0f, 0.0f / 255.0f));
		colorPalette.push_back(glm::vec3(202.0f / 255.0f, 103.0f / 255.0f, 2.0f / 255.0f));
		colorPalette.push_back(glm::vec3(187.0f / 255.0f, 62.0f / 255.0f, 3.0f / 255.0f));
		colorPalette.push_back(glm::vec3(174.0f / 255.0f, 32.0f / 255.0f, 18.0f / 255.0f));

		//colorPalette.push_back(glm::vec3(128.0f / 255.0f, 93.0f / 255.0f, 147.0f / 255.0f));
		//colorPalette.push_back(glm::vec3(244.0f / 255.0f, 159.0f / 255.0f, 188.0f / 255.0f));
		//colorPalette.push_back(glm::vec3(255.0f / 255.0f, 211.0f / 255.0f, 186.0f / 255.0f));
		//colorPalette.push_back(glm::vec3(158.0f / 255.0f, 189.0f / 255.0f, 110.0f / 255.0f));
		//colorPalette.push_back(glm::vec3(22.0f / 255.0f, 152.0f / 255.0f, 115.0f / 255.0f));

		lightColorPalette.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
		lightColorPalette.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
		lightColorPalette.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
		lightColorPalette.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		lightColorPalette.push_back(glm::vec3(1.0f, 0.0f, 1.0f));
		lightColorPalette.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
		lightColorPalette.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
		lightColorPalette.push_back(glm::vec3(0.5f, 1.0f, 0.2f));
		lightColorPalette.push_back(glm::vec3(0.0f, 0.5f, 0.1f));
		lightColorPalette.push_back(glm::vec3(0.3f, 0.1f, 0.8f));
	}

	spherePositions = new glm::vec3[sphereCount];
	sphereRadii = new float[sphereCount];
	sphereAlbedos = new glm::vec3[sphereCount];

	lightPositions = new glm::vec3[lightCount];
	lightColors = new glm::vec3[lightCount];
	lightBools = new bool[lightCount];

	srand(time(NULL));
	for (int i = 0; i < sphereCount; i++)
	{
		// generate uniformly distributed points in a ball
		glm::vec3 direction = glm::normalize(glm::gaussRand(glm::vec3(0.0f), glm::vec3(1.0f)));
		float radius = glm::pow(glm::linearRand(0.0f, 1.0f), 1.0f / 3.0f) * params.worldBorder;
		spherePositions[i] = radius * direction;
		sphereRadii[i] = 0.2f * (i % 5 + 1);
		sphereAlbedos[i] = colorPalette[i % colorPalette.size()];
	}

	for (int i = 0; i < lightCount; i++)
	{
		glm::vec3 direction = glm::normalize(glm::gaussRand(glm::vec3(0.0f), glm::vec3(1.0f)));
		float radius = glm::pow(glm::linearRand(0.0f, 1.0f), 1.0f / 3.0f) * params.worldBorder;
		lightPositions[i] = radius * direction;
		lightColors[i] = lightColorPalette[i % lightColorPalette.size()];
		lightBools[i] = true;
	}

	size_t sphereVec3ArrSize = sphereCount * sizeof(glm::vec3);
	size_t sphereFloatArrSize = sphereCount * sizeof(float);
	size_t lightVec3ArrSize = lightCount * sizeof(glm::vec3);
	size_t lightBoolArrSize = lightCount * sizeof(bool);

	gpuErrchk(cudaMalloc(&cudaSpherePositions, sphereVec3ArrSize));
	gpuErrchk(cudaMalloc(&cudaSphereAlbedos, sphereVec3ArrSize));
	gpuErrchk(cudaMalloc(&cudaSphereRadii, sphereFloatArrSize));
	gpuErrchk(cudaMalloc(&cudaLightPositions, lightVec3ArrSize));
	gpuErrchk(cudaMalloc(&cudaLightColors, lightVec3ArrSize));
	gpuErrchk(cudaMalloc(&cudaLightBools, lightBoolArrSize));

	gpuErrchk(cudaMemcpy(cudaSpherePositions, spherePositions, sphereVec3ArrSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cudaSphereAlbedos, sphereAlbedos, sphereVec3ArrSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cudaSphereRadii, sphereRadii, sphereFloatArrSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cudaLightPositions, lightPositions, lightVec3ArrSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cudaLightColors, lightColors, lightVec3ArrSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cudaLightBools, lightBools, lightBoolArrSize, cudaMemcpyHostToDevice));
}

void Scene::free()
{
	delete[] spherePositions;
	delete[] sphereRadii;
	delete[] sphereAlbedos;
	delete[] lightPositions;
	delete[] lightColors;
	delete[] lightBools;

	gpuErrchk(cudaFree(cudaSpherePositions));
	gpuErrchk(cudaFree(cudaSphereAlbedos));
	gpuErrchk(cudaFree(cudaSphereRadii));
	gpuErrchk(cudaFree(cudaLightPositions));
	gpuErrchk(cudaFree(cudaLightColors));
	gpuErrchk(cudaFree(cudaLightBools));
}

void Scene::updateScene(float deltaTime)
{
	lightPositions[0] = glm::vec3(
		params.worldBorder * glm::sin(glfwGetTime()),
		params.worldBorder * glm::cos(glfwGetTime()),
		params.worldBorder * glm::sin(glfwGetTime()) * glm::cos(glfwGetTime())
	);

	dirtyLightPos = true;
}

void Scene::updateCuda()
{
	if (dirtyLightPos)
	{
		gpuErrchk(cudaMemcpy(cudaLightPositions, lightPositions, lightCount * sizeof(glm::vec3), cudaMemcpyHostToDevice));
		dirtyLightPos = false;
	}
	if (dirtyLightCol)
	{
		gpuErrchk(cudaMemcpy(cudaLightColors, lightColors, lightCount * sizeof(glm::vec3), cudaMemcpyHostToDevice));
		dirtyLightCol = false;
	}
	if (dirtyLightBools)
	{
		gpuErrchk(cudaMemcpy(cudaLightBools, lightBools, lightCount * sizeof(bool), cudaMemcpyHostToDevice));
		dirtyLightBools = false;
	}
	if (dirtySphereCol)
	{
		gpuErrchk(cudaMemcpy(cudaSphereAlbedos, sphereAlbedos, sphereCount * sizeof(glm::vec3), cudaMemcpyHostToDevice));
		dirtySphereCol = false;

	}
}

void Scene::drawImGui()
{
	if (ImGui::CollapsingHeader("Information"))
	{
		ImGui::Text("Number of spheres: %d", sphereCount);
		ImGui::Text("Number of light sources: %d", lightCount);
	}

	if (ImGui::CollapsingHeader("Camera controls"))
	{
		ImGui::SeparatorText("Position");
		ImGui::BulletText("W - forward");
		ImGui::BulletText("S - backward");
		ImGui::BulletText("A - left");
		ImGui::BulletText("R - right");
		ImGui::BulletText("Space - up");
		ImGui::BulletText("LShift - down");

		ImGui::SeparatorText("Angles");
		ImGui::BulletText("Q - look left");
		ImGui::BulletText("E - look right");
		ImGui::BulletText("1 - look up");
		ImGui::BulletText("3 - look down");

		ImGui::SeparatorText("Misc");
		ImGui::BulletText("F - toggle mouse/keyboard");
	}

	if (ImGui::CollapsingHeader("Environment"))
	{
		ImGui::ColorEdit3("Sky color", glm::value_ptr(params.skyColor));
		ImGui::ColorEdit3("Ambient color", glm::value_ptr(params.ambientColor));
	}

	if (ImGui::CollapsingHeader("Sphere material"))
	{
		ImGui::SliderFloat("Diffuse", &params.kDiffuse, 0.0f, 1.0f);
		ImGui::SliderFloat("Specular", &params.kSpecular, 0.0f, 1.0f);
		ImGui::SliderFloat("Shininess", &params.kShininess, 1.0f, 100.0f);
		ImGui::SliderFloat("Ambient", &params.kAmbient, 0.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Light sources"))
	{
		static int attenuationEnum = params.startAttIdx;

		ImGui::DragInt("Attenuation", &attenuationEnum, 0.1f, 0, 12);

		params.linearAtt = params.linearAttValues[attenuationEnum];
		params.quadraticAtt = params.quadraticAttValues[attenuationEnum];

		if (ImGui::Button("Turn all lights on", ImVec2(150, 0)))
		{
			for (int i = 0; i < lightCount; i++)
				lightBools[i] = true;
			dirtyLightBools = true;
		}
		if (ImGui::Button("Turn all lights off", ImVec2(150, 0)))
		{
			for (int i = 0; i < lightCount; i++)
				lightBools[i] = false;
			dirtyLightBools = true;
		}

		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

		ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
		ImGui::BeginChild("Child1", ImVec2(0, 600), ImGuiChildFlags_Border | ImGuiChildFlags_ResizeY, ImGuiWindowFlags_None);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Menu"))
			{
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}
		if (ImGui::BeginTable("split1", 1, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings))
		{
			for (int i = 0; i < lightCount; ++i)
			{
				ImGui::TableNextColumn();

				ImGui::SeparatorText(("Light " + std::to_string(i)).c_str());
				{
					float border = params.worldBorder;
					ImGui::PushID(i);
					if (ImGui::DragFloat3("Position", glm::value_ptr(lightPositions[i]), 0.1f, -border - 5, border + 5))
						dirtyLightPos = true;
					if (ImGui::ColorEdit3("Color", glm::value_ptr(lightColors[i])))
						dirtyLightCol = true;

					bool& on = lightBools[i];

					if (ImGui::Checkbox(on ? "Turn off" : "Turn on", &on))
						dirtyLightBools = true;
				}
			}

			ImGui::EndTable();
		}
		ImGui::EndChild();
		ImGui::PopStyleVar();

	}
}