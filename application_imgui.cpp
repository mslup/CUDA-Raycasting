#include "application.hpp"

#include <glm/gtc/type_ptr.hpp>

void Application::imGuiFrame(int fps)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	//ImGui::ShowDemoWindow();
	if (!ImGui::Begin("Menu", NULL, 0))
	{
		ImGui::End();
		return;
	}

	ImGui::PushItemWidth(-ImGui::GetWindowWidth() * 0.45f);

	static const char* items[] = { "CPU", "GPU", "GPU shadows" };
	static int selectedItem = 1;

	if (ImGui::Combo("Solution", &selectedItem, items, IM_ARRAYSIZE(items)))
	{
		solutionMode = (solutionModes)selectedItem;
	}

	ImGui::Text("%d fps", fps);
	
	if (ImGui::CollapsingHeader("Information"))
	{
		ImGui::Text("Number of spheres: %d", renderer->scene.sphereCount);
		ImGui::Text("Number of light sources: %d", renderer->scene.lightCount);
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

	// todo maybe other function
	if (ImGui::CollapsingHeader("Lights"))
	{
		ImGui::SliderFloat("Linear attenuation", &renderer->scene.linearAtt, 0.014f, 0.7f);
		ImGui::SliderFloat("Quadratic attenuation", &renderer->scene.quadraticAtt, 0.0f, 0.5f);

		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

		ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
		ImGui::BeginChild("ChildR", ImVec2(0, 100), ImGuiChildFlags_Border | ImGuiChildFlags_ResizeY, window_flags);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Menu"))
			{
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}
		if (ImGui::BeginTable("split", 1, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings))
		{
			for (int i = 0; i < renderer->scene.lightCount; ++i)
			{
				ImGui::TableNextColumn();

				ImGui::SeparatorText(("Light " + std::to_string(i)).c_str());
				{
					float border = renderer->scene.worldBorder;
					ImGui::PushID(i);
					ImGui::DragFloat3("Position", glm::value_ptr(renderer->scene.lightPositions[i]), 0.1f, -border - 5, border + 5);
					ImGui::ColorEdit3("Color", glm::value_ptr(renderer->scene.lightColors[i]));

					bool& on = renderer->scene.lightBools[i];
					ImGui::Checkbox(on ? "Turn off" : "Turn on", &on);
				}
			}

			ImGui::EndTable();
		}
		ImGui::EndChild();
		ImGui::PopStyleVar();
	}

	if (ImGui::CollapsingHeader("Sphere material"))
	{
		ImGui::SliderFloat("Diffuse", &renderer->scene.kDiffuse, 0.0f, 1.0f);
		ImGui::SliderFloat("Specular", &renderer->scene.kSpecular, 0.0f, 1.0f);
		ImGui::SliderFloat("Shininess", &renderer->scene.kShininess, 1.0f, 100.0f);
		ImGui::SliderFloat("Ambient", &renderer->scene.kAmbient, 0.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Environment"))
	{
		ImGui::ColorEdit3("Sky color", glm::value_ptr(renderer->scene.skyColor));
		ImGui::ColorEdit3("Ambient color", glm::value_ptr(renderer->scene.ambientColor));
	}

	ImGui::End();

}