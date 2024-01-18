#pragma once

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct HitPayload {
	float hitDistance;
	glm::vec3 hitPoint;
	glm::vec3 normal;
	int objectIndex;
};
