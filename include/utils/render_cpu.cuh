#ifndef RENDER_CPU_H
#define RENDER_CPU_H

#include "../scene/scene.cuh"
#include "../scene/light.cuh"
#include "../scene/camera.cuh"

#include "../objects/object.cuh"
#include "../objects/sphere.cuh"

#include <float.h>
#include <helper_math.h>

// Main kernel, considers each ray parallelly
void RenderCPU(unsigned char* pixels, int max_x, int max_y, Scene& scene)
{
	for (int i = 0; i < max_x; i++)
	{
		for (int j = 0; j < max_y; j++)
		{
			int pixel_index = j * max_x + i;

			float u = float(i);
			float v = float(j);

			Beam beam = GetBeam(scene.camera, u, v);

			float3 c = CalcPixelColor(beam, scene);

			SetPixel(pixel_index, (int)c.x, (int)c.y, (int)c.z, pixels);
		}
	}
}

#endif