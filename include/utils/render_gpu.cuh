#ifndef RENDER_GPU_H
#define RENDER_GPU_H

#include <cuda_runtime.h>
#include <helper_math.h>

#include "../scene/scene.cuh"
#include "../scene/light.cuh"
#include "../scene/camera.cuh"

#include "../objects/object.cuh"
#include "../objects/sphere.cuh"

#include "render_gpu.cuh"
#include <float.h>
#include <helper_math.h>

// Assigns the color of pixel to bitmap array
__host__ __device__ void SetPixel(int pixelIdx, int r, int g, int b, GLubyte* bitmap)
{
	int position = pixelIdx * 3;

	bitmap[position] = r;
	bitmap[position + 1] = g;
	bitmap[position + 2] = b;
}

// Calculates color to be displayed at the point of intersection
__host__ __device__ float3 CalcPhongReflection(Lights& lights, HitRecord& hitRecord, Spheres& spheres)
{
	float3 i_a = make_float3(spheres.color.r[hitRecord.i],
		spheres.color.g[hitRecord.i],
		spheres.color.b[hitRecord.i]);

	float3 N = normalize(hitRecord.normal);
	float3 V = normalize(hitRecord.viewer);

	float3 diffuse = make_float3(0, 0, 0);
	float3 specular = make_float3(0, 0, 0);

	float k_a = spheres.ka[hitRecord.i];
	float k_d = spheres.kd[hitRecord.i];
	float k_s = spheres.ks[hitRecord.i];
	float alpha = spheres.alpha[hitRecord.i];

	float3 ka_ia = k_a * i_a;

	for (int m = 0; m < Lights::TotalCount; m++)
	{
		float3 i_m = make_float3(lights.i_m.r[m],
			lights.i_m.g[m],
			lights.i_m.b[m]);
		float3 i_d = i_a;
		float3 i_s = make_float3(lights.i_s.r[m],
			lights.i_s.g[m],
			lights.i_s.b[m]);

		float3 kd_im_id = k_d * i_m * i_d;
		float3 ks_im_is = k_s * i_m * i_s;

		float3 position = make_float3(lights.lpos.x[m], lights.lpos.y[m], lights.lpos.z[m]);
		float3 L_m = normalize(position - hitRecord.position);

		float3 R_m = normalize(2 * dot(L_m, N) * N - L_m);

		float dot_L_N = max(dot(L_m, N), 0.0);
		float dot_R_V = max(dot(R_m, V), 0.0);
		float dot_R_V_to_alpha = pow(dot_R_V, alpha);

		specular += ks_im_is * dot_R_V_to_alpha;
		diffuse += kd_im_id * dot_L_N;
	}

	float3 Ip = ka_ia + diffuse + specular;

	float r = max(min(Ip.x, 1.0f) * 255, 0.0f);
	float g = max(min(Ip.y, 1.0f) * 255, 0.0f);
	float b = max(min(Ip.z, 1.0f) * 255, 0.0f);

	return make_float3(r, g, b);
}

// Returns beam shot from camera based on its index
__host__ __device__ Beam GetBeam(const Camera& camera, float u, float v)
{
	Beam beam{};
	float3 target = camera.bottomLeft + camera.leftToRight * u / camera.resolutionHorizontal + camera.bottomToTop * v / camera.resolutionVertical;
	beam.origin = camera.origin;
	beam.direction = normalize(target - beam.origin);
	return beam;
}

__host__ __device__ float3 CalcPixelColor(Beam beam, Scene scene)
{
	HitRecord rec{};
	if (Hit(beam, scene.spheres, 0.001f, FLT_MAX, rec))
	{
		return CalcPhongReflection(scene.lights, rec, scene.spheres);
	}
	else
	{
		return { 0.0f, 0.0f, 0.0f };
	}
}

// Main kernel, considers each ray parallelly
__global__ void Render(GLubyte* bitmap, int max_x, int max_y, Scene scene)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < max_x || j < max_y) {
		int pixelIdx = j * max_x + i;

		Beam beam = GetBeam(scene.camera, i, j);

		float3 pixelColor = CalcPixelColor(beam, scene);

		SetPixel(pixelIdx, pixelColor.x, pixelColor.y, pixelColor.z, bitmap);
	}
}

#endif