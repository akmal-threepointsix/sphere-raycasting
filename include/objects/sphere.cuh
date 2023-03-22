#ifndef SPHERE_H
#define SPHERE_H

#include "object.cuh"
#include <helper_math.h>

struct Spheres
{
	static constexpr int TotalCount = 1000;

	Positions pos;
	Colors color;

	float* radius;

	float* ka;
	float* kd;
	float* ks;

	float* alpha;
};

// Calculates point of intersection between line and sphere of index i
__host__ __device__ bool Hit(const Beam& r, Spheres spheres, int i, float t_min, float t_max, HitRecord& rec)
{
	float radius = spheres.radius[i];

	float3 center = make_float3(spheres.pos.x[i], spheres.pos.y[i], spheres.pos.z[i]);
	float3 oc = r.origin - center;

	float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;

	rec.viewer = r.direction;
	rec.viewer = -rec.viewer;
	rec.i = i;

	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.position = point_at_parameter(r, rec.t);
			rec.normal = (rec.position - center) / radius;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.position = point_at_parameter(r, rec.t);
			rec.normal = (rec.position - center) / radius;
			return true;
		}
	}
	return false;
}

// Returns data about closest point of intersection between spheres and one ray
__host__ __device__ bool Hit(Beam& r, Spheres spheres, float t_min, float t_max, HitRecord& rec)
{
	HitRecord temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;
	for (int i = 0; i < Spheres::TotalCount; i++)
	{
		if (Hit(r, spheres, i, t_min, closest_so_far, temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

#endif