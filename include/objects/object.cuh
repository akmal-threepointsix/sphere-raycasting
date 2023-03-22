#ifndef OBJECT_H
#define OBJECT_H

#include <helper_math.h>
#include <array>

struct HitRecord
{
    float t;
    float3 position;
    float3 normal;
    float3 viewer;

    int i;
};

struct Beam
{
    float3 origin;
    float3 direction;
};

struct Colors
{
    int n;

    float *r;
    float *g;
    float *b;
};

struct Positions
{
    int n;

    float *x;
    float *y;
    float *z;
    float *angle;
};

struct InitSpheres {
    float initX[1000];
    float initY[1000];
    float initZ[1000];
};

// Returns coordinates of point translated by direction vector of ray
__host__ __device__ float3 point_at_parameter(const Beam &ray, float t)
{
    return ray.origin + t * ray.direction;
}

#endif