#ifndef CAMERA_H
#define CAMERA_H

#include <helper_math.h>

struct Camera
{
    float3 origin;
    float3 direction;

    float3 left;
    float3 right;

    float3 top;
    float3 bottom;

    float3 bottomLeft;
    float3 topRight;

    float3 leftToRight;
    float3 bottomToTop;

    float resolutionHorizontal;
    float resolutionVertical;

    float aspectRatio;
};

// Directs camera at certain point in 3d space
void SetDirection(Camera &camera, float x, float y, float z)
{
    float3 a, b;

    camera.direction.x = x - camera.origin.x;
    camera.direction.y = y - camera.origin.y;
    camera.direction.z = z - camera.origin.z;

    normalize(camera.direction);

    constexpr int EPS = 0.0001;
    if (abs(camera.direction.x) < EPS && abs(camera.direction.y) < EPS)
    {
        a = make_float3(0, -camera.direction.z, 0);
        b = make_float3(0, camera.direction.z, 0);
    }
    else
    {
        a = make_float3(camera.direction.y, -camera.direction.x, 0);
        b = make_float3(-camera.direction.y, camera.direction.x, 0);
    }

    camera.left = camera.aspectRatio * normalize(a);
    camera.right = camera.aspectRatio * normalize(b);

    camera.top = normalize(cross(camera.left, camera.direction));
    camera.bottom = normalize(cross(camera.right, camera.direction));

    camera.bottomLeft = camera.origin + (camera.left + camera.bottom + 0.007 * camera.direction);
    camera.topRight = camera.origin + (camera.right + camera.top + 0.007 * camera.direction);

    camera.leftToRight = (camera.right - camera.left);
    camera.bottomToTop = (camera.top - camera.bottom);
}

// Directs camera at certain point in 3d space
void SetDirection(Camera &camera, float3 &point)
{
    SetDirection(camera, point.x, point.y, point.z);
}

// Sets resolution of camera
void SetResolution(Camera &camera, int resolution_horizontal, int resolution_vertical)
{
    camera.resolutionHorizontal = (float)resolution_horizontal;
    camera.resolutionVertical = (float)resolution_vertical;
    camera.aspectRatio = (float)resolution_horizontal / (float)resolution_vertical;
}

#endif