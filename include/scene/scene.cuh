#ifndef SCENE_H
#define SCENE_H

#include "../objects/object.cuh"
#include "../objects/sphere.cuh"

#include "light.cuh"
#include "camera.cuh"

struct Scene
{
    Camera camera;
    Spheres spheres;
    Lights lights;
};


#endif
