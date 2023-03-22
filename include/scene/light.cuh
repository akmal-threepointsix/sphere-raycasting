#ifndef LIGHT_H
#define LIGHT_H

#include "../objects/object.cuh"

struct Lights
{
    static constexpr int TotalCount = 100;

    Positions lpos;

    Colors i_m;
    Colors i_d;
    Colors i_s;
};

#endif