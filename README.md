# Sphere raycasting in CUDA

## Goal

There are beams sent from the user’s eye perpendicularly to the screen, one beam goes
through one pixel. Color of the pixel must be calculated when the beam hits the
closest sphere (you may create a z-buffer). At the hit point hit angle can be easily
calculated having the sphere position and radius.
Phong reflection model is described for example here:
https://en.wikipedia.org/wiki/Phong_reflection_model. In raycasting viewer
direction vector is parallel to Z axis.

Input: Random set of colored spheres (at least 1k) and color sources of light in 3d
space (at least 10).

Output: Graphics displayed in a window, user can rotate the scene or lights.

## Prerequisites
- Windows (recommended version >= 10)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (recommended version >= 12)

### Optional
- [GnuWin](https://gnuwin32.sourceforge.net/install.html) (if you are crazy enough to use `make` on Windows)

## Usage

### Microsoft Visual Studio
1. Open project solution `sphere-raycasting.sln`
2. Add `lib/freeglut.lib` as an additional dependency:
    - Open `Project->Properties`;
    - Open `Linker->Additional Library Directories` and add `lib`;
    - Open `Linker->Input->Additional Dependencies` and add `freeglut.lib`.
3. Start without debugging (`Ctrl+F5`). Note: Use only `Release` version, otherwise the program will run very slow.

### Makefile
1. Open project folder in the command prompt
2. Build the project:
```
$ make
```
3. Run the project:
```
$ main.exe
``` 

## Screenshots
![Application](docs/application.png)
![FPS in command prompt](docs/FPS.png)

## Notes
- Inspired by [jtarkowski27](https://github.com/jtarkowski27/sphere-raycasting)
