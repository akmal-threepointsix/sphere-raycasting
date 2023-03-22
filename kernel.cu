#define RENDER_GPU

// OpenGL
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA core
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

// CUDA helpers
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_timer.h>

// STL
#include <string>
#include <cmath>
#include <windows.h>

// My headers
#include "scene/scene.cuh"
#include "utils/render_gpu.cuh"
#include "utils/render_cpu.cuh"

bool renderWithGpu = true;

// Screen
constexpr int ScreenWidth = 1600;
constexpr int ScreenHeight = 900;

// Display image
GLubyte h_Bitmap[ScreenWidth * ScreenHeight * 3];
GLubyte* d_Bitmap;

constexpr int PixelsCount = ScreenWidth * ScreenHeight;
constexpr size_t BitmapSize = PixelsCount * sizeof(GLubyte) * 3;

// Scene
Scene h_Scene;
Scene d_Scene;

// Rotation
float angle_x = 0;
float angle_y = 0;

int start_x = -1;
int start_y = -1;

float start_angle_x = 0;
float start_angle_y = 0;

float prev_angle_x = 0;
float prev_angle_y = 0;

// FPS Timer
StopWatchInterface* fps_timer = NULL;

clock_t start, stop;
clock_t second_start, second_stop;

int frames;
int fpsCount = 0;
float avgFPS = 1.0f;
int fpsLimit = 1;
double first_second = 2;

char fps[512];

// Source: https://stackoverflow.com/questions/5289613/generate-random-float-between-two-floats
float RandFloat(float min, float max)
{
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = max - min;
	float r = random * diff;
	return min + r;
}

// glutDisplayFunc() event handler
void DisplayHandler()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	glDrawPixels(ScreenWidth, ScreenHeight, GL_RGB, GL_UNSIGNED_BYTE, h_Bitmap);
	glutSwapBuffers();
}

InitSpheres initSpheres;
// Generate one random 3D point and assign its coordinates at index i
void assign_position(Positions& positions, int i)
{
	float x = RandFloat(-150, 150);
	float y = RandFloat(-150, 150);

	float dist = sqrt(x * x + y * y);

	float s_x = x / dist;
	float s_y = y / dist;

	positions.angle[i] = atan2(s_y, s_x);

	initSpheres.initX[i] = positions.x[i] = x;
	initSpheres.initY[i] = positions.y[i] = y;
	initSpheres.initZ[i] = positions.z[i] = RandFloat(-100, 100);
}

// Generate one random 3D point and assign its coordinates at index i
void assign_position_lights(Positions& positions, int i)
{
	float x = RandFloat(-200, 200);
	float y = RandFloat(-200, 200);

	float dist = sqrt(x * x + y * y);

	float s_x = x / dist;
	float s_y = y / dist;

	positions.angle[i] = atan2(s_y, s_x);

	positions.x[i] = x;
	positions.y[i] = y;
	positions.z[i] = RandFloat(-100, 100);
}

// Generate random spheres and lights
void GenerateScene()
{
	// Camera
	h_Scene.camera.origin = make_float3(-800, 450, 2500);
	SetResolution(h_Scene.camera, ScreenWidth, ScreenHeight);
	SetDirection(h_Scene.camera, 0, 0, 0);

	d_Scene.camera.origin = make_float3(-800, 450, 2500);
	SetResolution(d_Scene.camera, ScreenWidth, ScreenHeight);
	SetDirection(d_Scene.camera, 0, 0, 0);

	// Spheres
	for (int i = 0; i < Spheres::Spheres::TotalCount; i++)
	{
		// Random position
		assign_position(h_Scene.spheres.pos, i);

		// Random radius
		h_Scene.spheres.radius[i] = RandFloat(2, 6);

		// Random reflection coefficients
		h_Scene.spheres.ka[i] = RandFloat(0, 0.2);
		h_Scene.spheres.kd[i] = RandFloat(0, 0.03);
		h_Scene.spheres.ks[i] = RandFloat(0, 0.6);
		h_Scene.spheres.alpha[i] = (rand() % 400) + 10;

		// Random color
		h_Scene.spheres.color.r[i] = RandFloat(0, 1);
		h_Scene.spheres.color.g[i] = RandFloat(0, 1);
		h_Scene.spheres.color.b[i] = RandFloat(0, 1);
	}

	// Lights
	for (int i = 0; i < Lights::TotalCount; i++)
	{
		// Random position
		assign_position_lights(h_Scene.lights.lpos, i);

		h_Scene.lights.i_m.r[i] = RandFloat(0, 1);
		h_Scene.lights.i_m.g[i] = RandFloat(0, 1);
		h_Scene.lights.i_m.b[i] = RandFloat(0, 1);

		// Random intensity of the diffuse component of the light source
		h_Scene.lights.i_d.r[i] = RandFloat(0, 1);
		h_Scene.lights.i_d.g[i] = RandFloat(0, 1);
		h_Scene.lights.i_d.b[i] = RandFloat(0, 1);

		// Random intensity of the specular component of the light source
		h_Scene.lights.i_s.r[i] = RandFloat(0, 1);
		h_Scene.lights.i_s.g[i] = RandFloat(0, 1);
		h_Scene.lights.i_s.b[i] = RandFloat(0, 1);
	}
}

// Rotate list of positions
void Rotate(Positions* positions, int n, float rotate)
{
	for (int i = 0; i < n; i++)
	{
		float x = positions->x[i];
		float y = positions->y[i];

		float angle = positions->angle[i] + rotate;
		float dist = sqrt(x * x + y * y);

		positions->x[i] = dist * cos(angle);
		positions->y[i] = dist * sin(angle);

		positions->angle[i] = angle;
	}
}

// Copy data from host to device
void CopyHostToDevice()
{
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.radius, h_Scene.spheres.radius, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.spheres.ka, h_Scene.spheres.ka, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.kd, h_Scene.spheres.kd, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.ks, h_Scene.spheres.ks, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.alpha, h_Scene.spheres.alpha, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.spheres.pos.x, h_Scene.spheres.pos.x, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.pos.y, h_Scene.spheres.pos.y, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.pos.z, h_Scene.spheres.pos.z, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.spheres.color.r, h_Scene.spheres.color.r, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.color.g, h_Scene.spheres.color.g, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.spheres.color.b, h_Scene.spheres.color.b, sizeof(float) * Spheres::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.lights.lpos.x, h_Scene.lights.lpos.x, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.lpos.y, h_Scene.lights.lpos.y, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.lpos.z, h_Scene.lights.lpos.z, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_m.r, h_Scene.lights.i_m.r, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_m.g, h_Scene.lights.i_m.g, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_m.b, h_Scene.lights.i_m.b, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_d.r, h_Scene.lights.i_d.r, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_d.g, h_Scene.lights.i_d.g, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_d.b, h_Scene.lights.i_d.b, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_s.r, h_Scene.lights.i_s.r, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_s.g, h_Scene.lights.i_s.g, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Scene.lights.i_s.b, h_Scene.lights.i_s.b, sizeof(float) * Lights::TotalCount, cudaMemcpyHostToDevice));
}

// Allocate memory for scene objects
void InitScene()
{
	sdkCreateTimer(&fps_timer);

	h_Scene.spheres.radius = new float[Spheres::TotalCount];

	// Host scene spheres positions and angle
	h_Scene.spheres.pos.x = new float[Spheres::TotalCount];
	h_Scene.spheres.pos.y = new float[Spheres::TotalCount];
	h_Scene.spheres.pos.z = new float[Spheres::TotalCount];
	h_Scene.spheres.pos.angle = new float[Spheres::TotalCount];

	// Host scene spheres colors
	h_Scene.spheres.color.r = new float[Spheres::TotalCount];
	h_Scene.spheres.color.g = new float[Spheres::TotalCount];
	h_Scene.spheres.color.b = new float[Spheres::TotalCount];

	// Host scene spheres reflection constants
	h_Scene.spheres.ka = new float[Spheres::TotalCount];
	h_Scene.spheres.kd = new float[Spheres::TotalCount];
	h_Scene.spheres.ks = new float[Spheres::TotalCount];
	h_Scene.spheres.alpha = new float[Spheres::TotalCount];

	// Host scene lights positions and angle
	h_Scene.lights.lpos.x = new float[Lights::TotalCount];
	h_Scene.lights.lpos.y = new float[Lights::TotalCount];
	h_Scene.lights.lpos.z = new float[Lights::TotalCount];

	h_Scene.lights.i_m.r = new float[Lights::TotalCount];
	h_Scene.lights.i_m.g = new float[Lights::TotalCount];
	h_Scene.lights.i_m.b = new float[Lights::TotalCount];

	// Host scene intensities of the specular components of the light sources
	h_Scene.lights.i_s.r = new float[Lights::TotalCount];
	h_Scene.lights.i_s.g = new float[Lights::TotalCount];
	h_Scene.lights.i_s.b = new float[Lights::TotalCount];

	// Host scene intensities of the diffuse components of the light sources
	h_Scene.lights.i_d.r = new float[Lights::TotalCount];
	h_Scene.lights.i_d.g = new float[Lights::TotalCount];
	h_Scene.lights.i_d.b = new float[Lights::TotalCount];

	h_Scene.lights.lpos.angle = new float[Lights::TotalCount];

	// Device bitmap (= output image)
	cudaFree(d_Bitmap);
	checkCudaErrors(cudaMallocManaged((void**)&d_Bitmap, BitmapSize));

	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.radius, Spheres::TotalCount * sizeof(float)));

	// Device spheres positions
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.pos.x, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.pos.y, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.pos.z, Spheres::TotalCount * sizeof(float)));

	// Device spheres colors
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.color.r, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.color.g, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.color.b, Spheres::TotalCount * sizeof(float)));

	// Device spheres reflection constants
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.ka, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.kd, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.ks, Spheres::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.spheres.alpha, Spheres::TotalCount * sizeof(float)));

	// Device lights positions
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.lpos.x, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.lpos.y, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.lpos.z, Lights::TotalCount * sizeof(float)));

	// Host scene intensities of the diffuse components of the light sources
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_d.r, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_d.g, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_d.b, Lights::TotalCount * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_m.r, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_m.g, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_m.b, Lights::TotalCount * sizeof(float)));

	// Host scene intensities of the specular components of the light sources
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_s.r, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_s.g, Lights::TotalCount * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Scene.lights.i_s.b, Lights::TotalCount * sizeof(float)));

}

// Deallocate memory of scene objects
void DeleteScene()
{
	delete[] h_Scene.spheres.pos.x;
	delete[] h_Scene.spheres.pos.y;
	delete[] h_Scene.spheres.pos.z;
	delete[] h_Scene.spheres.pos.angle;

	delete[] h_Scene.spheres.color.r;
	delete[] h_Scene.spheres.color.g;
	delete[] h_Scene.spheres.color.b;

	delete[] h_Scene.spheres.radius;
	delete[] h_Scene.spheres.ka;
	delete[] h_Scene.spheres.kd;
	delete[] h_Scene.spheres.ks;
	delete[] h_Scene.spheres.alpha;

	delete[] h_Scene.lights.lpos.x;
	delete[] h_Scene.lights.lpos.y;
	delete[] h_Scene.lights.lpos.z;

	delete[] h_Scene.lights.i_m.r;
	delete[] h_Scene.lights.i_m.g;
	delete[] h_Scene.lights.i_m.b;

	delete[] h_Scene.lights.i_d.r;
	delete[] h_Scene.lights.i_d.g;
	delete[] h_Scene.lights.i_d.b;

	delete[] h_Scene.lights.i_s.r;
	delete[] h_Scene.lights.i_s.g;
	delete[] h_Scene.lights.i_s.b;

	delete[] h_Scene.lights.lpos.angle;

	cudaFree(d_Bitmap);

	cudaFree(d_Scene.spheres.pos.x);
	cudaFree(d_Scene.spheres.pos.y);
	cudaFree(d_Scene.spheres.pos.z);

	cudaFree(d_Scene.spheres.color.r);
	cudaFree(d_Scene.spheres.color.g);
	cudaFree(d_Scene.spheres.color.b);

	cudaFree(d_Scene.spheres.radius);
	cudaFree(d_Scene.spheres.ka);
	cudaFree(d_Scene.spheres.kd);
	cudaFree(d_Scene.spheres.ks);
	cudaFree(d_Scene.spheres.alpha);

	cudaFree(d_Scene.lights.lpos.x);
	cudaFree(d_Scene.lights.lpos.y);
	cudaFree(d_Scene.lights.lpos.z);

	cudaFree(d_Scene.lights.i_m.r);
	cudaFree(d_Scene.lights.i_m.g);
	cudaFree(d_Scene.lights.i_m.b);

	cudaFree(d_Scene.lights.i_d.r);
	cudaFree(d_Scene.lights.i_d.g);
	cudaFree(d_Scene.lights.i_d.b);

	cudaFree(d_Scene.lights.i_s.r);
	cudaFree(d_Scene.lights.i_s.g);
	cudaFree(d_Scene.lights.i_s.b);
}

// Execute Raycasting algorithm and read-write calculated values
void RenderController()
{
	constexpr int h_threadsX = 16;
	constexpr int h_threadsY = 16;

	const dim3 h_blocksNum(ScreenWidth / h_threadsX + 1, ScreenHeight / h_threadsY + 1);
	const dim3 h_threadsNum(h_threadsX, h_threadsY);

	if (renderWithGpu) {
		Render << <h_blocksNum, h_threadsNum >> > (d_Bitmap, ScreenWidth, ScreenHeight, d_Scene);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// Copy bitmap from device to host
		checkCudaErrors(cudaMemcpy(h_Bitmap, d_Bitmap, BitmapSize, cudaMemcpyDeviceToHost));
	}
	else {
		RenderCPU(h_Bitmap, ScreenWidth, ScreenHeight, h_Scene);
	}
}


// Handle mouse events (glutMouseFunc() handler)
void MouseHandler(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		start_x = x;
		start_y = y;

		start_angle_x = angle_x;
		start_angle_y = angle_y;
	}
}

// Scale objects when window is resized
void ReshapeHandler(int newWidth, int newHeight)
{
	// Set new camera resolution
	SetResolution(h_Scene.camera, newWidth, newHeight);
	SetResolution(d_Scene.camera, newWidth, newHeight);

	// Update GL viewport
	glViewport(0, 0, (GLsizei)newWidth, (GLsizei)newHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, newWidth, newHeight, 0);
	glMatrixMode(GL_MODELVIEW);
}

// Handle drag events (glutMotionFunc() handler)
void MotionHandler(int x, int y)
{
	angle_x = start_angle_x + ((float)(x - start_x) / 300.0);
	angle_y = start_angle_y + ((float)(y - start_y) / 300.0);
}

// Display measured times of algorithm execution
void computeFPS()
{
	frames++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&fps_timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&fps_timer);
	}

	second_stop = clock();
	double second_passed = ((double)(second_stop - second_start)) / CLOCKS_PER_SEC + first_second;

	if (second_passed > 1)
	{
		first_second = 0;
		second_start = clock();

		sprintf(fps, "Unreal Engine 5: %.2f FPS", avgFPS);

		std::cout << fps << "\n";
	}

	glutSetWindowTitle(fps);
}

// Main "loop"
void TimerHandler(int)
{
	sdkStartTimer(&fps_timer);

	// Mark the current window as needing to be redisplayed.
	glutPostRedisplay();

	// Rotation
	float angle_diff = -(angle_x - prev_angle_x);

	Rotate(&(h_Scene.spheres.pos), Spheres::TotalCount, angle_diff);

	prev_angle_x = angle_x;
	prev_angle_y = angle_y;

	if (renderWithGpu) {
		// Copy data from host to device (update device data)
		start = clock(); // FPS
		CopyHostToDevice();
		stop = clock();
	}

	// Render (update image)
	RenderController();

	// Register next timer (simulate loop)
	constexpr int nextTimerMs = 10;
	glutTimerFunc(nextTimerMs, TimerHandler, 0);


	sdkStopTimer(&fps_timer);
	computeFPS();
}

// Setup OpenGL
void SetupOpenGL(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	glutInitWindowSize(ScreenWidth, ScreenHeight);
	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - ScreenWidth) / 2,
		(glutGet(GLUT_SCREEN_HEIGHT) - ScreenHeight) / 2);

	glutCreateWindow("Unreal Engine 5");

	glClearColor(0.0, 0.0, 0.0, 0);

	glutDisplayFunc(DisplayHandler);
	glutMouseFunc(MouseHandler);
	glutReshapeFunc(ReshapeHandler);
	glutMotionFunc(MotionHandler);
	glutTimerFunc(0, TimerHandler, 0);
	glutMainLoop();
}

int main(int argc, char** argv)
{
	if (argc == 2 && std::string{ argv[1] } == "-cpu") {
		renderWithGpu = false;
		std::cout << "Rendering in CPU!" << std::endl;
	}

	InitScene();
	checkCudaErrors(cudaGetLastError());
	GenerateScene();
	CopyHostToDevice();

	RenderController();

	SetupOpenGL(0, argv);

	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	DeleteScene();

	cudaDeviceReset();
}