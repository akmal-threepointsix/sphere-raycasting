main: kernel.cu
	nvcc kernel.cu -o main -I"include" -l"freeglut" -L"lib"
clean:
	rm main
.PHONY:
	clean

