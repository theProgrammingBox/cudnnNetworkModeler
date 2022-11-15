#pragma once
#include "Layer.h"

class modeler
{
public:
	modeler()
	{
		curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
		cudnnCreate(&cudnnHandle);
	}
	~modeler()
	{
		curandDestroyGenerator(randomGenerator);
		cudnnDestroy(cudnnHandle);
	}
	
	curandGenerator_t randomGenerator;
	cudnnHandle_t cudnnHandle;
	vector<Layer*> layers;
	
	float* input;
	size_t inputSize;
	size_t inputBytes;

	float* output;

	void set_input(size_t size)
	{
		inputSize = size;
		inputBytes = size * sizeof(float);
		cudaMalloc(&input, inputBytes);
	}
	void addLayer(Layer* layer)
	{
		layers.push_back(layer);
	}
	void init()
	{
		for (int i = 0; i < layers.size(); i++)
		{
			layers[i]->input = i > 0 ? layers[i - 1]->output : input;
			layers[i]->init();
		}
	}
	float* get_output()
	{
		output = new float[layers.back()->outputSize];
		cudaMemcpy(output, layers.back()->output, layers.back()->outputBytes, cudaMemcpyDeviceToHost);
		return output;
	}
};