#pragma once
#include "Layer.h"

class modeler
{
public:
	curandGenerator_t randomGenerator;
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;

	int maxPropagationAlgorithms;
	
	size_t batchSize;			// defined locally, constructor defined
	size_t inputFeatures;		// defined locally, constructor defined
	
	size_t inputSize;			// defined locally
	size_t inputBytes;			// defined locally
	float* gpuInput;			// defined locally, pass to the first layer, user gives the cpuInput*
	float* gpuInputGradient;	// defined locally, pass to the first layer
	float* cpuInputGradient;	// defined locally, return to user
	cudnnTensorDescriptor_t inputDescriptor;	// defined locally, pass to the first layer

	size_t* outputSize;			// defined externally, after all layers are created
	size_t* outputBytes;		// defined externally, after all layers are created
	float* gpuOutputGradient;	// defined externally, after all layers are created
	float* cpuOutput;			// defined locally, after all layers are created, return to user
	float* gpuTarget;			// defined locally, after all layers are created, user gives the cpuTarget*
	cudnnTensorDescriptor_t outputDescriptor;	// defined externally, after all layers are created
	
	modeler(size_t batchSize, size_t inputFeatures)
	{
		curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
		cudnnCreate(&cudnnHandle);
		cublasCreate(&cublasHandle);
		cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
		
		this->batchSize = batchSize;
		this->inputFeatures = inputFeatures;
		
		this->inputSize = batchSize * inputFeatures;
		this->inputBytes = inputSize * sizeof(float);
		cudaMalloc(&gpuInput, inputBytes);
		cudaMalloc(&gpuInputGradient, inputBytes);
		cpuInputGradient = new float[inputSize];
		cudnnCreateTensorDescriptor(&inputDescriptor);
		cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputFeatures, 1, 1);
	}
	
	~modeler()
	{
		curandDestroyGenerator(randomGenerator);
		cudnnDestroy(cudnnHandle);
		cublasDestroy(cublasHandle);
		
		cudaFree(gpuInput);
		cudaFree(gpuInputGradient);
		delete[] cpuInputGradient;
		cudnnDestroyTensorDescriptor(inputDescriptor);
	}
	
	void addLayer(Layer* layer)
	{
		layers.push_back(layer);
	}
	
	float* init()
	{
		layers[0]->gpuInput = gpuInput;
		layers[0]->inputDescriptor = inputDescriptor;
		layers[0]->inputSize = inputSize;
		layers[0]->inputBytes = inputBytes;
		for (int i = 1; i < layers.size(); i++)
		{
			layers[i]->gpuInput = layers[i - 1]->gpuOutput;
			layers[i]->inputDescriptor = layers[i - 1]->outputDescriptor;
			layers[i]->inputSize = layers[i - 1]->outputSize;
			layers[i]->inputBytes = layers[i - 1]->outputBytes;
		}
		cpuOutput = new float[layers.back()->outputSize];
		cudaMemcpy(cpuOutput, layers.back()->gpuOutput, layers.back()->outputBytes, cudaMemcpyDeviceToHost);
		return cpuOutput;
	}
};