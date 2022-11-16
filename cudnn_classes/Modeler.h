#pragma once
#include "Layer.h"

class modeler
{
public:
	curandGenerator_t randomGenerator;	//defined in constructor
	cudnnHandle_t cudnnHandle;			//defined in constructor
	cublasHandle_t cublasHandle;		//defined in constructor
	int maxPropagationAlgorithms;		//defined in constructor

	size_t batchSize;			//constructor defined
	size_t inputFeatures;		//constructor defined
	
	size_t inputSize;			//defined in constructor
	size_t inputBytes;			//defined in constructor
	float* gpuInput;			//defined in constructor, user pass in cpuInput in forwardPropagation
	float* gpuInputGradient;	//defined in constructor
	float* cpuInputGradient;	//defined in constructor, return to user in init
	cudnnTensorDescriptor_t inputDescriptor;	//defined in constructor

	size_t* outputSize;			//set in init by layers
	size_t* outputBytes;		//set in init by layers
	float* cpuOutput;			//defined in init, return to user in init
	float* gpuOutput;			//defined in init by layers
	float* gpuOutputGradient;	//defined in init by layers, user pass in cpuTarget, subtracted in backpropagation
	cudnnTensorDescriptor_t* outputDescriptor;	//defined in init by layers
	
	size_t workspaceBytes = 0;	//defined in init, uses layers
	void* gpuWorkspace;			//defined in init, uses layers
	
	vector<Layer*> layers;		//user pass in layers
	
	modeler(size_t batchSize, size_t inputFeatures)
	{
		if (batchSize <= 0 || inputFeatures <= 0)
		{
			cout << "Invalid batch size or input features" << endl;
			assert(false);
		}
		
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
		cpuInputGradient = (float*)malloc(inputBytes);
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
		free(cpuInputGradient);
		cudnnDestroyTensorDescriptor(inputDescriptor);

		free(cpuOutput);
		
		cudaFree(gpuWorkspace);
		
		for (size_t i = layers.size(); i--;) delete layers[i];
	}
	
	void addLayer(Layer* layer)
	{
		layers.push_back(layer);
	}
	
	pair<float*, float*> init()
	{
		if (layers.size() <= 0)
		{
			cout << "No layers added" << endl;
			assert(false);
		}
		
		layers[0]->init(&randomGenerator, &cudnnHandle, &cublasHandle, &maxPropagationAlgorithms,
			&batchSize, &inputFeatures,
			&inputSize, &inputBytes,
			gpuInput, gpuInputGradient, &inputDescriptor,
			&workspaceBytes, gpuWorkspace);
		
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->init(&randomGenerator, &cudnnHandle, &cublasHandle, &maxPropagationAlgorithms,
				&batchSize, &layers[i - 1]->outputFeatures,
				&layers[i - 1]->outputSize, &layers[i - 1]->outputBytes,
				layers[i - 1]->gpuOutput, layers[i - 1]->gpuOutputGradient, &layers[i - 1]->outputDescriptor,
				&workspaceBytes, gpuWorkspace);
		}

		cudaMalloc(&gpuWorkspace, workspaceBytes);
		
		outputSize = &layers.back()->outputSize;
		outputBytes = &layers.back()->outputBytes;
		gpuOutput = layers.back()->output;
		gpuOutputGradient = layers.back()->outputGradient;
		outputDescriptor = &layers.back()->outputDescriptor;
		cpuOutput = (float*)malloc(*outputBytes);

		return make_pair(cpuInputGradient, cpuOutput);
	}
};