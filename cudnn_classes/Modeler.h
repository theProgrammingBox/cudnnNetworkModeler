#pragma once
#include "Layer.h"

class Modeler
{
public:
	curandGenerator_t randomGenerator;	//defined in constructor
	cudnnHandle_t cudnnHandle;			//defined in constructor
	cublasHandle_t cublasHandle;		//defined in constructor
	int maxPropagationAlgorithms;		//defined in constructor

	size_t batchSize;			//constructor defined
	size_t inputFeatures;		//constructor defined
	float learningRate;			//constructor defined
	
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
	
	Modeler(size_t batchSize, size_t inputFeatures, float learningRate)
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
		this->learningRate = learningRate;
		
		this->inputSize = batchSize * inputFeatures;
		this->inputBytes = inputSize * sizeof(float);
		cudaMalloc(&gpuInput, inputBytes);
		cudaMalloc(&gpuInputGradient, inputBytes);
		cpuInputGradient = (float*)malloc(inputBytes);
		cudnnCreateTensorDescriptor(&inputDescriptor);
		cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputFeatures, 1, 1);
	}
	
	~Modeler()
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
	
	void init()
	{
		if (layers.size() <= 0)
		{
			cout << "No layers added" << endl;
			assert(false);
		}
		
		layers[0]->init(&randomGenerator, &cudnnHandle, &cublasHandle, &maxPropagationAlgorithms,
			&batchSize, &inputFeatures, &learningRate,
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
		gpuOutput = layers.back()->gpuOutput;
		gpuOutputGradient = layers.back()->gpuOutputGradient;
		outputDescriptor = &layers.back()->outputDescriptor;
		cpuOutput = (float*)malloc(*outputBytes);
	}

	float* getCpuOutput()
	{
		cudaMemcpy(cpuOutput, gpuOutput, *outputBytes, cudaMemcpyDeviceToHost);
		return cpuOutput;
	}

	float* getCpuInputGradient()
	{
		cudaMemcpy(cpuInputGradient, gpuInputGradient, inputBytes, cudaMemcpyDeviceToHost);
		return cpuInputGradient;
	}

	void forwardPropagate(float* cpuInput)
	{
		cudaMemcpy(gpuInput, cpuInput, inputBytes, cudaMemcpyHostToDevice);
		for (size_t i = 0; i < layers.size(); i++)
		{
			layers[i]->forwardPropagate();
		}
	}

	void backPropagate(float* cpuTarget)
	{
		float minusOne = -1.0f;
		cudaMemcpy(gpuOutputGradient, cpuTarget, *outputBytes, cudaMemcpyHostToDevice);
		cublasSaxpy(cublasHandle, *outputSize, &minusOne, gpuOutput, 1, gpuOutputGradient, 1);
		/*for (size_t i = layers.size(); i--;)
		{
			layers[i]->backPropagate();
		}*/
	}

	void print()
	{
		float* cpuInput = new float[inputSize];
		cudaMemcpy(cpuInput, gpuInput, inputBytes, cudaMemcpyDeviceToHost);
		cout << "Input:" << endl;
		for (size_t i = 0; i < batchSize; i++)
		{
			for (size_t j = 0; j < inputFeatures; j++)
			{
				cout << cpuInput[i * inputFeatures + j] << " ";
			}
			cout << endl;
		}
		cout << endl;
		delete[] cpuInput;
		
		for (size_t i = 0; i < layers.size(); i++)
		{
			layers[i]->printWeights();
			layers[i]->printBias();
			layers[i]->printOutput();/**/
			//cout << "BatchSize: " << *layers[i]->batchSize << endl;
			//cout << "InputFeatures: " << *layers[i]->inputFeatures << endl;
			//cout << "OutputFeatures: " << layers[i]->outputFeatures << endl;
		}
	}

	void randomizeInput()
	{
		curandGenerateNormal(randomGenerator, gpuInput, inputSize + (inputSize & 1), 0.0f, 1.0f);
	}
};