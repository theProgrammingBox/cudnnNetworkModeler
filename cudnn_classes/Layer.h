#pragma once
#include "Header.h"

class Layer
{
public:
	curandGenerator_t* randomGenerator;	//defined in init by modeler
	cudnnHandle_t* cudnnHandle;			//defined in init by modeler
	cublasHandle_t* cublasHandle;		//defined in init by modeler
	int* maxPropagationAlgorithms;		//defined in init by modeler
	
	size_t* batchSize;			//defined in init by modeler
	size_t* inputFeatures;		//defined in init by modeler
	
	size_t* inputSize;			//defined in init by modeler
	size_t* inputBytes;			//defined in init by modeler
	float* gpuInput;			//defined in init by modeler
	float* gpuInputGradient;	//defined in init by modeler
	cudnnTensorDescriptor_t* inputDescriptor;	//defined in init by modeler

	size_t outputFeatures;		//constructor defined
	
	size_t outputSize;			//defined in init
	size_t outputBytes;			//defined in init
	float* gpuOutput;			//defined in init
	float* gpuOutputGradient;	//defined in init
	cudnnTensorDescriptor_t outputDescriptor;	//defined in init
	
	size_t weightSize;			//defined in init
	size_t weightBytes;			//defined in init
	float* gpuWeight;			//defined in init
	float* gpuWeightGradient;	//defined in init
	cudnnFilterDescriptor_t weightDescriptor;	//defined in init
	
	size_t biasSize;			//defined in init
	size_t biasBytes;			//defined in init
	float* gpuBias;				//defined in init
	float* gpuBiasGradient;		//defined in init
	cudnnTensorDescriptor_t biasDescriptor;		//defined in init
	
	size_t* workspaceBytes;		//defined in init by modeler
	void* workspace;			//defined in init by modeler

	cudnnConvolutionDescriptor_t propagationDescriptor;					//defined in init
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;				//defined in init
	cudnnConvolutionBwdDataAlgo_t inputBackwardPropagationAlgorithm;	//defined in init
	cudnnConvolutionBwdFilterAlgo_t weightBackwardPropagationAlgorithm;	//defined in init
	
	Layer(size_t outputFeatures)
	{
		if (outputFeatures <= 0)
		{
			cout << "Error: Layer output features must be greater than 0." << endl;
			assert(false);
		}
		
		this->outputFeatures = outputFeatures;
	}
	
	~Layer()
	{
		cudaFree(gpuOutput);
		cudaFree(gpuOutputGradient);
		cudnnDestroyTensorDescriptor(outputDescriptor);
		
		cudaFree(gpuWeight);
		cudaFree(gpuWeightGradient);
		cudnnDestroyFilterDescriptor(weightDescriptor);
		
		cudaFree(gpuBias);
		cudaFree(gpuBiasGradient);
		cudnnDestroyTensorDescriptor(biasDescriptor);
		
		cudnnDestroyConvolutionDescriptor(propagationDescriptor);
	}

	void init(curandGenerator_t* randomGenerator, cudnnHandle_t* cudnnHandle, cublasHandle_t* cublasHandle, int* maxPropagationAlgorithms,
		size_t* batchSize, size_t* inputFeatures,
		size_t* inputSize, size_t* inputBytes,
		float* gpuInput, float* gpuInputGradient, cudnnTensorDescriptor_t* inputDescriptor,
		size_t* workspaceBytes, void* gpuWorkspace)
	{
		outputSize = *batchSize * outputFeatures;
		outputBytes = outputSize * sizeof(float);
		cudaMalloc(&gpuOutput, outputBytes);
		cudaMalloc(&gpuOutputGradient, outputBytes);
		cudnnCreateTensorDescriptor(&outputDescriptor);
		cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, *batchSize, outputFeatures, 1, 1);
		
		weightSize = outputFeatures * *inputFeatures;
		weightBytes = weightSize * sizeof(float);
		cudaMalloc(&gpuWeight, weightBytes);
		cudaMalloc(&gpuWeightGradient, weightBytes);
		cudnnCreateFilterDescriptor(&weightDescriptor);
		cudnnSetFilter4dDescriptor(weightDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputFeatures, *inputFeatures, 1, 1);

		biasSize = outputFeatures;
		biasBytes = biasSize * sizeof(float);
		cudaMalloc(&gpuBias, biasBytes);
		cudaMalloc(&gpuBiasGradient, biasBytes);
		cudnnCreateTensorDescriptor(&biasDescriptor);
		cudnnSetTensor4dDescriptor(biasDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outputFeatures, 1, 1);
		
		cudnnCreateConvolutionDescriptor(&propagationDescriptor);
		cudnnSetConvolution2dDescriptor(propagationDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		
		cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
		cudnnFindConvolutionForwardAlgorithm(*cudnnHandle, *inputDescriptor, weightDescriptor, propagationDescriptor, outputDescriptor, *maxPropagationAlgorithms, maxPropagationAlgorithms, forwardPropagationAlgorithms);
		forwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
		delete[] forwardPropagationAlgorithms;

		cudnnConvolutionBwdDataAlgoPerf_t* inputBackwardPropagationAlgorithms = new cudnnConvolutionBwdDataAlgoPerf_t[maxPropagationAlgorithms];
		cudnnFindConvolutionBackwardDataAlgorithm(*cudnnHandle, weightDescriptor, *inputDescriptor, propagationDescriptor, outputDescriptor, *maxPropagationAlgorithms, maxPropagationAlgorithms, inputBackwardPropagationAlgorithms);
		inputBackwardPropagationAlgorithm = inputBackwardPropagationAlgorithms[0].algo;
		delete[] inputBackwardPropagationAlgorithms;

		cudnnConvolutionBwdFilterAlgoPerf_t* weightBackwardPropagationAlgorithms = new cudnnConvolutionBwdFilterAlgoPerf_t[maxPropagationAlgorithms];
		cudnnFindConvolutionBackwardFilterAlgorithm(*cudnnHandle, *inputDescriptor, outputDescriptor, propagationDescriptor, weightDescriptor, *maxPropagationAlgorithms, maxPropagationAlgorithms, weightBackwardPropagationAlgorithms);
		weightBackwardPropagationAlgorithm = weightBackwardPropagationAlgorithms[0].algo;
		delete[] weightBackwardPropagationAlgorithms;

		size_t tempWorkspaceBytes;
		cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle, *inputDescriptor, weightDescriptor, propagationDescriptor, outputDescriptor, forwardPropagationAlgorithm, tempWorkspaceBytes);
		
	}
};