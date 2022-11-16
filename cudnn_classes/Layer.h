#pragma once
#include "Header.h"

class Layer
{
public:
	size_t inputSize;			// defined externally
	size_t inputBytes;			// defined externally
	float* gpuInput;			// defined externally
	float* gpuInputGradient;	// defined externally
	
	size_t weightSize;			// defined locally
	size_t weightBytes;			// defined locally
	float* gpuWeight;			// defined locally
	float* gpuWeightGradient;	// defined locally
	
	size_t biasSize;			// defined locally
	size_t biasBytes;			// defined locally
	float* gpuBias;				// defined locally
	float* gpuBiasGradient;		// defined locally
	
	size_t outputSize;			// defined locally
	size_t outputBytes;			// defined locally
	float* gpuOutput;			// defined locally
	float* gpuOutputGradient;	// defined locally

	cudnnTensorDescriptor_t inputDescriptor;
	cudnnFilterDescriptor_t weightDescriptor;
	cudnnTensorDescriptor_t biasDescriptor;
	cudnnTensorDescriptor_t outputDescriptor;
	
	cudnnConvolutionDescriptor_t propagationDescriptor;
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;
	cudnnConvolutionBwdDataAlgo_t inputBackwardPropagationAlgorithm;
	cudnnConvolutionBwdFilterAlgo_t weightBackwardPropagationAlgorithm;
	
	Layer();
	~Layer();
	
	virtual void forward() = 0;
	virtual void backward() = 0;

	float* input;
	float* weight;	//
	float* bias;	//
	float* output;	//
	
	float* inputGradient;
	float* weightGradient;	//
	float* biasGradient;	//
	float* outputGradient;	//
};