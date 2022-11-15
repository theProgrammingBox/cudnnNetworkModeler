#pragma once
#include "Header.h"

class Layer
{
public:
	Layer();
	~Layer();
	
	virtual void forward() = 0;
	virtual void backward() = 0;
	
	cudnnTensorDescriptor_t inputDescriptor;
	cudnnFilterDescriptor_t weightDescriptor;
	cudnnTensorDescriptor_t biasDescriptor;
	cudnnTensorDescriptor_t outputDescriptor;
	
	cudnnConvolutionDescriptor_t propagationDescriptor;
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;
	cudnnConvolutionBwdDataAlgo_t inputBackwardPropagationAlgorithm;
	cudnnConvolutionBwdFilterAlgo_t weightBackwardPropagationAlgorithm;

	float* input;
	float* weight;	//
	float* bias;	//
	float* output;	//
	
	float* inputGradient;	//
	float* weightGradient;	//
	float* biasGradient;	//
	float* outputGradient;

	size_t inputSize;
	size_t weightSize;
	size_t biasSize;
	size_t outputSize;
	
	size_t inputBytes;
	size_t weightBytes;
	size_t biasBytes;
	size_t outputBytes;
};