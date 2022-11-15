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
	cudnnTensorDescriptor_t biasDescriptor;
	cudnnTensorDescriptor_t outputDescriptor;
	cudnnFilterDescriptor_t weightDescriptor;
	
	cudnnConvolutionDescriptor_t propagationDescriptor;
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;
	cudnnConvolutionBwdDataAlgo_t inputBackwardPropagationAlgorithm;
	cudnnConvolutionBwdFilterAlgo_t weightBackwardPropagationAlgorithm;

	float* input;
	float* bias;	//
	float* output;	//
	float* weight;	//
	
	float* inputGradient;
	float* biasGradient;
};