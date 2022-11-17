#include "Modeler.h"
#include "Layer.h"

int main()
{
	const size_t batchSize = 7;
	const size_t inputFeatures = 5;
	const size_t outputFeatures = 3;

	float* cpuInput;
	float* cpuInputGradient;
	float* cpuOutput;
	float* cpuTarget;

	cout << "hi\n";

	Modeler modeler(batchSize, inputFeatures);
	modeler.addLayer(new Layer(outputFeatures));
	modeler.init();
	
	cout << "hi\n";

	// init cpu memory
	cpuInput = new float[batchSize * inputFeatures];
	for (size_t i = batchSize; i--;)
	{
		for (size_t j = inputFeatures; j--;)
		{
			cpuInput[i * inputFeatures + j] = i * inputFeatures + j;
		}
	}

	cpuTarget = new float[batchSize * outputFeatures];
	for (size_t i = batchSize; i--;)
	{
		for (size_t j = outputFeatures; j--;)
		{
			cpuTarget[i * outputFeatures + j] = i * outputFeatures + j;
		}
	}
	cout << "hi1\n";

	modeler.forwardPropagate(cpuInput);
	cout << "hi2\n";
	cpuOutput = modeler.getCpuOutput();
	cout << "hi3\n";
	modeler.backPropagate(cpuTarget);
	cout << "hi4\n";
	//cpuInputGradient = modeler.getCpuInputGradient();

	//print output
	for (size_t i = batchSize; i--;)
	{
		for (size_t j = outputFeatures; j--;)
		{
			cout << cpuOutput[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	return 0;

	//const size_t batchSize = 7;
	//const size_t inputFeatures = 5;
	//const size_t outputFeatures = 3;

	//float* cpuInput = new float[batchSize * inputFeatures];
	//float* cpuTarget = new float[batchSize * outputFeatures];
	//
	//// MATRIX MULTIPLICATION
	//// INITIALIZE RANDOM NUMBER GENERATOR AND HANDLE
	//curandGenerator_t randomGenerator;
	//curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	//curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	//
	//cudnnHandle_t cudnnHandle;
	//cudnnCreate(&cudnnHandle);
	//cublasHandle_t cublasHandle;
	//cublasCreate(&cublasHandle);

	//int maxPropagationAlgorithms;
	//cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
	//
	//// DEFINE INITIAL PARAMS
	//cudnnTensorDescriptor_t inputDescriptor;
	//cudnnCreateTensorDescriptor(&inputDescriptor);
	//cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputFeatures, 1, 1);
	//
	//// DEFINE LAYER PARAMS
	//cudnnFilterDescriptor_t weightDescriptor;
	//cudnnCreateFilterDescriptor(&weightDescriptor);
	//cudnnSetFilter4dDescriptor(weightDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputFeatures, inputFeatures, 1, 1);
	//
	//cudnnTensorDescriptor_t biasDescriptor;
	//cudnnCreateTensorDescriptor(&biasDescriptor);
	//cudnnSetTensor4dDescriptor(biasDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outputFeatures, 1, 1);
	//
	//cudnnTensorDescriptor_t outputDescriptor;
	//cudnnCreateTensorDescriptor(&outputDescriptor);
	//cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outputFeatures, 1, 1);
	//
	//cudnnConvolutionDescriptor_t propagationDescriptor;
	//cudnnCreateConvolutionDescriptor(&propagationDescriptor);
	//cudnnSetConvolution2dDescriptor(propagationDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
	//
	//// DEFINE ALGORITHMS
	//cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
	//cudnnFindConvolutionForwardAlgorithm(cudnnHandle, inputDescriptor, weightDescriptor, propagationDescriptor, outputDescriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, forwardPropagationAlgorithms);
	//cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
	//delete[] forwardPropagationAlgorithms;
	//
	//cudnnConvolutionBwdFilterAlgoPerf_t* weightBackwardPropagationAlgorithms = new cudnnConvolutionBwdFilterAlgoPerf_t[maxPropagationAlgorithms];
	//cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle, inputDescriptor, outputDescriptor, propagationDescriptor, weightDescriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, weightBackwardPropagationAlgorithms);
	//cudnnConvolutionBwdFilterAlgo_t weightBackwardPropagationAlgorithm = weightBackwardPropagationAlgorithms[0].algo;
	//delete[] weightBackwardPropagationAlgorithms;

	//cudnnConvolutionBwdDataAlgoPerf_t* inputBackwardPropagationAlgorithms = new cudnnConvolutionBwdDataAlgoPerf_t[maxPropagationAlgorithms];
	//cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle, weightDescriptor, outputDescriptor, propagationDescriptor, inputDescriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, inputBackwardPropagationAlgorithms);
	//cudnnConvolutionBwdDataAlgo_t inputBackwardPropagationAlgorithm = inputBackwardPropagationAlgorithms[0].algo;
	//delete[] inputBackwardPropagationAlgorithms;

	//// DEFINE WORKSPACE
	//size_t tempBytes;
	//size_t workspaceBytes = 0;
	//cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDescriptor, weightDescriptor, propagationDescriptor, outputDescriptor, forwardPropagationAlgorithm, &tempBytes);
	//workspaceBytes = max(workspaceBytes, tempBytes);
	//cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, weightDescriptor, outputDescriptor, propagationDescriptor, inputDescriptor, inputBackwardPropagationAlgorithm, &tempBytes);
	//workspaceBytes = max(workspaceBytes, tempBytes);
	//cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDescriptor, outputDescriptor, propagationDescriptor, weightDescriptor, weightBackwardPropagationAlgorithm, &tempBytes);
	//workspaceBytes = max(workspaceBytes, tempBytes);
	//void* workspace;
	//cudaMalloc(&workspace, workspaceBytes);

	//// DEFINE DATA
	//size_t inputSize = batchSize * inputFeatures;
	//size_t weightSize = outputFeatures * inputFeatures;
	//size_t biasSize = outputFeatures;
	//size_t outputSize = batchSize * outputFeatures;

	//size_t inputBytes = inputSize * sizeof(float);
	//size_t weightBytes = weightSize * sizeof(float);
	//size_t biasBytes = biasSize * sizeof(float);
	//size_t outputBytes = outputSize * sizeof(float);
	//
	//float* gpuInput;
	//float* gpuWeight;
	//float* gpuBias;
	//float* gpuOutput;
	//float* gpuTarget;
	//
	//cudaMalloc(&gpuInput, inputBytes);
	//cudaMalloc(&gpuWeight, weightBytes);
	//cudaMalloc(&gpuBias, biasBytes);
	//cudaMalloc(&gpuOutput, outputBytes);
	//cudaMalloc(&gpuTarget, outputBytes);

	//float* gpuInputGradient;
	//float* gpuWeightGradient;
	//float* gpuBiasGradient;
	//float* gpuOutputGradient;
	//
	//cudaMalloc(&gpuInputGradient, inputBytes);
	//cudaMalloc(&gpuWeightGradient, weightBytes);
	//cudaMalloc(&gpuBiasGradient, biasBytes);
	//cudaMalloc(&gpuOutputGradient, outputBytes);
	//
	//// INITIALIZE DATA
	//curandGenerateNormal(randomGenerator, gpuWeight, weightSize + (weightSize & 1), 0, 1);
	//curandGenerateNormal(randomGenerator, gpuBias, biasSize + (biasSize & 1), 0, 1);

	//// READ DATA
	//float* cpuWeight = new float[weightSize];
	//float* cpuBias = new float[biasSize];
	//cudaMemcpy(cpuWeight, gpuWeight, weightBytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(cpuBias, gpuBias, biasBytes, cudaMemcpyDeviceToHost);
	//
	//// PRINT DATA
	//cout << "Weight:" << endl;
	//for (int i = 0; i < outputFeatures; i++)
	//{
	//	for (int j = 0; j < inputFeatures; j++)
	//	{
	//		cout << cpuWeight[i * inputFeatures + j] << " ";
	//	}
	//	cout << endl;
	//}
	//cout << endl;
	//cout << "Bias:" << endl;
	//for (int i = 0; i < outputFeatures; i++)
	//{
	//	cout << cpuBias[i] << " ";
	//}
	//cout << endl << endl;

	//// TRAINING
	//float alpha = 1.0f;
	//float beta = 0.0f;
	//float learningRate = 0.01f;
	//size_t iterations = 100;
	//while (iterations--)
	//{
	//	// READ DATA
	//	//cudaMemcpy(gpuInput, cpuInput, inputBytes, cudaMemcpyHostToDevice);
	//	curandGenerateNormal(randomGenerator, gpuInput, inputSize + (inputSize & 1), 0, 1);
	//	cudaMemcpy(cpuInput, gpuInput, inputBytes, cudaMemcpyDeviceToHost);
	//	/*cout << "Input:" << endl;
	//	for (int i = 0; i < batchSize; i++)
	//	{
	//		for (int j = 0; j < inputFeatures; j++)
	//		{
	//			cout << cpuInput[i * inputFeatures + j] << " ";
	//		}
	//		cout << endl;
	//	}
	//	cout << endl;*/

	//	// FORWARD PROPAGATION
	//	cudnnConvolutionForward(cudnnHandle, &alpha, inputDescriptor, gpuInput, weightDescriptor, gpuWeight, propagationDescriptor, forwardPropagationAlgorithm, workspace, workspaceBytes, &beta, outputDescriptor, gpuOutput);
	//	cudnnAddTensor(cudnnHandle, &alpha, biasDescriptor, gpuBias, &alpha, outputDescriptor, gpuOutput);

	//	// READ AND PRINT DATA
	//	/*float* cpuOutput = new float[outputSize];
	//	cudaMemcpy(cpuOutput, gpuOutput, outputBytes, cudaMemcpyDeviceToHost);
	//	cout << "Output:" << endl;
	//	for (int i = 0; i < batchSize; i++)
	//	{
	//		for (int j = 0; j < outputFeatures; j++)
	//		{
	//			cout << cpuOutput[i * outputFeatures + j] << " ";
	//		}
	//		cout << endl;
	//	}
	//	cout << endl;*/
	//	
	//	// READ DATA
	//	//cudaMemcpy(cpuTarget, gpuTarget, outputBytes, cudaMemcpyDeviceToHost);
	//	//cout << "Target:" << endl;
	//	for (int i = 0; i < batchSize; i++)
	//	{
	//		for (int j = 0; j < outputFeatures; j++)
	//		{
	//			cpuTarget[i * outputFeatures + j] = j;
	//			for (int k = 0; k < inputFeatures; k++)
	//			{
	//				cpuTarget[i * outputFeatures + j] += cpuInput[i * inputFeatures + k] * k - j + cpuInput[i * inputFeatures];
	//			}
	//			//cout << cpuTarget[i * outputFeatures + j] << " ";
	//		}
	//		//cout << endl;
	//	}
	//	//cout << endl;

	//	// ERROR, TARGET - OUTPUT
	//	alpha = -1.0f;
	//	cudaMemcpy(gpuOutputGradient, cpuTarget, outputBytes, cudaMemcpyHostToDevice);
	//	cublasSaxpy(cublasHandle, outputSize, &alpha, gpuOutput, 1, gpuOutputGradient, 1);
	//	
	//	// READ AND PRINT DATA
	//	float* cpuOutputGradient = new float[outputSize];
	//	cudaMemcpy(cpuOutputGradient, gpuOutputGradient, outputBytes, cudaMemcpyDeviceToHost);
	//	cout << "Output Gradient:" << endl;
	//	for (int i = 0; i < batchSize; i++)
	//	{
	//		for (int j = 0; j < outputFeatures; j++)
	//		{
	//			cout << cpuOutputGradient[i * outputFeatures + j] << " ";
	//		}
	//		cout << endl;
	//	}
	//	cout << endl;

	//	/*cudaMemcpy(cpuTarget, gpuTarget, outputBytes, cudaMemcpyDeviceToHost);
	//	cout << "Target:" << endl;
	//	for (int i = 0; i < batchSize; i++)
	//	{
	//		for (int j = 0; j < outputFeatures; j++)
	//		{
	//			cout << cpuTarget[i * outputFeatures + j] << " ";
	//		}
	//		cout << endl;
	//	}
	//	cout << endl;*/
	//	
	//	// BACKWARD PROPAGATION
	//	alpha = 1.0f;
	//	cudnnConvolutionBackwardBias(cudnnHandle, &alpha, outputDescriptor, gpuOutputGradient, &beta, biasDescriptor, gpuBiasGradient);
	//	cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, inputDescriptor, gpuInput, outputDescriptor, gpuOutputGradient, propagationDescriptor, weightBackwardPropagationAlgorithm, workspace, workspaceBytes, &beta, weightDescriptor, gpuWeightGradient);
	//	cudnnConvolutionBackwardData(cudnnHandle, &alpha, weightDescriptor, gpuWeight, outputDescriptor, gpuOutputGradient, propagationDescriptor, inputBackwardPropagationAlgorithm, workspace, workspaceBytes, &beta, inputDescriptor, gpuInputGradient);

	//	// UPDATE PARAMETERS
	//	cublasSaxpy(cublasHandle, weightSize, &learningRate, gpuWeightGradient, 1, gpuWeight, 1);
	//	cublasSaxpy(cublasHandle, biasSize, &learningRate, gpuBiasGradient, 1, gpuBias, 1);
	//}

	//// READ DATA
	//cudaMemcpy(cpuWeight, gpuWeight, weightBytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(cpuBias, gpuBias, biasBytes, cudaMemcpyDeviceToHost);

	//// PRINT DATA
	//cout << "Weight:" << endl;
	//for (int i = 0; i < outputFeatures; i++)
	//{
	//	for (int j = 0; j < inputFeatures; j++)
	//	{
	//		cout << cpuWeight[i * inputFeatures + j] << " ";
	//	}
	//	cout << endl;
	//}
	//cout << endl;
	//cout << "Bias:" << endl;
	//for (int i = 0; i < outputFeatures; i++)
	//{
	//	cout << cpuBias[i] << " ";
	//}
	//cout << endl << endl;
	//
	///*
	//* size_t batchSize
	//* size_t inputFeatures
	//* size_t outputFeatures
	//* 
	//* cudnnTensorDescriptor_t inputDescriptor
	//* cudnnFilterDescriptor_t weightDescriptor
	//* cudnnTensorDescriptor_t biasDescriptor
	//* cudnnTensorDescriptor_t outputDescriptor
	//* 
	//* cudnnConvolutionDescriptor_t propagationDescriptor
	//* cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm
	//* cudnnConvolutionBwdDataAlgo_t inputBackwardPropagationAlgorithm
	//* cudnnConvolutionBwdFilterAlgo_t weightBackwardPropagationAlgorithm
	//* 
	//* size_t workspaceBytes
	//* void* workspace
	//*/
	//
	//return 0;
}