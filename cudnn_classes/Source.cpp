#include "Modeler.h"
#include "Layer.h"

int main()
{
	srand(time(NULL));

	const size_t batchSize = 5;
	const size_t inputFeatures = 2;
	const size_t outputFeatures = 1;
	const float learningRate = 0.002f;

	float* cpuInput;
	float* cpuInputGradient;
	float* cpuOutput;
	float* cpuTarget;
	cpuInput = new float[batchSize * inputFeatures];
	cpuTarget = new float[batchSize * outputFeatures];

	Modeler modeler(batchSize, inputFeatures, learningRate);
	modeler.addLayer(new Layer(outputFeatures));
	modeler.init();

	size_t iteration = 1;
	while (iteration--)
	{
		//modeler.randomizeInput();
		for (size_t i = batchSize; i--;)
		{
			for (size_t j = inputFeatures; j--;)
			{
				cpuInput[i * inputFeatures + j] = (float)rand() / RAND_MAX;
			}
		}

		for (size_t i = batchSize; i--;)
		{
			for (size_t j = outputFeatures; j--;)
			{
				// sum of input features
				cpuTarget[i * outputFeatures + j] = 0;
				for (size_t k = inputFeatures; k--;)
				{
					cpuTarget[i * outputFeatures + j] += cpuInput[i * inputFeatures + k];
				}
			}
		}

		modeler.forwardPropagate(cpuInput);
		cpuOutput = modeler.getCpuOutput();

		modeler.backPropagate(cpuTarget);
		//cpuInputGradient = modeler.getCpuInputGradient();
	}

	cout << "Target:" << endl;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << cpuTarget[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	modeler.print();

	//print output
	float* cpuWeight;
	cpuWeight = (float*)malloc(sizeof(float) * inputFeatures * outputFeatures);
	cudaMemcpy(cpuWeight, modeler.layers[0]->gpuWeight, sizeof(float) * inputFeatures * outputFeatures, cudaMemcpyDeviceToHost);
	
	float* cpuBias;
	cpuBias = (float*)malloc(sizeof(float) * outputFeatures);
	cudaMemcpy(cpuBias, modeler.layers[0]->gpuBias, sizeof(float) * outputFeatures, cudaMemcpyDeviceToHost);
	
	cout << "Output manual:" << endl;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			float sum = cpuBias[j];
			for (size_t k = 0; k < inputFeatures; k++)
			{
				sum += cpuInput[i * inputFeatures + k] * cpuWeight[j * inputFeatures + k];
			}
			cout << sum << " ";
		}
		cout << endl;
	}
	cout << endl;

	return 0;
}