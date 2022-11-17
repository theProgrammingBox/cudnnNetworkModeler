#include "Modeler.h"
#include "Layer.h"

int main()
{
	const size_t batchSize = 7;
	const size_t inputFeatures = 5;
	const size_t outputFeatures = 3;
	const float learningRate = 0.01f;

	float* cpuInput;
	float* cpuInputGradient;
	float* cpuOutput;
	float* cpuTarget;

	Modeler modeler(batchSize, inputFeatures, learningRate);
	modeler.addLayer(new Layer(outputFeatures));
	modeler.init();

	//modeler.randomizeInput();
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

	modeler.forwardPropagate(cpuInput);
	cpuOutput = modeler.getCpuOutput();
	
	modeler.backPropagate(cpuTarget);
	//cpuInputGradient = modeler.getCpuInputGradient();
	
	modeler.print();

	return 0;
}