#pragma once
#include "Layer.h"

class modeler
{
public:
	modeler()
	{
		curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
		cudnnCreate(&cudnnHandle);
	}
	~modeler()
	{
		curandDestroyGenerator(randomGenerator);
		cudnnDestroy(cudnnHandle);
	}
	
	curandGenerator_t randomGenerator;
	cudnnHandle_t cudnnHandle;
	vector<Layer*> layers;
};