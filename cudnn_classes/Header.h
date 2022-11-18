#pragma once
#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::vector;
using std::max;