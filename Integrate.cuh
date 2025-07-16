#pragma once
#include "CalculateContactForceTorque.cuh"
#include "ExternalForceTorque.cuh"

__global__ void clearForceTorque(double3* forces, double3* torques, int num);

__global__ void integrateBeforeContactCalculation(DynamicState state, double3 gravity, double timeStep, int num);

__global__ void integrateAfterContactCalculation(DynamicState state, double3 gravity, double timeStep, int num);

__global__ void correctSPHMotion(Sphere sph, SPH SPHP, double timeStep);

__global__ void calClumpForceTorque(Clump clump, Sphere sph);

__global__ void updatePebbleMotionBeforeContact(Clump clump, Sphere sph, double timeStep);

void integrateBeforeContact(DeviceData& d, double3 gravity, double timeStep, int maxThreadsPerBlock);

void integrateAfterContact(DeviceData& d, double3 gravity, double timeStep, int maxThreadsPerBlock);
