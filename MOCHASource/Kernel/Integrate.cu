#include "Integrate.cuh"

__global__ void integrateBeforeContactCalculation(DynamicState state,
	double3 gravity,
	double timeStep,
	int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;
	double invMass = state.inverseMass[idx];
	state.velocities[idx] += (state.forces[idx] * invMass + gravity * (invMass != 0)) * 0.5 * timeStep;
	state.positions[idx] += state.velocities[idx] * timeStep;
	state.angularVelocities[idx] += (rotateInverseInertiaTensor(state.orientations[idx], state.inverseInertia[idx]) * state.torques[idx]) * 0.5 * timeStep * (invMass != 0);
	state.orientations[idx] = quaternionRotate(state.orientations[idx], state.angularVelocities[idx], timeStep);
}

__global__ void integrateAfterContactCalculation(DynamicState state,
	double3 gravity,
	double timeStep,
	int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;
	double invMass = state.inverseMass[idx];
	state.velocities[idx] += (state.forces[idx] * invMass + gravity * (invMass != 0)) * 0.5 * timeStep;
	state.angularVelocities[idx] += (rotateInverseInertiaTensor(state.orientations[idx], state.inverseInertia[idx]) * state.torques[idx]) * 0.5 * timeStep * (invMass != 0);
}

__global__ void correctWCSPHMotion(Sphere sph,
	SPH SPHP,
	double timeStep)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= sph.num) return;
	int SPHIndex = sph.SPHIndex[idx];
	if (SPHIndex < 0) return;
	sph.state.positions[idx] += SPHP.XSPHVariant[SPHIndex] * timeStep;
}

__global__ void updatePebblesMotionBeforeContact(Clump clump,
	Sphere sph,
	double timeStep)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= clump.num) return;

	for (int i = clump.pebbleStart[idx]; i < clump.pebbleEnd[idx];++i)
	{
		if (sph.clumpIndex[i] != idx) continue;
		sph.state.positions[i] -= sph.state.velocities[i] * timeStep;

		sph.state.velocities[i] = clump.state.velocities[idx] + cross(clump.state.angularVelocities[idx], sph.state.positions[i] - clump.state.positions[idx]);
		sph.state.angularVelocities[i] = clump.state.angularVelocities[idx];
		sph.state.positions[i] += sph.state.velocities[i] * timeStep;
		sph.state.orientations[i] = clump.state.orientations[idx];
	}
}

void integrateBeforeContact(DeviceData& d, double3 gravity, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	integrateBeforeContactCalculation << <grid, block >> > (d.spheres.state, 
		gravity, 
		timeStep, 
		d.spheres.num);
	//cudaDeviceSynchronize();

	//if (d.SPHParticles.num > 0.)
	//{
	//	correctWCSPHMotion << <grid, block >> > (d.spheres,
	//		d.SPHParticles,
	//		timeStep);
	//	//cudaDeviceSynchronize();
	//}

	if (d.triangleWalls.num > 0)
	{
		numObjects = d.triangleWalls.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		integrateBeforeContactCalculation << <grid, block >> > (d.triangleWalls.state, 
			gravity, 
			timeStep, 
			d.triangleWalls.num);
		//cudaDeviceSynchronize();
	}

	if (d.clumps.num > 0)
	{
		numObjects = d.clumps.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		integrateBeforeContactCalculation << <grid, block >> > (d.clumps.state,
			gravity,
			timeStep,
			d.clumps.num);
		updatePebblesMotionBeforeContact << <grid, block >> > (d.clumps,
			d.spheres,
			timeStep);
		//cudaDeviceSynchronize();
	}
}

void integrateAfterContact(DeviceData& d, double3 gravity, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	integrateAfterContactCalculation << <grid, block >> > (d.spheres.state,
		gravity,
		timeStep,
		d.spheres.num);
	//cudaDeviceSynchronize();

	if (d.triangleWalls.num > 0)
	{
		numObjects = d.triangleWalls.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		integrateAfterContactCalculation << <grid, block >> > (d.triangleWalls.state,
			gravity,
			timeStep,
			d.triangleWalls.num);
		//cudaDeviceSynchronize();
	}

	if (d.clumps.num > 0)
	{
		numObjects = d.clumps.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		integrateAfterContactCalculation << <grid, block >> > (d.clumps.state,
			gravity,
			timeStep,
			d.clumps.num);
		//cudaDeviceSynchronize();
	}
}