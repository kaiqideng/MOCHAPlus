#include "ExternalForceTorque.cuh"

__global__ void calHydroForce(Sphere sph,
	double3 currentVel,
	double waterDensity,
	double waterLevel0,
	double Cd)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= sph.num)  return;

	double r = sph.radii[idx];
	double3 pos = sph.state.positions[idx];
	double3 vel = sph.state.velocities[idx];
	double3 ang = sph.state.angularVelocities[idx];
	double3 hydroForce = make_double3(0., 0., 0.);
	double3 hydroTorque = make_double3(0., 0., 0.);
	double depth = r - (pos.z - waterLevel0);
	if (depth >= 2 * r)
	{
		double vol = 4. / 3. * pi() * pow(r, 3);
		hydroForce = make_double3(0., 0., waterDensity * vol * 9.81);
	}
	else if (depth > 0)
	{
		double vol = (pi() / 3) * depth * depth * (3 * r - depth);
		hydroForce = make_double3(0., 0., waterDensity * vol * 9.81);
	}
	if (pos.z - r < waterLevel0)
	{
		double3 relVel = currentVel - vel;
		double3 dragForce = 0.5 * Cd * pi() * pow(r, 2) * waterDensity * length(relVel) * relVel;
		hydroForce += dragForce;
		hydroTorque = 0.5 * Cd * pi() * pow(r, 4) * waterDensity * length(ang) * (-ang);
	}
	sph.state.forces[idx] += hydroForce;
	sph.state.torques[idx] += hydroTorque;
}

__global__ void calGlobalDampingForceTorque(Sphere sph,
	double globalDampingCoff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= sph.num) return;

	double absoluteForce = length(sph.state.forces[idx]);
	double absoluteTorque = length(sph.state.torques[idx]);
	double absoluteVelocity = length(sph.state.velocities[idx]);
	double absoluteAngularVelocity = length(sph.state.angularVelocities[idx]);
	if (absoluteVelocity > EPS_DOT) sph.state.forces[idx] -= globalDampingCoff * absoluteForce * normalize(sph.state.velocities[idx]);
	if (absoluteAngularVelocity > EPS_DOT) sph.state.torques[idx] -= globalDampingCoff * absoluteTorque * normalize(sph.state.angularVelocities[idx]);
}

void calculateHydroForce(Sphere sph,
	double3 currentVel,
	double waterDensity,
	double waterLevel0,
	double Cd,
	int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;
	numObjects = sph.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calHydroForce << <grid, block >> > (sph,
			currentVel,
			waterDensity,
			waterLevel0,
			Cd);
		//cudaDeviceSynchronize();
	}
}

void calculateGlobalDampingForceTorque(Sphere sph,
	double globalDampingCoff,
	int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;
	numObjects = sph.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calGlobalDampingForceTorque << <grid, block >> > (sph,
			globalDampingCoff);
		//cudaDeviceSynchronize();
	}
}