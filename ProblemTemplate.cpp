#include "Solver.h"

//template
class Problem : public DEMSolver
{
public:
	Problem() :DEMSolver() {}

	void loadHostData(HostData& h)override
	{
		//Step1: Set contact parameters
		//Step2: Set particle data
		//Step3: Set wall data
		//Step4: Set simulation parameters
	}

	void handleDataBeforeContact()override
	{
		// This function is called before the contact calculation step.
		// You can modify the device data here if needed.
	}

	void handleDataAfterContact()override
	{
		// This function is called after the contact calculation step.
		// You can modify the device data here if needed.
	}

	void outputData(int frame) override
	{
		// Upload(Device -> Host)...
		// Output data to files, such as VTU files.
	}
};

class Cantilever : public DEMSolver
{
public:
	Cantilever() :DEMSolver() {}

	void loadHostData(HostData& h)override
	{
		h.contactPara = HostContactParameter(1);
		h.contactPara.Bond.elasticModulus[0] = 200e9;
		h.contactPara.Bond.kN_to_kS_ratio[0] = 2.6;

		double r = 0.2;
		h.contactPara.Bond.maxContactGap[0] = 0.1 * r;
		double mass = 4. / 3. * pi() * pow(r, 3) * 7800;
		double inertia = 0.4 * mass * pow(r, 2);
		h.spheres = HostSphere(11);
		for (int i = 0; i < 11; i++)
		{
			h.spheres.radii[i] = r;
			h.spheres.materialIndex[i] = 0;
			h.spheres.bondClusterIndex[i] = 0;
			h.spheres.state.positions[i] = make_double3(2 * r * i, 0, 0);
			h.spheres.state.inverseMass[i] = 1. / mass * (i != 0);
			h.spheres.state.inverseInertia[i] = inverse(make_symMatrix(inertia, inertia, inertia, 0, 0, 0));
		}

		setDomainGravity(make_double3(-0.5, -0.5, -0.5), make_double3(5, 1, 1), make_double3(0, 0, 0));
		setTimeMaxTimeStepPrintNumber(5, 1.e-5, 10);
	}

	void handleDataAfterContact() override
	{
		HostSphere hSph = getHostSphere();
		hSph.state.forces[10] += make_double3(0, 0, 100e3);
		auto& dSph = getDeviceSphere();
		dSph.downloadState(hSph);
		calculateGlobalDampingForceTorque(dSph, 0.1, 256);
	}

	void outputData(int frame) override
	{
		if (getCurrentStep() == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSolidSpheresVTU("solidSpheres", getHostSphere(), frame, getCurrentTime(), getCurrentStep());
		writeBondedInteractionsVTU("sphSphBondedInteractions", getHostBondedInteraction(), frame, getCurrentTime(), getCurrentStep());
	}
};

class DamBreak: public SPHSolver
{
public:
	DamBreak():SPHSolver(){}

	void loadHostData(HostData& h)override
	{
		h.contactPara = HostContactParameter(2);
		h.contactPara.material.elasticModulus[1] = 200e9;
		h.contactPara.material.poissonRatio[1] = 0.3;

		h.SPHParticles.createBlockSample(h.spheres, make_double3(0, 0, 0), make_double3(0.4, 0.6, 0.3), 1000, 0.0125, 0.01, 0, 30, 0);
		h.triangleWalls.addBoxWall(make_double3(0.9, 0.24, 0.), make_double3(0.12, 0.12, 0.6), 1);
		//h.triangleWalls.addBoxWall(make_double3(0, 0, 0), make_double3(1.6, 0.6, 0.6), 1);
		setDomainGravity(make_double3(0, 0, 0), make_double3(1.6, 0.6, 0.6), make_double3(0., 0., -9.81));
		setBoundaryWallX(0);
		setBoundaryWallY(0);
		setBoundaryWallZ(0);
		setTimeMaxTimeStepPrintNumber(10, 0.25 * h.spheres.radii[0] / h.SPHParticles.c0, 250);
	}

	void outputData(int frame) override
	{
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSPHSpheresVTU("SPH", getHostSPH(), getHostSphere(), frame, getCurrentTime(), getCurrentStep());
		writeTriangleWallVTU("triangles", getHostTriangleWall(), frame, getCurrentTime(), getCurrentStep());
	}
};

class DamBreak2 : public DEMSPHSolver
{
public:
	DamBreak2() :DEMSPHSolver() {}

	double calTimeStep(double stiffness, double mass, double restitution)
	{
		if (stiffness <= 0 || mass <= 0 || restitution <= 0 || restitution > 1)
		{
			std::cerr << "Invalid parameters for time step calculation.\n";
			return 0.0;
		}
		double dissipation = -log(restitution) / sqrt(log(restitution) * log(restitution) + pi() * pi());
		double dt = 2. * pi() / sqrt(stiffness / mass);
		dt /= (1.0 - pow(dissipation, 2)); // Adjust for dissipation
		dt /= 100.;
		return dt;
	}

	void loadHostData(HostData& h)override
	{
		h.contactPara = HostContactParameter(3);
		h.contactPara.material.elasticModulus[0] = 3e9;
		h.contactPara.material.poissonRatio[0] = 0.3;
		h.contactPara.material.elasticModulus[1] = 3e9;
		h.contactPara.material.poissonRatio[1] = 0.3;

		int iCP01 = h.contactPara.getContactParameterIndex(0, 1);
		h.contactPara.Hertzian.restitution[iCP01] = 0.2;
		h.contactPara.Hertzian.friction.sliding[iCP01] = 0.45;
		int iCP11 = h.contactPara.getContactParameterIndex(1, 1);
		h.contactPara.Hertzian.restitution[iCP11] = 0.2;
		h.contactPara.Bond.maxContactGap[iCP11] = 0.002;
		h.contactPara.Hertzian.friction.sliding[iCP11] = 0.35;

		h.clumps.createBlockSample(h.spheres, make_double3(5.2, 0.0625, 0.), make_double3(0.15, 0.15, 0.15), make_double3(0., 0., 0.), 800, 0.03, 0.015, 1);
		h.clumps.createBlockSample(h.spheres, make_double3(5.2, 0.275, 0.), make_double3(0.15, 0.15, 0.15), make_double3(0., 0., 0.), 800, 0.03, 0.015, 1);
		h.clumps.createBlockSample(h.spheres, make_double3(5.2, 0.4875, 0.), make_double3(0.15, 0.15, 0.15), make_double3(0., 0., 0.), 800, 0.03, 0.015, 1);
		h.clumps.createBlockSample(h.spheres, make_double3(5.2, 0.16875, 0.15), make_double3(0.15, 0.15, 0.15), make_double3(0., 0., 0.), 800, 0.03, 0.015, 1);
		h.clumps.createBlockSample(h.spheres, make_double3(5.2, 0.38125, 0.15), make_double3(0.15, 0.15, 0.15), make_double3(0., 0., 0.), 800, 0.03, 0.015, 1);
		h.clumps.createBlockSample(h.spheres, make_double3(5.2, 0.275, 0.3), make_double3(0.15, 0.15, 0.15), make_double3(0., 0., 0.), 800, 0.03, 0.015, 1);

		h.SPHParticles.createBlockSample(h.spheres, make_double3(0., 0., 0.), make_double3(3.5, 0.7, 0.4), 1000., 0.02, 0.01, 0.0, 40, 2);
		
		setDomainGravity(make_double3(0, 0, 0), make_double3(8, 0.7, 0.7), make_double3(0., 0., -9.81));
		setBoundaryWallX(0);
		setBoundaryWallY(0);
		setBoundaryWallZ(0);
		double stiffness = 0.5 * h.contactPara.material.elasticModulus[1] * pi() * h.spheres.radii[0];
		double mass = 1. / h.spheres.state.inverseMass[0];
		double res = h.contactPara.Hertzian.restitution[iCP11];
		double timeStep = calTimeStep(stiffness, mass, res);
		setTimeMaxTimeStepPrintNumber(6, timeStep, 150);
	}

	void outputData(int frame) override
	{
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSPHSpheresVTU("SPH", getHostSPH(), getHostSphere(), frame, getCurrentTime(), getCurrentStep());
		writeSolidSpheresVTU("solidSpheres",getHostSphere(), frame, getCurrentTime(), getCurrentStep());
	}
};

class Compression :public DEMSolver
{
public:
	Compression() :DEMSolver() {}

	double calTimeStep(double stiffness, double mass, double restitution)
	{
		if (stiffness <= 0 || mass <= 0 || restitution <= 0 || restitution > 1)
		{
			std::cerr << "Invalid parameters for time step calculation.\n";
			return 0.0;
		}
		double dissipation = -log(restitution) / sqrt(log(restitution) * log(restitution) + pi() * pi());
		double dt = 2. * pi() / sqrt(stiffness / mass);
		dt /= (1.0 - pow(dissipation, 2)); // Adjust for dissipation
		dt /= 100.;
		return dt;
	}

	void loadHostData(HostData& h)override
	{
		h.contactPara = HostContactParameter(2);

		int iCP01 = h.contactPara.getContactParameterIndex(0, 1);
		int iCP11 = h.contactPara.getContactParameterIndex(1, 1);
		h.contactPara.Bond.elasticModulus[iCP11] = 3e9;
		h.contactPara.Bond.kN_to_kS_ratio[iCP11] = 2.6;
		h.contactPara.Bond.tensileStrength[iCP11] = 1e6;
		h.contactPara.Bond.cohesion[iCP11] = 1e6;
		h.contactPara.Bond.frictionCoeff[iCP11] = 0.1;

		double spacing = 0.0125;
		double r = 0.5 * spacing;
		double k = h.contactPara.Bond.elasticModulus[iCP11] * pi() * r / 2.;
		h.contactPara.Linear.stiffness.normal[iCP01] = k;
		h.contactPara.Linear.stiffness.sliding[iCP01] = k / h.contactPara.Bond.kN_to_kS_ratio[iCP11];
		h.contactPara.Linear.dissipation.normal[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.sliding[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());;
		h.contactPara.Linear.friction.sliding[iCP01] = 0.1;
		h.contactPara.Linear.stiffness.normal[iCP11] = k;
		h.contactPara.Linear.stiffness.sliding[iCP11] = k/ h.contactPara.Bond.kN_to_kS_ratio[iCP11];
		h.contactPara.Linear.dissipation.normal[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());;
		h.contactPara.Linear.dissipation.sliding[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());;
		h.contactPara.Linear.friction.sliding[iCP11] = 0.1;
		h.contactPara.Bond.maxContactGap[iCP11] = 0.1 * r;
		h.spheres.createHEXBlockSample(make_double3(0.2, 0.2, 0.), make_double3(0.2, 0.2, 0.6), make_double3(0., 0., 0.), 900, spacing, r, 0, 1);

		h.triangleWalls.addPlaneWall(make_double3(0.3, 0.3, 0.), make_double3(0.1, 0.1, 0.), make_double3(0.5, 0.1, 0.), make_double3(0.5, 0.5, 0.), make_double3(0.1, 0.5, 0.), 0);
		h.triangleWalls.addPlaneWall(make_double3(0.3, 0.3, 0.), make_double3(0.1, 0.1, 0.6), make_double3(0.5, 0.1, 0.6), make_double3(0.5, 0.5, 0.6), make_double3(0.1, 0.5, 0.6), 0);
		h.triangleWalls.state.velocities[1] = make_double3(0., 0., -0.1);

		double mass = 1. / h.spheres.state.inverseMass[0];
		double timeStep = calTimeStep(k, mass, 1);
		setDomainGravity(make_double3(0, 0, 0), make_double3(0.6, 0.6, 0.6), make_double3(0., 0., 0.));
		setTimeMaxTimeStepPrintNumber(0.1, timeStep, 100);
	}

	void outputData(int frame) override
	{
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSolidSpheresVTU("solidSpheres", getHostSphere(), frame, getCurrentTime(), getCurrentStep());
		writeBondedInteractionsVTU("sphSphBondedInteractions", getHostBondedInteraction(), frame, getCurrentTime(), getCurrentStep());
		writeTriangleWallVTU("triangles", getHostTriangleWall(), frame, getCurrentTime(), getCurrentStep());
	}
};

class Icebreaker: public DEMSolver
{
public:
	Icebreaker() :DEMSolver() {}

	double calTimeStep(double stiffness, double mass, double res)
	{
		if (stiffness <= 0 || mass <= 0 || res <= 0 || res > 1)
		{
			std::cerr << "Invalid parameters for time step calculation.\n";
			return 0.0;
		}
		double dissipation = -log(res) / sqrt(log(res) * log(res) + pi() * pi());
		double dt = pi() / sqrt(stiffness / mass);
		dt /= (1.0 - pow(dissipation, 2)); // Adjust for dissipation
		dt /= 50.;
		return dt;
	}

	void loadHostData(HostData& h)override
	{
		h.contactPara = HostContactParameter(3);
		h.contactPara.material.elasticModulus[0] = 200e9;
		h.contactPara.material.poissonRatio[0] = 0.3; // Elastic modulus and Poisson's ratio for the first material (e.g., steel)
		h.contactPara.material.elasticModulus[1] = 1e9; // Elastic modulus for ice
		h.contactPara.material.poissonRatio[1] = 0.3; // Poisson's ratio for ice
		h.contactPara.material.elasticModulus[2] = 1e9; // Elastic modulus for ice
		h.contactPara.material.poissonRatio[2] = 0.3; // Poisson's ratio for ice

		int iCP01 = h.contactPara.getContactParameterIndex(0, 1);
		int iCP11 = h.contactPara.getContactParameterIndex(1, 1);
		int iCP12 = h.contactPara.getContactParameterIndex(1, 2);
		h.contactPara.Bond.elasticModulus[iCP11] = 1e9;
		h.contactPara.Bond.kN_to_kS_ratio[iCP11] = 2.6;
		h.contactPara.Bond.tensileStrength[iCP11] = 0.5e6;
		h.contactPara.Bond.cohesion[iCP11] = 0.5e6;
		h.contactPara.Bond.frictionCoeff[iCP11] = 0.2;
		h.contactPara.Bond.elasticModulus[iCP12] = 1e7;
		h.contactPara.Bond.kN_to_kS_ratio[iCP12] = 2.6;

		double hi = 1.2;
		int nz = 2;
		double spacing = 3 * hi / double(nz) / sqrt(6.);
		if (nz == 1) spacing = hi;
		double r = 0.5 * spacing;
		h.contactPara.Bond.maxContactGap[iCP11] = 0.1 * r;
		h.contactPara.Bond.maxContactGap[iCP12] = 0.5 * r;
		double k = h.contactPara.Bond.elasticModulus[iCP11] * pi() * r / 2.;
		double ks = k / h.contactPara.Bond.kN_to_kS_ratio[iCP11];
		h.contactPara.Linear.stiffness.normal[iCP01] = k;
		h.contactPara.Linear.stiffness.sliding[iCP01] = ks;
		h.contactPara.Linear.dissipation.normal[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.sliding[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.friction.sliding[iCP01] = 0.1;
		h.contactPara.Linear.stiffness.normal[iCP11] = k;
		h.contactPara.Linear.stiffness.sliding[iCP11] = ks;
		h.contactPara.Linear.dissipation.normal[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.sliding[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.friction.sliding[iCP11] = 0.2;
		
		h.spheres.createHEXBlockSample(make_double3(0, 0, -0.9 * hi), make_double3(200, 100, hi), make_double3(0., 0., 0.), 910, spacing, r, 0, 1);
		loadTriangleWallInfo("Ship.dat",h.triangleWalls);
		h.triangleWalls.state.velocities[0] = make_double3(1.0, 0., 0.);

		double volMerge = 0;
		for (int i = 0; i < h.spheres.num; ++i)
		{
			double3 pos = h.spheres.state.positions[i];
			double depth = r - (pos.z - 0);
			if (depth >= 2 * r)
			{
				volMerge += 4. / 3. * pi() * pow(r, 3);
			}
			else if (depth > 0)
			{
				volMerge += (pi() / 3) * depth * depth * (3 * r - depth);
			}
		}
		double volSpheres = 4. / 3. * pi() * pow(r, 3) * h.spheres.num;
		double mass = 4. / 3. * pow(r, 3) * pi() * volMerge / volSpheres * 1000;
		double inertia = 0.4 * mass * pow(r, 2);
		for (int i = 0; i < h.spheres.num; ++i)
		{
			h.spheres.state.inverseMass[i] = 1. / mass;
			symMatrix inertiaTensor = make_symMatrix(inertia, inertia, inertia, 0., 0., 0.);
			h.spheres.state.inverseInertia[i] = inverse(inertiaTensor);
			double3 pos = h.spheres.state.positions[i];
			if (pos.y < 6 * r || pos.y > 100 - 6 * r || pos.x > 200 - 6 * r)
			{
				h.spheres.state.inverseMass[i] = 0.;
				h.spheres.materialIndex[i] = 2; // Set fixed particles to material index 2
			}
		}
		setDomainGravity(make_double3(-40, 0, -12), make_double3(245, 100, 15), make_double3(0., 0., -9.81));
		setTimeMaxTimeStepPrintNumber(180, calTimeStep(k, mass, 0.3), 250);
	}

	void handleDataAfterContact()override
	{
		double waterDensity = 1000.0; // Density of the fluid (e.g., water)
		double waterLevel0 = 0; // Initial water level
		double Cd = 0.1; // Drag coefficient for spheres in water
		double3 waterVel = make_double3(0., 0., 0.); // Current velocity of the fluid
		calculateHydroForce(getDeviceSphere(), waterVel, waterDensity, waterLevel0, Cd, 256);

		int step = getCurrentStep();
		int gap = int(0.01 / getTimeStep());
		if (step % gap == 0)
		{
			writeHostDynamicStateToDat(getHostTriangleWall().state, 0, "wallDynamic", getCurrentTime());
		}
	}

	void outputData(int frame) override
	{
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSolidSpheresVTU("solidSpheres", getHostSphere(), frame, getCurrentTime(), getCurrentStep());
		writeBondedInteractionsVTU("sphSphBondedInteractions", getHostBondedInteraction(), frame, getCurrentTime(), getCurrentStep());
		writeTriangleWallPressureVTU("triangles", getHostFaceSphereInteraction(), getHostEdgeSphereInteraction(), getHostVertexSphereInteraction(), getHostTriangleWall(), frame, getCurrentTime(), getCurrentStep());
	}
};

int main()
{
	DamBreak problem;
	problem.solve();
}