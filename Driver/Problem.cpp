#include "DEMSolver.h"

//template
class Problem : public DEMSolver
{
public:
	Problem() :DEMSolver() {}

	void loadHostData()override
	{
		auto& h = getHostData();
		//Step1: Set contact parameters
		//Step2: Set particle data
		//Step3: Set wall data
		//Step4: Set simulation parameters
	}

	void handleDataBeforeContact()override
	{
		auto& h = getHostData();
		auto& d = getDeviceData();
		// This function is called before the contact calculation step.
		// You can modify the host data or device data here if needed.
	}

	void handleDataAfterContact()override
	{
		auto& h = getHostData();
		auto& d = getDeviceData();
		// This function is called after the contact calculation step.
		// You can modify the host data or device data here if needed.
	}

	void outputData(int frame, int step) override
	{
		//Upload(Device -> Host)...
		uploadSphereState();
		auto& h = getHostData();
		// Output data to files, such as VTU files.
		writeSpheresVTU("spheres", h.spheres, frame, h.simulation.currentTime, step);
	}
};

class Cantilever : public DEMSolver
{
public:
	Cantilever() :DEMSolver() {}

	void loadHostData()override
	{
		auto& h = getHostData();
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
		h.simulation.domainOrigin = make_double3(-0.5, -0.5, -0.5);
		h.simulation.domainSize = make_double3(5, 1, 1);
		h.simulation.timeStep = 1.e-5;
		h.simulation.timeMax = 5.;
		h.simulation.nPrint = 10;
	}

	void handleDataAfterContact() override
	{
		auto& h = getHostData();
		uploadSphereState();
		h.spheres.state.forces[10] += make_double3(0, 0, 100e3);
		auto& d = getDeviceData();
		calculateGlobalDampingForceTorque(d.spheres, 0.1, h.simulation.maxThreadsPerBlock);
	}

	void outputData(int frame, int step) override
	{
		uploadSphereState();
		uploadBondedInteraction();
		auto& h = getHostData();
		if (step == 0)
		{
			std::cout << "Number of spheres: " << h.spheres.num << std::endl;
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSolidSpheresVTU("solidSpheres", h.spheres, frame, h.simulation.currentTime, step);
		writeBondedInteractionsVTU("sphSphBondedInteractions", h.sphSphBondedInteract, frame, h.simulation.currentTime, step);
	}
};

class DamBreak : public DEMSolver
{
public:
	DamBreak() :DEMSolver() {}

	void loadHostData()override
	{
		auto& h = getHostData();
		h.contactPara = HostContactParameter(2);
		h.contactPara.material.elasticModulus[1] = 200e9;
		h.contactPara.material.poissonRatio[1] = 0.3;

		h.SPHParticles.createBlockSample(h.spheres, make_double3(0., 0., 0.), make_double3(0.4, 0.6, 0.3), 1000., 0.0125, 0.01, 0.0, 30, 0);

		h.triangleWalls.addBoxWall(make_double3(0.9, 0.24, 0.), make_double3(0.12, 0.12, 0.6), 1);

		h.simulation.domainSize = make_double3(1.6, 0.6, 0.6);
		h.simulation.addBoundaryWalls = true;
		h.simulation.gravity = make_double3(0., 0., -9.81);
		h.simulation.timeStep = 0.25 * h.spheres.radii[0] / h.SPHParticles.c0;
		h.simulation.timeMax = 5.;
		h.simulation.nPrint = 500;
	}

	void outputData(int frame, int step) override
	{
		auto& h = getHostData();
		if (step == 0)
		{
			std::cout << "Number of spheres: " << h.spheres.num << std::endl;
			std::cout << "Number of triangle walls: " << h.triangleWalls.num << std::endl;
			if (h.triangleWalls.num > 0)
			{
				std::cout << "Number of triangle wall faces: " << h.triangleWalls.face.num << std::endl;
				std::cout << "Number of triangle wall edges: " << h.triangleWalls.edge.num << std::endl;
				std::cout << "Number of triangle wall vertices: " << h.triangleWalls.vertex.num << std::endl;
			}
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
			writeBoxSurfaceVTU("boundaryWalls", h.simulation.domainOrigin, h.simulation.domainOrigin + h.simulation.domainSize);
		}
		writeSPHSpheresVTU("SPH", h.SPHParticles, h.spheres, frame, h.simulation.currentTime, step);
		if (h.triangleWalls.num > 0.)
		{
			writeTriangleWallVTU("triangles", h.triangleWalls.vertex, h.triangleWalls.face, h.triangleWalls.state, frame, h.simulation.currentTime, step);
		}
	}
};

class DamBreak2 : public DEMSolver
{
public:
	DamBreak2() :DEMSolver() {}

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

	void loadHostData()override
	{
		auto& h = getHostData();
		h.contactPara = HostContactParameter(2);
		h.contactPara.material.elasticModulus[0] = 0.3e9;
		h.contactPara.material.poissonRatio[0] = 0.3;
		h.contactPara.material.elasticModulus[1] = 0.3e9;
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

		h.SPHParticles.createBlockSample(h.spheres, make_double3(0., 0., 0.), make_double3(3.5, 0.7, 0.4), 1000., 0.035, 0.01, 0.0, 40, 0);

		h.simulation.domainSize = make_double3(8, 0.7, 0.7);
		h.simulation.addBoundaryWalls = true;
		h.simulation.gravity = make_double3(0., 0., -9.81);
		double stiffness = 0.5 * h.contactPara.material.elasticModulus[1] * pi() * h.spheres.radii[0];
		double mass = 1. / h.spheres.state.inverseMass[0];
		double res = h.contactPara.Hertzian.restitution[iCP11];
		h.simulation.timeStep = calTimeStep(stiffness, mass, res);
		h.simulation.timeMax = 3.;
		h.simulation.nPrint = 300;
	}

	void outputData(int frame, int step) override
	{
		uploadSphereState();
		uploadSPHState();
		auto& h = getHostData();
		if (step == 0)
		{
			std::cout << "Number of spheres: " << h.spheres.num << std::endl;
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
			writeBoxSurfaceVTU("boundaryWalls", h.simulation.domainOrigin, h.simulation.domainOrigin + h.simulation.domainSize);
		}
		writeSPHSpheresVTU("SPH", h.SPHParticles, h.spheres, frame, h.simulation.currentTime, step);
		writeSolidSpheresVTU("solidSpheres", h.spheres, frame, h.simulation.currentTime, step);
	}
};

class Compression : public DEMSolver
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

	void loadHostData()override
	{
		auto& h = getHostData();
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
		h.contactPara.Linear.dissipation.normal[iCP01] = 0.01;
		h.contactPara.Linear.dissipation.sliding[iCP01] = 0.01;
		h.contactPara.Linear.friction.sliding[iCP01] = 0.1;
		h.contactPara.Linear.stiffness.normal[iCP11] = k;
		h.contactPara.Linear.stiffness.sliding[iCP11] = k / h.contactPara.Bond.kN_to_kS_ratio[iCP11];
		h.contactPara.Linear.dissipation.normal[iCP11] = 0.01;
		h.contactPara.Linear.dissipation.sliding[iCP11] = 0.01;
		h.contactPara.Linear.friction.sliding[iCP11] = 0.1;
		h.contactPara.Bond.maxContactGap[iCP11] = 0.1 * r;
		h.spheres.createHEXBlockSample(make_double3(0.2, 0.2, 0.), make_double3(0.2, 0.2, 0.6), make_double3(0., 0., 0.), 900, spacing, r, 0, 1);

		h.triangleWalls.addPlaneWall(make_double3(0.3, 0.3, 0.), make_double3(0.1, 0.1, 0.), make_double3(0.5, 0.1, 0.), make_double3(0.5, 0.5, 0.), make_double3(0.1, 0.5, 0.), 0);
		h.triangleWalls.addPlaneWall(make_double3(0.3, 0.3, 0.), make_double3(0.1, 0.1, 0.6), make_double3(0.5, 0.1, 0.6), make_double3(0.5, 0.5, 0.6), make_double3(0.1, 0.5, 0.6), 0);
		h.triangleWalls.state.velocities[1] = make_double3(0., 0., -0.1);

		h.simulation.domainSize = make_double3(0.6, 0.6, 0.6);
		//h.simulation.gravity = make_double3(0., 0., -9.81);
		double mass = 1. / h.spheres.state.inverseMass[0];
		h.simulation.timeStep = calTimeStep(k, mass, 1);
		h.simulation.timeMax = 0.1;
		h.simulation.nPrint = 100;
	}

	void outputData(int frame, int step) override
	{
		uploadSphereState();
		uploadBondedInteraction();
		uploadTriangleWallState();
		auto& h = getHostData();
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
		}
		writeSolidSpheresVTU("solidSpheres", h.spheres, frame, h.simulation.currentTime, step);
		//writeBasicInteractionsVTU("sphSphInteractions", h.sphSphInteract, frame, h.simulation.currentTime, step);
		writeBondedInteractionsVTU("sphSphBondedInteractions", h.sphSphBondedInteract, frame, h.simulation.currentTime, step);
		writeTriangleWallVTU("triangles", h.triangleWalls.vertex, h.triangleWalls.face, h.triangleWalls.state, frame, h.simulation.currentTime, step);
	}
};

class LevelIce : public DEMSolver
{
public:
	LevelIce() :DEMSolver() {}

	double calTimeStep(double stiffness, double mass, double restitution)
	{
		if (stiffness <= 0 || mass <= 0 || restitution <= 0 || restitution > 1)
		{
			std::cerr << "Invalid parameters for time step calculation.\n";
			return 0.0;
		}
		double dissipation = -log(restitution) / sqrt(log(restitution) * log(restitution) + pi() * pi());
		double dt = pi() / sqrt(stiffness / mass);
		dt /= (1.0 - pow(dissipation, 2)); // Adjust for dissipation
		dt /= 50.;
		return dt;
	}

	void loadHostData()override
	{
		auto& h = getHostData();
		h.contactPara = HostContactParameter(2);
		h.contactPara.material.elasticModulus[0] = 200e9;
		h.contactPara.material.poissonRatio[0] = 0.3; // Elastic modulus and Poisson's ratio for the first material (e.g., steel)
		h.contactPara.material.elasticModulus[1] = 1e9; // Elastic modulus for ice
		h.contactPara.material.poissonRatio[1] = 0.3; // Poisson's ratio for ice

		int iCP01 = h.contactPara.getContactParameterIndex(0, 1);
		h.contactPara.Hertzian.restitution[iCP01] = 0.3;
		int iCP11 = h.contactPara.getContactParameterIndex(1, 1);
		h.contactPara.Bond.elasticModulus[iCP11] = 1e9;
		h.contactPara.Bond.kN_to_kS_ratio[iCP11] = 10;
		h.contactPara.Bond.tensileStrength[iCP11] = 0.5e6;
		h.contactPara.Bond.cohesion[iCP11] = 0.5e6;
		h.contactPara.Bond.frictionCoeff[iCP11] = 0.1;
		h.contactPara.Bond.criticalDamping[iCP11] = 0.1;

		double hi = 0.1;
		int nz = 3;
		double spacing = 3 * hi / double(nz) / sqrt(6.);
		double r = 0.5 * spacing;
		double k = h.contactPara.Bond.elasticModulus[iCP11] * pi() * r / 2.;
		h.contactPara.Linear.stiffness.normal[iCP11] = k;
		h.contactPara.Linear.stiffness.sliding[iCP11] = k / h.contactPara.Bond.kN_to_kS_ratio[iCP11];
		h.contactPara.Linear.dissipation.normal[iCP11] = 0.1;
		h.contactPara.Linear.dissipation.sliding[iCP11] = 0.1;
		h.contactPara.Linear.friction.sliding[iCP11] = 0.1;
		h.contactPara.Bond.maxContactGap[iCP11] = 0.1 * r;
		h.spheres.createHEXBlockSample(make_double3(0, 0, -0.9 * hi), make_double3(5, 10, hi), make_double3(0.1, 0., 0.), 900, spacing, r, 0, 1);

		h.triangleWalls.addVerticalCylinder(make_double3(5.55, 5, -2.), 0.5, 2.5, 0, 24);

		h.simulation.domainOrigin = make_double3(0., 0, -2);
		h.simulation.domainSize = make_double3(10, 10, 2.5);
		h.simulation.gravity = make_double3(0., 0., -9.81);
		//h.simulation.addBoundaryWalls = true;
		double mass = 1. / h.spheres.state.inverseMass[0];
		h.simulation.timeStep = calTimeStep(k, mass, 1);
		h.simulation.timeMax = 20;
		h.simulation.nPrint = 500;

		//fixed particles
		for (int i = 0; i < h.spheres.num; ++i)
		{
			double3 pos = h.spheres.state.positions[i];
			if (pos.y < 4 * r || pos.y > 10 - 4 * r || pos.x < 4 * r)
			{
				h.spheres.state.inverseMass[i] = 0.;
			}
		}
	}

	void handleDataAfterContact()override
	{
		auto& h = getHostData();
		auto& d = getDeviceData();
		double waterDensity = 1000.0; // Density of the fluid (e.g., water)
		double waterLevel0 = 0; // Initial water level
		double Cd = 0.4; // Drag coefficient for spheres in water
		double3 currentVel = make_double3(0.1, 0., 0.); // Current velocity of the fluid
		calculateHydroForce(d.spheres, currentVel, waterDensity, waterLevel0, Cd, h.simulation.maxThreadsPerBlock);
	}

	void outputData(int frame, int step) override
	{
		uploadSphereState();
		uploadBondedInteraction();
		uploadTriangleWallState();
		auto& h = getHostData();
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
			writeBoxSurfaceVTU("boundaryWalls", h.simulation.domainOrigin, h.simulation.domainOrigin + h.simulation.domainSize);
			writeTriangleWallVTU("triangles", h.triangleWalls.vertex, h.triangleWalls.face, h.triangleWalls.state, frame, h.simulation.currentTime, step);
		}
		writeSolidSpheresVTU("solidSpheres", h.spheres, frame, h.simulation.currentTime, step);
		writeBondedInteractionsVTU("sphSphBondedInteractions", h.sphSphBondedInteract, frame, h.simulation.currentTime, step);
	}
};

class Icebreaker : public DEMSolver
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

	void loadHostData()override
	{
		auto& h = getHostData();
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
		h.contactPara.Bond.criticalDamping[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Bond.elasticModulus[iCP12] = 1e8;
		h.contactPara.Bond.kN_to_kS_ratio[iCP12] = 2.6;
		h.contactPara.Bond.criticalDamping[iCP12] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());

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
		h.contactPara.Linear.stiffness.rolling[iCP01] = ks;
		h.contactPara.Linear.stiffness.torsion[iCP01] = ks;
		h.contactPara.Linear.dissipation.normal[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.sliding[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.rolling[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.torsion[iCP01] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.friction.sliding[iCP01] = 0.1;
		h.contactPara.Linear.friction.rolling[iCP01] = 0.1;
		h.contactPara.Linear.friction.torsion[iCP01] = 0.1;
		h.contactPara.Linear.stiffness.normal[iCP11] = k;
		h.contactPara.Linear.stiffness.sliding[iCP11] = ks;
		h.contactPara.Linear.stiffness.rolling[iCP11] = ks;
		h.contactPara.Linear.stiffness.torsion[iCP11] = ks;
		h.contactPara.Linear.dissipation.normal[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.sliding[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.rolling[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.dissipation.torsion[iCP11] = -log(0.3) / sqrt(log(0.3) * log(0.3) + pi() * pi());
		h.contactPara.Linear.friction.sliding[iCP11] = 0.1;
		h.contactPara.Linear.friction.rolling[iCP11] = 0.1;
		h.contactPara.Linear.friction.torsion[iCP11] = 0.1;

		h.spheres.createHEXBlockSample(make_double3(0, 0, -0.9 * hi), make_double3(200, 100, hi), make_double3(0., 0., 0.), 910, spacing, r, 0, 1);
		loadTriangleWallInfo("Ship.dat", h.triangleWalls);
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

		h.simulation.domainOrigin = make_double3(-40, 0, -12);
		h.simulation.domainSize = make_double3(245, 100, 15);
		h.simulation.gravity = make_double3(0., 0., -9.81);
		h.simulation.timeStep = calTimeStep(k, mass, 0.3);
		h.simulation.timeMax = 180;
		h.simulation.nPrint = 250;
	}

	void handleDataBeforeContact()override
	{
		auto& h = getHostData();
		int step = int(h.simulation.currentTime / h.simulation.timeStep);
		int gap = int(0.01 / h.simulation.timeStep);
		if (step % gap == 0)
		{
			uploadTriangleWallState();
			writeHostDynamicStateToDat(h.triangleWalls.state, "wallDynamic", h.simulation.currentTime);
		}
	};

	void handleDataAfterContact()override
	{
		auto& h = getHostData();
		auto& d = getDeviceData();
		double waterDensity = 1000.0; // Density of the fluid (e.g., water)
		double waterLevel0 = 0; // Initial water level
		double Cd = 0.1; // Drag coefficient for spheres in water
		double3 currentVel = make_double3(0., 0., 0.); // Current velocity of the fluid
		calculateHydroForce(d.spheres, currentVel, waterDensity, waterLevel0, Cd, h.simulation.maxThreadsPerBlock);
	}

	void outputData(int frame, int step) override
	{
		auto& h = getHostData();
		uploadSphereState();
		uploadBondedInteraction();
		uploadTriangleWallState();
		if (frame == 0)
		{
			int n = removeVtuFiles("outputData");
			printf("Removed %d .vtu files in ./outputData\n", n);
			writeBoxSurfaceVTU("boundaryWalls", h.simulation.domainOrigin, h.simulation.domainOrigin + h.simulation.domainSize);
		}
		writeSpheresVTU("solidSpheres", h.spheres, frame, h.simulation.currentTime, step);
		writeBondedInteractionsVTU("sphSphBondedInteractions", h.sphSphBondedInteract, frame, h.simulation.currentTime, step);
		writeTriangleWallPressureVTU("triangles", h.faceSphInteract, h.edgeSphInteract, h.vertexSphInteract, h.triangleWalls, frame, h.simulation.currentTime, step);
	}
};

int main()
{
	Icebreaker problem;
	problem.solve();
}