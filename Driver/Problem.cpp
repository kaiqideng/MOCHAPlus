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
		writeSolidSpheresVTU("solidSpheres", getHostSphere(), frame, getCurrentTime(), getCurrentStep());
	}
};

int main()
{
	DamBreak2 problem;
	problem.solve();
}