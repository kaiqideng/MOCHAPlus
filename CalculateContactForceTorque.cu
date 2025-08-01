#include "CalculateContactForceTorque.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)       // sm 6.0+
__device__ __forceinline__
double atomicAddDouble(double* addr, double val)
{
	return atomicAdd(addr, val);
}
#else                                                   
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	auto  addr_ull = reinterpret_cast<unsigned long long*>(addr);
	unsigned long long old = *addr_ull, assumed;

	do {
		assumed = old;
		double  old_d = __longlong_as_double(assumed);
		double  new_d = old_d + val;
		old = atomicCAS(addr_ull, assumed, __double_as_longlong(new_d));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

__device__ __forceinline__ void atomicAddDouble3(double3* arr, int idx, const double3& v)   // arr[idx] += v;
{
	atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}

__global__ void calSolidSphereContactForceTorque(BasicInteraction I, 
	Sphere sph, 
	ContactParameter CP, 
	double timeStep)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= I.num) return;

	int idxA = I.objectPointed[idx];
	int idxB = I.objectPointing[idx];

	if (sph.SPHIndex[idxA] >= 0 || sph.SPHIndex[idxB] >= 0) return;//If either sphere is SPH, skip

	double radA = sph.radii[idxA];
	double radB = sph.radii[idxB];
	double3 posA = sph.state.positions[idxA];
	double3 posB = sph.state.positions[idxB];
	//Update contact information of every sphere-sphere interaction (The contact information is updated in Neighbor Search every 30 steps)
	double normalOverlap = radA + radB - length(posA - posB);
	double3 contactNormal = normalize(posA - posB);
	double3 contactPoint = posB + (radB - 0.5 * normalOverlap) * contactNormal;
	I.normalOverlap[idx] = normalOverlap;
	I.contactNormal[idx] = contactNormal;
	I.contactPoint[idx] = contactPoint;

	double3 contactForce = make_double3(0., 0., 0.);
	double3 contactTorque = make_double3(0., 0., 0.);
	double effectiveRadius = radA * radB / (radA + radB);
	double inverseMassA = sph.state.inverseMass[idxA];
	double inverseMassB = sph.state.inverseMass[idxB];
	double effectiveMass = 1. / (inverseMassA + inverseMassB);//checked in NeighborSearch
	double3 velA = sph.state.velocities[idxA];
	double3 velB = sph.state.velocities[idxB];
	double3 angVelA = sph.state.angularVelocities[idxA];
	double3 angVelB = sph.state.angularVelocities[idxB];
	double3 slidingSpring = I.slidingSpring[idx];
	double3 rollingSpring = I.rollingSpring[idx];
	double3 torsionSpring = I.torsionSpring[idx];
	double3 relativeVelocityAtContact = velA + cross(angVelA, contactPoint - posA) - (velB + cross(angVelB, contactPoint - posB));
	double3 relativeAngularVelocityAtContact = angVelA - angVelB;
	int iMA = sph.materialIndex[idxA];
	int iMB = sph.materialIndex[idxB];
	int iCP = CP.getContactParameterIndex(iMA, iMB);
	double E1 = CP.material.elasticModulus[iMA];
	double E2 = CP.material.elasticModulus[iMB];
	if (CP.Linear.stiffness.normal[iCP] > 0.)
	{
		LinearContact(contactForce, contactTorque, slidingSpring, rollingSpring, torsionSpring, relativeVelocityAtContact, relativeAngularVelocityAtContact, contactNormal, normalOverlap, effectiveMass, effectiveRadius, timeStep, CP.Linear.stiffness.normal[iCP], CP.Linear.stiffness.sliding[iCP], CP.Linear.stiffness.rolling[iCP], CP.Linear.stiffness.torsion[iCP], CP.Linear.dissipation.normal[iCP], CP.Linear.dissipation.sliding[iCP], CP.Linear.dissipation.rolling[iCP], CP.Linear.dissipation.torsion[iCP], CP.Linear.friction.sliding[iCP], CP.Linear.friction.rolling[iCP], CP.Linear.friction.torsion[iCP]);
	}
	else if(E1 != 0. && E2 != 0.)
	{
		double v1 = CP.material.poissonRatio[iMA];
		double v2 = CP.material.poissonRatio[iMB];
		double effElasticModulus = 0, effShearModulus = 0;
	    effElasticModulus = 1. / (((1 - v1 * v1) / E1) + ((1 - v2 * v2) / E2));
		double G1 = E1 / (2 * (1 + v1)), G2 = E2 / (2 * (1 + v2));
		effShearModulus = 1. / ((2 - v1) / G1 + (2 - v2) / G2);
		double r = CP.Hertzian.restitution[iCP];
		double logR = log(r);
		double dissipation = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(contactForce, contactTorque, slidingSpring, rollingSpring, torsionSpring, relativeVelocityAtContact, relativeAngularVelocityAtContact, contactNormal, normalOverlap, effectiveMass, effectiveRadius, timeStep, effElasticModulus, effShearModulus, dissipation, CP.Hertzian.kR_to_kS_ratio[iCP], CP.Hertzian.kT_to_kS_ratio[iCP], CP.Hertzian.friction.sliding[iCP], CP.Hertzian.friction.rolling[iCP], CP.Hertzian.friction.torsion[iCP]);
	}
	I.contactForce[idx] = contactForce;
	I.contactTorque[idx] = contactTorque;
	I.slidingSpring[idx] = slidingSpring;
	I.rollingSpring[idx] = rollingSpring;
	I.torsionSpring[idx] = torsionSpring;
}

__global__ void calTriangleWallSolidSphereContactForceTorque(BasicInteraction I, 
	Sphere sph, 
	int* element2Wall,
	int* wallMaterialIndex,
	DynamicState wallState, 
	ContactParameter CP, 
	double timeStep)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= I.num) return;

	int idxA = I.objectPointed[idx];
	int idxB = I.objectPointing[idx];

	if (sph.SPHIndex[idxB] >= 0) return;//If sphere is SPH, skip

	double3 contactForce = make_double3(0., 0., 0.);
	double3 contactTorque = make_double3(0., 0., 0.);
	int iw = element2Wall[idxA];
	double inverseMassA = wallState.inverseMass[iw];
	double inverseMassB = sph.state.inverseMass[idxB];
	if (inverseMassB == 0.) return;
	double effectiveMass = 1. / (inverseMassA + inverseMassB);
	double radB = sph.radii[idxB];
	double effectiveRadius = radB;
	double3 velA = wallState.velocities[iw];
	double3 velB = sph.state.velocities[idxB];
	double3 angVelA = wallState.angularVelocities[iw];
	double3 angVelB = sph.state.angularVelocities[idxB];
	double3 posA = wallState.positions[iw];
	double3 posB = sph.state.positions[idxB];
	double normalOverlap = I.normalOverlap[idx];
	double3 contactNormal = I.contactNormal[idx];
	double3 contactPoint = I.contactPoint[idx];
	double3 slidingSpring = I.slidingSpring[idx];
	double3 rollingSpring = I.rollingSpring[idx];
	double3 torsionSpring = I.torsionSpring[idx];
	double3 relativeVelocityAtContact = velA + cross(angVelA, contactPoint - posA) - (velB + cross(angVelB, contactPoint - posB));
	double3 relativeAngularVelocityAtContact = angVelA - angVelB;
	int iMW = wallMaterialIndex[iw];
	int iMB = sph.materialIndex[idxB];
	int iCP = CP.getContactParameterIndex(iMW, iMB);
	double E1 = CP.material.elasticModulus[iMW];
	double E2 = CP.material.elasticModulus[iMB];
	if (CP.Linear.stiffness.normal[iCP] > 0.)
	{
		LinearContact(contactForce, contactTorque, slidingSpring, rollingSpring, torsionSpring, relativeVelocityAtContact, relativeAngularVelocityAtContact, contactNormal, normalOverlap, effectiveMass, effectiveRadius, timeStep, CP.Linear.stiffness.normal[iCP], CP.Linear.stiffness.sliding[iCP], CP.Linear.stiffness.rolling[iCP], CP.Linear.stiffness.torsion[iCP], CP.Linear.dissipation.normal[iCP], CP.Linear.dissipation.sliding[iCP], CP.Linear.dissipation.rolling[iCP], CP.Linear.dissipation.torsion[iCP], CP.Linear.friction.sliding[iCP], CP.Linear.friction.rolling[iCP], CP.Linear.friction.torsion[iCP]);
	}
	else if (E1 != 0. && E2 != 0.)
	{
		double v1 = CP.material.poissonRatio[iMW];
		double v2 = CP.material.poissonRatio[iMB];
		double effElasticModulus = 0, effShearModulus = 0;
		effElasticModulus = 1. / (((1 - v1 * v1) / E1) + ((1 - v2 * v2) / E2));
		double G1 = E1 / (2 * (1 + v1)), G2 = E2 / (2 * (1 + v2));
		effShearModulus = 1. / ((2 - v1) / G1 + (2 - v2) / G2);
		double r = CP.Hertzian.restitution[iCP];
		double logR = log(r);
		double dissipation = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(contactForce, contactTorque, slidingSpring, rollingSpring, torsionSpring, relativeVelocityAtContact, relativeAngularVelocityAtContact, contactNormal, normalOverlap, effectiveMass, effectiveRadius, timeStep, effElasticModulus, effShearModulus, dissipation, CP.Hertzian.kR_to_kS_ratio[iCP], CP.Hertzian.kT_to_kS_ratio[iCP], CP.Hertzian.friction.sliding[iCP], CP.Hertzian.friction.rolling[iCP], CP.Hertzian.friction.torsion[iCP]);
	}
	I.contactForce[idx] = contactForce;
	I.contactTorque[idx] = contactTorque;
	I.slidingSpring[idx] = slidingSpring;
	I.rollingSpring[idx] = rollingSpring;
	I.torsionSpring[idx] = torsionSpring;
}

__global__ void calSphereSphereBondedForceTorque(BondedInteraction I,
	BasicInteraction sphSphI,
	Sphere sph,
	ContactParameter CP,
	double timeStep)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= I.num) return;
	if (I.isBonded[idx] == 0)
	{
		I.normalForce[idx] = 0;
		I.torsionTorque[idx] = 0;
		I.shearForce[idx] = make_double3(0, 0, 0);
		I.bendingTorque[idx] = make_double3(0, 0, 0);
		return;
	}

	int idxA = I.objectPointed[idx];
	int idxB = I.objectPointing[idx];

	bool find = false;
	double3 contactNormal = make_double3(0., 0., 0.), contactPoint = make_double3(0., 0., 0.);
	int iContactPair = -1;
	int neighborStartA = idxA > 0 ? sph.neighbor.prefixSum[idxA - 1] : 0;
	int neighborEndA = sph.neighbor.prefixSum[idxA];
	for (int j = neighborStartA; j < neighborEndA; j++)
	{
		if (sphSphI.objectPointing[j] == idxB)
		{
			contactNormal = sphSphI.contactNormal[j];
			contactPoint = sphSphI.contactPoint[j];
			iContactPair = j;
			find = true;
			break;
		}
	}
	if (!find)
	{
		I.isBonded[idx] = 0;
		return;
	}

	double radA = sph.radii[idxA];
	double radB = sph.radii[idxB];
	double3 velA = sph.state.velocities[idxA];
	double3 velB = sph.state.velocities[idxB];
	double3 angVelA = sph.state.angularVelocities[idxA];
	double3 angVelB = sph.state.angularVelocities[idxB];
	double3 posA = sph.state.positions[idxA];
	double3 posB = sph.state.positions[idxB];
	double3 contactNormalPrev = I.contactNormal[idx];
	double3 relativeVelocityAtContact = velA + cross(angVelA, contactPoint - posA) - (velB + cross(angVelB, contactPoint - posB));
	int iCP = CP.getContactParameterIndex(sph.materialIndex[idxA], sph.materialIndex[idxB]);
	double nF = I.normalForce[idx], tT = I.torsionTorque[idx];
	double3 sF = I.shearForce[idx], bT = I.bendingTorque[idx];

	I.isBonded[idx] = ParallelBondedContact(nF, tT, sF, bT, contactNormalPrev, contactNormal, relativeVelocityAtContact, angVelA, angVelB, radA, radB, timeStep, CP.Bond.multiplier[iCP], CP.Bond.elasticModulus[iCP], CP.Bond.kN_to_kS_ratio[iCP], CP.Bond.tensileStrength[iCP], CP.Bond.cohesion[iCP], CP.Bond.frictionCoeff[iCP]);

	I.normalForce[idx] = nF;
	I.torsionTorque[idx] = tT;
	I.shearForce[idx] = sF;
	I.bendingTorque[idx] = bT;
	I.contactNormal[idx] = contactNormal;
	I.contactPoint[idx] = contactPoint;

	//There is a one to one relationship between I(bonded) and sphSphI, and atomicAdd can be disregard
	sphSphI.contactForce[iContactPair] += nF * contactNormal + sF;
	sphSphI.contactTorque[iContactPair] += tT * contactNormal + bT;
}

__global__ void updateSPHDensity(SPH SPHP,
	Sphere sph,
	BasicInteraction sphSphI,
	double timeStep)
{
	int idxA = blockIdx.x * blockDim.x + threadIdx.x;
	if (idxA >= sph.num) return;
	int SPHA = sph.SPHIndex[idxA];
	if (SPHA < 0) return;
	double3 posA = sph.state.positions[idxA];
	double3 velA = sph.state.velocities[idxA];
	double dRhoA = 0;
	for (int i = idxA > 0 ? sph.neighbor.prefixSum[idxA - 1] : 0; i < sph.neighbor.prefixSum[idxA]; i++)
	{
		int idxB = sphSphI.objectPointing[i];
		int SPHB = sph.SPHIndex[idxB];
		if (SPHB >= 0)
		{
			double massB = 1 / sph.state.inverseMass[idxB];
			double3 posB = sph.state.positions[idxB];
			double3 velB = sph.state.velocities[idxB];
			double h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
			double3 relDist = posA - posB;
			double3 relVel = velA - velB;
			double3 gradW = gradCubicSplineKernel3D(relDist, h);
			dRhoA += massB * dot(relVel, gradW);
		}
	}

	if (sph.sphereRange.start[idxA] != 0xFFFFFFFF)
	{
		for (int i = sph.sphereRange.start[idxA]; i < sph.sphereRange.end[idxA]; i++)
		{
			int j = sphSphI.hash.index[i];
			int idxB = sphSphI.objectPointed[j];//careful, idxA is the sphSphI.objectPointing(the greater index)
			int SPHB = sph.SPHIndex[idxB];
			if (SPHB >= 0)
			{
				double massB = 1 / sph.state.inverseMass[idxB];
				double3 posB = sph.state.positions[idxB];
				double3 velB = sph.state.velocities[idxB];
				double h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
				double3 relDist = posA - posB;
				double3 relVel = velA - velB;
				double3 gradW = gradCubicSplineKernel3D(relDist, h);
				dRhoA += massB * dot(relVel, gradW);
			}
		}
	}
	SPHP.density[SPHA] += dRhoA * timeStep;
}

__global__ void calSPHSPSStress(SPH SPHP,
	Sphere sph,
	BasicInteraction sphSphI)
{
	int idxA = blockIdx.x * blockDim.x + threadIdx.x;
	if (idxA >= sph.num) return;
	int SPHA = sph.SPHIndex[idxA];
	if (SPHA < 0) return;
	double3 posA = sph.state.positions[idxA];
	double3 velA = sph.state.velocities[idxA];
	symMatrix strainRateA = make_symMatrix(0., 0., 0., 0., 0., 0.);
	symMatrix stressA = make_symMatrix(0., 0., 0., 0., 0., 0.);
	double hAll = 0;
	int nPairs = 0;
	for (int i = idxA > 0 ? sph.neighbor.prefixSum[idxA - 1] : 0; i < sph.neighbor.prefixSum[idxA]; i++)
	{
		int idxB = sphSphI.objectPointing[i];
		int SPHB = sph.SPHIndex[idxB];
		if (SPHB >= 0)
		{
			double volB = 1. / sph.state.inverseMass[idxB] / SPHP.density[SPHB];
			double3 posB = sph.state.positions[idxB];
			double3 velB = sph.state.velocities[idxB];
			double h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
			hAll += h;
			nPairs++;
			double3 relDist = posA - posB;
			double3 relVel = velA - velB;
			double3 gradW = gradCubicSplineKernel3D(relDist, h);
			strainRateA.xx += (-relVel.x * gradW.x * volB);
			strainRateA.yy += (-relVel.y * gradW.y * volB);
			strainRateA.zz += (-relVel.z * gradW.z * volB);
			strainRateA.xy += (-0.5 * (relVel.x * gradW.y + relVel.y * gradW.x) * volB);
			strainRateA.xz += (-0.5 * (relVel.x * gradW.z + relVel.z * gradW.x) * volB);
			strainRateA.yz += (-0.5 * (relVel.y * gradW.z + relVel.z * gradW.y) * volB);
		}
	}
	if (sph.sphereRange.start[idxA] != 0xFFFFFFFF)
	{
		for (int i = sph.sphereRange.start[idxA]; i < sph.sphereRange.end[idxA]; i++)
		{
			int j = sphSphI.hash.index[i];
			int idxB = sphSphI.objectPointed[j];//careful, idxA is the sphSphI.objectPointing(the greater index)
			int SPHB = sph.SPHIndex[idxB];
			if (SPHB >= 0)
			{
				double volB = 1. / sph.state.inverseMass[idxB] / SPHP.density[SPHB];
				double3 posB = sph.state.positions[idxB];
				double3 velB = sph.state.velocities[idxB];
				double h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
				hAll += h;
				nPairs++;
				double3 relDist = posA - posB;
				double3 relVel = velA - velB;
				double3 gradW = gradCubicSplineKernel3D(relDist, h);
				strainRateA.xx += (-relVel.x * gradW.x * volB);
				strainRateA.yy += (-relVel.y * gradW.y * volB);
				strainRateA.zz += (-relVel.z * gradW.z * volB);
				strainRateA.xy += (-0.5 * (relVel.x * gradW.y + relVel.y * gradW.x) * volB);
				strainRateA.xz += (-0.5 * (relVel.x * gradW.z + relVel.z * gradW.x) * volB);
				strainRateA.yz += (-0.5 * (relVel.y * gradW.z + relVel.z * gradW.y) * volB);
			}
		}
	}
	double Cs = 0.1;
	double S_norm = norm(strainRateA);
	double deltaL = 0.;
	if (nPairs > 0) deltaL = hAll / double(nPairs);
	double vt = pow(Cs * deltaL, 2) * S_norm;
	symMatrix term1 = (2. * vt) * deviatoric(strainRateA);
	double Ci = 6.6e-3;
	double scalar = (2.0 / 3.0) * Ci * deltaL * deltaL * S_norm * S_norm;
	symMatrix term2 = make_symMatrix(scalar, scalar, scalar, 0., 0., 0.);
	stressA = stressA + (term1 - term2);
	SPHP.SPSStress[SPHA] = stressA * SPHP.density[SPHA];
}

__global__ void calSPHEffectiveVolume(SPH SPHP,
	Sphere sph,
	BasicInteraction sphSphI)
{
	int idxA = blockIdx.x * blockDim.x + threadIdx.x;
	if (idxA >= sph.num) return;
	int SPHA = sph.SPHIndex[idxA];
	if (SPHA < 0) return;
	double h = sph.radii[idxA];//smooth Length
	double3 posA = sph.state.positions[idxA];
	double massA = 1. / sph.state.inverseMass[idxA];
	double rhoA = SPHP.density[SPHA];
	double effectiveVolumeA = massA * cubicSplineKernel3D(0, h) / rhoA;
	for (int i = idxA > 0 ? sph.neighbor.prefixSum[idxA - 1] : 0; i < sph.neighbor.prefixSum[idxA]; i++)
	{
		int idxB = sphSphI.objectPointing[i];
		int SPHB = sph.SPHIndex[idxB];
		if (SPHB >= 0)
		{
			double rhoB = SPHP.density[SPHB];
			if (rhoB <= 0) continue;
			h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
			double3 posB = sph.state.positions[idxB];
			double massB = 1 / sph.state.inverseMass[idxB];
			double dis = length(posA - posB);
			effectiveVolumeA += massB * cubicSplineKernel3D(dis, h) / rhoB;
		}
	}

	if (sph.sphereRange.start[idxA] != 0xFFFFFFFF)
	{
		for (int i = sph.sphereRange.start[idxA]; i < sph.sphereRange.end[idxA]; i++)
		{
			int j = sphSphI.hash.index[i];
			int idxB = sphSphI.objectPointed[j];//careful, idxA is the sphSphI.objectPointing(the greater index)
			int SPHB = sph.SPHIndex[idxB];
			if (SPHB >= 0)
			{
				double rhoB = SPHP.density[SPHB];
				if (rhoB <= 0) continue;
				h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
				double3 posB = sph.state.positions[idxB];
				double massB = 1 / sph.state.inverseMass[idxB];
				double dis = length(posA - posB);
				effectiveVolumeA += massB * cubicSplineKernel3D(dis, h) / rhoB;
			}
		}
	}
	SPHP.effectiveVolume[SPHA] = effectiveVolumeA;
}

__global__ void calFilteringSPHDensity(SPH SPHP,
	Sphere sph,
	BasicInteraction sphSphI)
{
	int idxA = blockIdx.x * blockDim.x + threadIdx.x;
	if (idxA >= sph.num) return;
	int SPHA = sph.SPHIndex[idxA];
	if (SPHA < 0) return;
	double effectiveVolumeA = SPHP.effectiveVolume[SPHA];
	if (effectiveVolumeA <= 0) return;

	double h = sph.radii[idxA];//smooth Length
	double3 posA = sph.state.positions[idxA];
	double massA = 1. / sph.state.inverseMass[idxA];
	double rhoA = massA * cubicSplineKernel3D(0, h);
	for (int i = idxA > 0 ? sph.neighbor.prefixSum[idxA - 1] : 0; i < sph.neighbor.prefixSum[idxA]; i++)
	{
		int idxB = sphSphI.objectPointing[i];
		int SPHB = sph.SPHIndex[idxB];
		if (SPHB >= 0)
		{
			h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
			double3 posB = sph.state.positions[idxB];
			double massB = 1 / sph.state.inverseMass[idxB];
			double dis = length(posA - posB);
			rhoA += massB * cubicSplineKernel3D(dis, h);
		}
	}

	if (sph.sphereRange.start[idxA] != 0xFFFFFFFF)
	{
		for (int i = sph.sphereRange.start[idxA]; i < sph.sphereRange.end[idxA]; i++)
		{
			int j = sphSphI.hash.index[i];
			int idxB = sphSphI.objectPointed[j];//careful, idxA is the sphSphI.objectPointing(the greater index)
			int SPHB = sph.SPHIndex[idxB];
			if (SPHB >= 0)
			{
				h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
				double3 posB = sph.state.positions[idxB];
				double massB = 1 / sph.state.inverseMass[idxB];
				double dis = length(posA - posB);
				rhoA += massB * cubicSplineKernel3D(dis, h);
			}
		}
	}

	SPHP.density[SPHA] = rhoA / effectiveVolumeA;
}

__global__ void calSPHPressure(SPH SPHP)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= SPHP.num) return;
	double rho = SPHP.density[idx];
	double gamma = 7;
	double rho0 = SPHP.density0;
	double kGas = rho0 * pow(SPHP.c0, 2) / gamma;
	SPHP.pressure[idx] = kGas * (pow(rho / rho0, gamma) - 1.0);
}

__global__ void calXSPHVelocityCorrection(SPH SPHP,
	Sphere sph,
	BasicInteraction sphSphI)
{
	int idxA = blockIdx.x * blockDim.x + threadIdx.x;
	if (idxA >= sph.num) return;
	int SPHA = sph.SPHIndex[idxA];
	if (SPHA < 0) return;
	double3 posA = sph.state.positions[idxA];
	double3 velA = sph.state.velocities[idxA];
	double3 dVelA = make_double3(0, 0, 0);
	for (int i = idxA > 0 ? sph.neighbor.prefixSum[idxA - 1] : 0; i < sph.neighbor.prefixSum[idxA]; i++)
	{
		int idxB = sphSphI.objectPointing[i];
		int SPHB = sph.SPHIndex[idxB];
		if (SPHB >= 0)
		{
			double massB = 1 / sph.state.inverseMass[idxB];
			double3 posB = sph.state.positions[idxB];
			double3 velB = sph.state.velocities[idxB];
			double h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
			double3 relDist = posA - posB;
			double3 relVel = velA - velB;
			double W = cubicSplineKernel3D(length(relDist), h);
			double rhoBar = 0.5 * (SPHP.density[SPHA] + SPHP.density[SPHB]);
			dVelA += massB * relVel * W / rhoBar;
		}
	}

	if (sph.sphereRange.start[idxA] != 0xFFFFFFFF)
	{
		for (int i = sph.sphereRange.start[idxA]; i < sph.sphereRange.end[idxA]; i++)
		{
			int j = sphSphI.hash.index[i];
			int idxB = sphSphI.objectPointed[j];//careful, idxA is the sphSphI.objectPointing(the greater index)
			int SPHB = sph.SPHIndex[idxB];
			if (SPHB >= 0)
			{
				double massB = 1 / sph.state.inverseMass[idxB];
				double3 posB = sph.state.positions[idxB];
				double3 velB = sph.state.velocities[idxB];
				double h = 0.5 * (sph.radii[idxA] + sph.radii[idxB]);
				double3 relDist = posA - posB;
				double3 relVel = velA - velB;
				double W = cubicSplineKernel3D(length(relDist), h);
				double rhoBar = 0.5 * (SPHP.density[SPHA] + SPHP.density[SPHB]);
				dVelA += massB * relVel * W / rhoBar;
			}
		}
	}

	SPHP.XSPHVariant[SPHA] = 0.5 * dVelA;//free factor = 0.5
}

__global__ void calSPHForce(BasicInteraction I,
	SPH SPHP, 
	Sphere sph)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= I.num) return;

	int idxA = I.objectPointed[idx];
	int idxB = I.objectPointing[idx];
	int SPHA = sph.SPHIndex[idxA];
	int SPHB = sph.SPHIndex[idxB];
	if (SPHA >= 0 && SPHB >= 0)
	{
		double radA = sph.radii[idxA];
		double radB = sph.radii[idxB];
		double3 posA = sph.state.positions[idxA];
		double3 posB = sph.state.positions[idxB];

		//Update contact information of sphere-sphere interactions (The contact information is updated in Neighbor Search every 30 steps)
		double normalOverlap = radA + radB - length(posA - posB);
		double3 contactNormal = normalize(posA - posB);
		double3 contactPoint = posB + (radB - 0.5 * normalOverlap) * contactNormal;
		I.normalOverlap[idx] = normalOverlap;
		I.contactNormal[idx] = contactNormal;
		I.contactPoint[idx] = contactPoint;

		double3 velA = sph.state.velocities[idxA];
		double3 velB = sph.state.velocities[idxB];
		double massA = 1. / sph.state.inverseMass[idxA];
		double massB = 1. / sph.state.inverseMass[idxB];
		double pressureA = SPHP.pressure[SPHA];
		double pressureB = SPHP.pressure[SPHB];
		double rhoA = SPHP.density[SPHA];
		double rhoB = SPHP.density[SPHB];
		double h = 0.5 * (radA + radB);//smooth Length

		double3 pressureForce = make_double3(0, 0, 0);
		double3 viscosityForce = make_double3(0., 0., 0.);
		double3 SPSForce = make_double3(0., 0., 0.);
		double3 relativeDist = posA - posB;
		double pressureTerm = pressureA / (pow(rhoA, 2)) + pressureB / (pow(rhoB, 2));
		double RAB = (pressureA > 0 && pressureB > 0.) ?
			0.01 * pressureTerm :
			0.2 * (abs(pressureA) / (pow(rhoA, 2)) + abs(pressureB) / (pow(rhoB, 2)));
		double WAB = cubicSplineKernel3D(length(relativeDist), h);
		double deltaS = cbrt(0.5 * (massA + massB) / SPHP.density0);
		double WS = cubicSplineKernel3D(deltaS, h);
		double fAB = WAB / WS;
		double tensileCorrectionTerm = RAB * pow(fAB, 4);
		double3 gradW = gradCubicSplineKernel3D(relativeDist, h);
		pressureForce = -massB * (pressureTerm + tensileCorrectionTerm) * gradW;
		pressureForce *= massA;

		double3 relativeVel = velA - velB;
		if (dot(relativeVel, relativeDist) < 0.)
		{
			double dis = length(relativeDist);
			double muAB = h * dot(relativeVel, relativeDist) / (dis * dis + 0.01 * h * h);
			double cAB = SPHP.c0;
			double rhoBar = 0.5 * (rhoA + rhoB);
			double PiAB = (-SPHP.alpha * cAB * muAB + SPHP.beta * muAB * muAB) / rhoBar;
			viscosityForce = -massB * PiAB * gradW;
			viscosityForce *= massA;
		}

		SPSForce = massB * ((SPHP.SPSStress[SPHA] * (1. / (rhoA * rhoA)) + SPHP.SPSStress[SPHB] * (1. / (rhoB * rhoB))) * gradW);
		SPSForce *= massA;

		I.contactForce[idx] = pressureForce + viscosityForce + SPSForce;
	}
}

__global__ void calDEMSPHForce(BasicInteraction I,
	SPH SPHP,
	Sphere sph)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= I.num) return;

	int idxA = I.objectPointed[idx];
	int idxB = I.objectPointing[idx];
	int SPHA = sph.SPHIndex[idxA];
	int SPHB = sph.SPHIndex[idxB];
	double3 posA = sph.state.positions[idxA];
	double3 posB = sph.state.positions[idxB];
	double radA = sph.radii[idxA];
	double radB = sph.radii[idxB];
	//Update contact information of sphere-sphere interactions (The contact information is updated in Neighbor Search every 30 steps)
	double normalOverlap = radA + radB - length(posA - posB);
	double3 contactNormal = normalize(posA - posB);
	double3 contactPoint = posB + (radB - 0.5 * normalOverlap) * contactNormal;
	I.normalOverlap[idx] = normalOverlap;
	I.contactNormal[idx] = contactNormal;
	I.contactPoint[idx] = contactPoint;
	double h = 0.;
	double q = 0.;
	double relZ = 0.;
	if (SPHA >= 0 && SPHB < 0)
	{
		I.contactForce[idx] = make_double3(0, 0, 0);
		h = sph.radii[idxA];
		double dis = length(posA - contactPoint);
		q = dis / h;
		relZ = posA.z - SPHP.z0;
	}
	else if (SPHA < 0 && SPHB >= 0)
	{
		I.contactForce[idx] = make_double3(0, 0, 0);
		h = sph.radii[idxB];
		double dis = length(posB - contactPoint);
		q = dis / h;
		relZ = posB.z - SPHP.z0;
	}
	else
	{
		return;
	}

	if (q > 0. && q <= 1.)
	{
		double u = dot(sph.state.velocities[idxA] - sph.state.velocities[idxB], contactNormal);
		double A = 0.01 * pow(SPHP.c0, 2) / h;
		//if (u < 0) A += SPHP.c0 * (-u) / h;
		double Rd = A * (1 - q) / sqrt(q);
		double absH = abs(relZ / SPHP.H0);
		double epsilon = relZ >= 0 ? 0.02 : absH + 0.02;
		if (absH > 1) epsilon = 1;
		if (u < 0) epsilon += abs(20 * u) >= SPHP.c0 ? 1 : abs(20 * u) / SPHP.c0;
		I.contactForce[idx] = contactNormal * Rd * epsilon;
	}
}

__global__ void calTriangleWallSPHForce(BasicInteraction I,
	SPH SPHP,
	Sphere sph,
	int* element2Wall,
	DynamicState wallState)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= I.num) return;

	int idxB = I.objectPointing[idx];
	int SPHB = sph.SPHIndex[idxB];
	if (SPHB < 0) return;

	double3 posB = sph.state.positions[idxB];
	double h = sph.radii[idxB];
	double3 contactPoint = I.contactPoint[idx];
	double dis = length(posB - contactPoint);
	double q = dis / h;
	if (q > 0. && q <= 1.)
	{
		int iw = element2Wall[I.objectPointed[idx]];
		double3 velW = wallState.velocities[iw] + cross(contactPoint - wallState.positions[iw], wallState.angularVelocities[iw]);
		double3 contactNormal = normalize(contactPoint - posB);
		double u = dot(velW - sph.state.velocities[idxB], contactNormal);
		double A = 0.01 * pow(SPHP.c0, 2) / h;
		//if (u < 0) A += SPHP.c0 * (- u) / h;
		double Rd = A * (1 - q) / sqrt(q);
		double relZ = posB.z - SPHP.z0;
		double absH = abs(relZ / SPHP.H0);
		double epsilon = relZ >= 0 ? 0.02 : absH + 0.02;
		if (absH > 1) epsilon = 1;
		if (u < 0) epsilon += abs(20 * u) >= SPHP.c0 ? 1 : abs(20 * u) / SPHP.c0;
		I.contactForce[idx] = contactNormal * Rd * epsilon;
	}
}

__global__ void addBoundaryForceTorque2DEM(Sphere sph,
	ContactParameter CP,
	BoundaryWall wall,
	double3 minBound,
	double3 maxBound,
	double3 coordinateAxisDirection,
	double timeStep)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= sph.num) return;
	if (sph.SPHIndex[idx] >= 0.) return;

	double3 pos = sph.state.positions[idx];
	double r = sph.radii[idx];
	double3 slidingSpring = wall.slidingSpring[idx];
	double3 rollingSpring = wall.rollingSpring[idx];
	double3 torsionSpring = wall.torsionSpring[idx];

	double3 contactNormal = -coordinateAxisDirection;
	double normalOverlap = r - dot(minBound - pos, contactNormal);
	double3 contactPoint = pos + contactNormal * dot(minBound - pos, contactNormal);

	if (normalOverlap <= 0.)
	{
		contactNormal = coordinateAxisDirection;
		normalOverlap = r - dot(maxBound - pos, contactNormal);
		contactPoint = pos + contactNormal * dot(maxBound - pos, contactNormal);
	}
	if (normalOverlap <= 0.)
	{
		wall.slidingSpring[idx] = make_double3(0., 0., 0.);
		wall.rollingSpring[idx] = make_double3(0., 0., 0.);
		wall.torsionSpring[idx] = make_double3(0., 0., 0.);
		return;
	}
	
	double3 contactForce = make_double3(0., 0., 0.);
	double3 contactTorque = make_double3(0., 0., 0.);
	double3 relativeVelocityAtContact = -(sph.state.velocities[idx] + cross(sph.state.angularVelocities[idx], contactPoint - pos));
	double3 relativeAngularVelocityAtContact = -sph.state.angularVelocities[idx];
	double effectiveMass = 1. / sph.state.inverseMass[idx];
	double effectiveRadius = r;
	int iMA = wall.materialIndex;
	int iMB = sph.materialIndex[idx];
	int iCP = CP.getContactParameterIndex(iMA, iMB);
	double E1 = CP.material.elasticModulus[iMA];
	double E2 = CP.material.elasticModulus[iMB];
	if (CP.Linear.stiffness.normal[iCP] > 0.)
	{
		LinearContact(contactForce, contactTorque, slidingSpring, rollingSpring, torsionSpring, relativeVelocityAtContact, relativeAngularVelocityAtContact, contactNormal, normalOverlap, effectiveMass, effectiveRadius, timeStep, CP.Linear.stiffness.normal[iCP], CP.Linear.stiffness.sliding[iCP], CP.Linear.stiffness.rolling[iCP], CP.Linear.stiffness.torsion[iCP], CP.Linear.dissipation.normal[iCP], CP.Linear.dissipation.sliding[iCP], CP.Linear.dissipation.rolling[iCP], CP.Linear.dissipation.torsion[iCP], CP.Linear.friction.sliding[iCP], CP.Linear.friction.rolling[iCP], CP.Linear.friction.torsion[iCP]);
	}
	else if (E1 != 0. && E2 != 0.)
	{
		double v1 = CP.material.poissonRatio[iMA];
		double v2 = CP.material.poissonRatio[iMB];
		double effElasticModulus = 0, effShearModulus = 0;
		effElasticModulus = 1. / (((1 - v1 * v1) / E1) + ((1 - v2 * v2) / E2));
		double G1 = E1 / (2 * (1 + v1)), G2 = E2 / (2 * (1 + v2));
		effShearModulus = 1. / ((2 - v1) / G1 + (2 - v2) / G2);
		double r = CP.Hertzian.restitution[iCP];
		double logR = log(r);
		double dissipation = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(contactForce, contactTorque, slidingSpring, rollingSpring, torsionSpring, relativeVelocityAtContact, relativeAngularVelocityAtContact, contactNormal, normalOverlap, effectiveMass, effectiveRadius, timeStep, effElasticModulus, effShearModulus, dissipation, CP.Hertzian.kR_to_kS_ratio[iCP], CP.Hertzian.kT_to_kS_ratio[iCP], CP.Hertzian.friction.sliding[iCP], CP.Hertzian.friction.rolling[iCP], CP.Hertzian.friction.torsion[iCP]);
	}
	wall.slidingSpring[idx] = slidingSpring;
	wall.rollingSpring[idx] = rollingSpring;
	wall.torsionSpring[idx] = torsionSpring;

	sph.state.forces[idx] -= contactForce;
	sph.state.torques[idx] -= contactTorque;
	sph.state.torques[idx] += cross(contactPoint - pos, -contactForce);
}

__global__ void addBoundaryForceTorque2SPH(Sphere sph,
	SPH SPHP,
	double3 minBound,
	double3 maxBound,
	double3 coordinateAxisDirection)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= sph.num) return;
	if (sph.SPHIndex[idx] < 0.) return;

	double3 pos = sph.state.positions[idx];
	double r = sph.radii[idx];
	
	double3 contactNormal = -coordinateAxisDirection;
	double normalOverlap = r - dot(minBound - pos, contactNormal);
	double3 contactPoint = pos + contactNormal * dot(minBound - pos, contactNormal);

	if (normalOverlap <= 0)
	{
		contactNormal = coordinateAxisDirection;
		normalOverlap = r - dot(maxBound - pos, contactNormal);
		contactPoint = pos + contactNormal * dot(maxBound - pos, contactNormal);
	}
	if (normalOverlap <= 0) return;

	double3 contactForce = make_double3(0., 0., 0.);
	double h = r;
	double dis = length(pos - contactPoint);
	double q = dis / h;
	if (q > 0. && q <= 1.)
	{
		double u = dot(-sph.state.velocities[idx], contactNormal);
		double A = 0.01 * pow(SPHP.c0, 2) / h;
		double Rd = A * (1 - q) / sqrt(q);
		double relZ = pos.z - SPHP.z0;
		double absH = abs(relZ / SPHP.H0);
		double epsilon = relZ >= 0 ? 0.02 : absH + 0.02;
		if (absH > 1) epsilon = 1;
		if (u < 0) epsilon += abs(20 * u) >= SPHP.c0 ? 1 : abs(20 * u) / SPHP.c0;
		contactForce = contactNormal * Rd * epsilon;
	}
	sph.state.forces[idx] -= contactForce;
}

__global__ void accumulateSphereForceTorqueA(Sphere sph,
	BasicInteraction I)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= sph.num) return;

	sph.state.forces[idx] = make_double3(0, 0, 0);
	sph.state.torques[idx] = make_double3(0, 0, 0);
	for (int i = idx > 0 ? sph.neighbor.prefixSum[idx - 1] : 0; i < sph.neighbor.prefixSum[idx]; i++)
	{
		double3 forceA = I.contactForce[i];
		double3 torqueA = I.contactTorque[i];
		torqueA += cross(I.contactPoint[i] - sph.state.positions[idx], forceA);
		sph.state.forces[idx] += forceA;
		sph.state.torques[idx] += torqueA;
	}
}

__global__ void accumulateSphereForceTorqueB(double3* forces, double3* torques,
	double3* positions,
	int* iRangeStart,
	int* iRangeEnd,
	BasicInteraction I,
	int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	if (iRangeStart[idx] == 0xFFFFFFFF) return;
	for (int i = iRangeStart[idx]; i < iRangeEnd[idx]; i++)
	{
		int j = I.hash.index[i];
		double3 forceB = -I.contactForce[j];
		double3 torqueB = -I.contactTorque[j];
		torqueB += cross(I.contactPoint[j] - positions[idx], forceB);
		forces[idx] += forceB;
		torques[idx] += torqueB;
	}
}

__global__ void clearTriangleWallForceTorque(double3* forces, double3* torques,
	int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;
	forces[idx] = make_double3(0., 0., 0.);
	torques[idx] = make_double3(0., 0., 0.);
}

__global__ void accumulateTriangleWallForceTorque(double3* forces, double3* torques,
	double3* positions,
	int* neighborPrefixSum,
	int* element2Wall,
	BasicInteraction I,
	int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	int idw = element2Wall[idx];
	double3 forceA = make_double3(0., 0., 0.);
	double3 torqueA = make_double3(0., 0., 0.);
	for (int i = idx > 0 ? neighborPrefixSum[idx - 1] : 0; i < neighborPrefixSum[idx]; i++)
	{
		forceA += I.contactForce[i];
		torqueA += I.contactTorque[i];
		torqueA += cross(I.contactPoint[i] - positions[idw], forceA);
	}
	atomicAddDouble3(forces, idw, forceA);
	atomicAddDouble3(torques, idw, torqueA);
}

void SPHDensityShephardFilter(DeviceData& d, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	calSPHEffectiveVolume << <grid, block >> > (d.SPHParticles,
		d.spheres,
		d.sphSphInteract);
	//cudaDeviceSynchronize();
	calFilteringSPHDensity << <grid, block >> > (d.SPHParticles,
		d.spheres,
		d.sphSphInteract);
	//cudaDeviceSynchronize();
}

void sphere2SphereDEM(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	numObjects = d.sphSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calSolidSphereContactForceTorque << <grid, block >> > (d.sphSphInteract,
			d.spheres,
			d.contactPara,
			timeStep);
		//cudaDeviceSynchronize();
		if (d.sphSphBondedInteract.num > 0)
		{
			computeGPUParameter(grid, block, d.sphSphBondedInteract.num, maxThreadsPerBlock);
			calSphereSphereBondedForceTorque << <grid, block >> > (d.sphSphBondedInteract,
				d.sphSphInteract,
				d.spheres,
				d.contactPara,
				timeStep);
			//cudaDeviceSynchronize();
		}
	}
}

void sphere2SphereSPH(DeviceData& d, double timeStep, int maxThreadsPerBlock, int iStep)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	if (iStep % 30 == 0)
	{
		SPHDensityShephardFilter(d, maxThreadsPerBlock);
	}
	numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	updateSPHDensity << <grid, block >> > (d.SPHParticles,
		d.spheres,
		d.sphSphInteract,
		timeStep);
	//cudaDeviceSynchronize();
	//calXSPHVelocityCorrection << <grid, block >> > (d.SPHParticles,
	//	d.spheres,
	//	d.sphSphInteract);
	//cudaDeviceSynchronize();
	calSPHSPSStress << <grid, block >> > (d.SPHParticles,
		d.spheres,
		d.sphSphInteract);
	//cudaDeviceSynchronize();
	numObjects = d.SPHParticles.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	calSPHPressure << <grid, block >> > (d.SPHParticles);
	//cudaDeviceSynchronize();
	numObjects = d.sphSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calSPHForce << <grid, block >> > (d.sphSphInteract,
			d.SPHParticles,
			d.spheres);
		//cudaDeviceSynchronize();
	}
}

void wall2SphereDEM(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	numObjects = d.faceSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calTriangleWallSolidSphereContactForceTorque << <grid, block >> > (d.faceSphInteract,
			d.spheres,
			d.triangleWalls.face.face2Wall,
			d.triangleWalls.materialIndex,
			d.triangleWalls.state,
			d.contactPara,
			timeStep);
		//cudaDeviceSynchronize();
	}

	numObjects = d.edgeSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calTriangleWallSolidSphereContactForceTorque << <grid, block >> > (d.edgeSphInteract,
			d.spheres,
			d.triangleWalls.edge.edge2Wall,
			d.triangleWalls.materialIndex,
			d.triangleWalls.state,
			d.contactPara,
			timeStep);
		//cudaDeviceSynchronize();
	}

	numObjects = d.vertexSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calTriangleWallSolidSphereContactForceTorque << <grid, block >> > (d.vertexSphInteract,
			d.spheres,
			d.triangleWalls.vertex.vertex2Wall,
			d.triangleWalls.materialIndex,
			d.triangleWalls.state,
			d.contactPara,
			timeStep);
		//cudaDeviceSynchronize();
	}
}

void wall2SphereSPH(DeviceData& d, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	numObjects = d.faceSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calTriangleWallSPHForce << <grid, block >> > (d.faceSphInteract,
			d.SPHParticles,
			d.spheres,
			d.triangleWalls.face.face2Wall,
			d.triangleWalls.state);
		//cudaDeviceSynchronize();
	}

	numObjects = d.edgeSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calTriangleWallSPHForce << <grid, block >> > (d.edgeSphInteract,
			d.SPHParticles,
			d.spheres,
			d.triangleWalls.edge.edge2Wall,
			d.triangleWalls.state);
		//cudaDeviceSynchronize();
	}

	numObjects = d.vertexSphInteract.num;
	if (numObjects > 0)
	{
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		calTriangleWallSPHForce << <grid, block >> > (d.vertexSphInteract,
			d.SPHParticles,
			d.spheres,
			d.triangleWalls.vertex.vertex2Wall,
			d.triangleWalls.state);
		//cudaDeviceSynchronize();
	}
}

void boundary2SphereDEM(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	if (d.boundaryWallX.numSpring > 0)
	{
		addBoundaryForceTorque2DEM << <grid, block >> > (d.spheres,
			d.contactPara,
			d.boundaryWallX,
			d.spatialGrids.minBound,
			d.spatialGrids.maxBound,
			make_double3(1., 0., 0.),
			timeStep);
		//cudaDeviceSynchronize();
	}
	if (d.boundaryWallY.numSpring > 0)
	{
		addBoundaryForceTorque2DEM << <grid, block >> > (d.spheres,
			d.contactPara,
			d.boundaryWallY,
			d.spatialGrids.minBound,
			d.spatialGrids.maxBound,
			make_double3(0., 1., 0.),
			timeStep);
		//cudaDeviceSynchronize();
	}
	if (d.boundaryWallZ.numSpring > 0)
	{
		addBoundaryForceTorque2DEM << <grid, block >> > (d.spheres,
			d.contactPara,
			d.boundaryWallZ,
			d.spatialGrids.minBound,
			d.spatialGrids.maxBound,
			make_double3(0., 0., 1.),
			timeStep);
		//cudaDeviceSynchronize();
	}
}

void boundary2SphereSPH(DeviceData& d, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	if (d.boundaryWallX.numSpring > 0)
	{
		addBoundaryForceTorque2SPH << <grid, block >> > (d.spheres,
			d.SPHParticles,
			d.spatialGrids.minBound,
			d.spatialGrids.maxBound,
			make_double3(1., 0., 0.));
		//cudaDeviceSynchronize();
	}
	if (d.boundaryWallY.numSpring > 0)
	{
		addBoundaryForceTorque2SPH << <grid, block >> > (d.spheres,
			d.SPHParticles,
			d.spatialGrids.minBound,
			d.spatialGrids.maxBound,
			make_double3(0., 1., 0.));
		//cudaDeviceSynchronize();
	}
	if (d.boundaryWallZ.numSpring > 0)
	{
		addBoundaryForceTorque2SPH << <grid, block >> > (d.spheres,
			d.SPHParticles,
			d.spatialGrids.minBound,
			d.spatialGrids.maxBound,
			make_double3(0., 0., 1.));
		//cudaDeviceSynchronize();
	}
}

void calculateContactForceTorqueDEM(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	sphere2SphereDEM(d, timeStep, maxThreadsPerBlock);

	wall2SphereDEM(d, timeStep, maxThreadsPerBlock);
}

void calculateContactForceTorqueSPH(DeviceData& d, double timeStep, int maxThreadsPerBlock, int iStep)
{
	sphere2SphereSPH(d, timeStep, maxThreadsPerBlock, iStep);

	wall2SphereSPH(d, maxThreadsPerBlock);
}

void calculateContactForceTorqueDEMSPH(DeviceData& d, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.sphSphInteract.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	if (numObjects > 0)
	{
		calDEMSPHForce << <grid, block >> > (d.sphSphInteract,
			d.SPHParticles,
			d.spheres);
		//cudaDeviceSynchronize();
	}
}

void accumulateForceTorque(DeviceData& d, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.spheres.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	accumulateSphereForceTorqueA << <grid, block >> > (d.spheres,
		d.sphSphInteract);
	//cudaDeviceSynchronize();

	accumulateSphereForceTorqueB << <grid, block >> > (d.spheres.state.forces, d.spheres.state.torques,
		d.spheres.state.positions,
		d.spheres.sphereRange.start,
		d.spheres.sphereRange.end,
		d.sphSphInteract,
		d.spheres.num);
	//cudaDeviceSynchronize();

	if (d.triangleWalls.num > 0)
	{
		numObjects = d.spheres.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		accumulateSphereForceTorqueB << <grid, block >> > (d.spheres.state.forces, d.spheres.state.torques,
			d.spheres.state.positions,
			d.spheres.faceRange.start,
			d.spheres.faceRange.end,
			d.faceSphInteract,
			d.spheres.num);
		//cudaDeviceSynchronize();
		accumulateSphereForceTorqueB << <grid, block >> > (d.spheres.state.forces, d.spheres.state.torques,
			d.spheres.state.positions,
			d.spheres.edgeRange.start,
			d.spheres.edgeRange.end,
			d.edgeSphInteract,
			d.spheres.num);
		//cudaDeviceSynchronize();
		accumulateSphereForceTorqueB << <grid, block >> > (d.spheres.state.forces, d.spheres.state.torques,
			d.spheres.state.positions,
			d.spheres.vertexRange.start,
			d.spheres.vertexRange.end,
			d.vertexSphInteract,
			d.spheres.num);
		//cudaDeviceSynchronize();

		numObjects = d.triangleWalls.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		clearTriangleWallForceTorque << <grid, block >> > (d.triangleWalls.state.forces, d.triangleWalls.state.torques,
			d.triangleWalls.num);
		//cudaDeviceSynchronize();
		numObjects = d.triangleWalls.face.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		accumulateTriangleWallForceTorque << <grid, block >> > (d.triangleWalls.state.forces, d.triangleWalls.state.torques,
			d.triangleWalls.state.positions,
			d.triangleWalls.face.neighbor.prefixSum,
			d.triangleWalls.face.face2Wall,
			d.faceSphInteract,
			d.triangleWalls.face.num);
		//cudaDeviceSynchronize();

		numObjects = d.triangleWalls.edge.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		accumulateTriangleWallForceTorque << <grid, block >> > (d.triangleWalls.state.forces, d.triangleWalls.state.torques,
			d.triangleWalls.state.positions,
			d.triangleWalls.edge.neighbor.prefixSum,
			d.triangleWalls.edge.edge2Wall,
			d.edgeSphInteract,
			d.triangleWalls.edge.num);
		//cudaDeviceSynchronize();

		numObjects = d.triangleWalls.vertex.num;
		computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
		accumulateTriangleWallForceTorque << <grid, block >> > (d.triangleWalls.state.forces, d.triangleWalls.state.torques,
			d.triangleWalls.state.positions,
			d.triangleWalls.vertex.neighbor.prefixSum,
			d.triangleWalls.vertex.vertex2Wall,
			d.vertexSphInteract,
			d.triangleWalls.vertex.num);
		//cudaDeviceSynchronize();
	}
}