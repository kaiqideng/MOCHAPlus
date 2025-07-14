#include "HostDataValidator.h"

bool validateContactParameter(const HostContactParameter& cp)
{
    std::cout << "Validating ContactParameter data ......\n";

    const int Nm = cp.material.num;
    if (Nm <= 0)
    {
        std::cerr << "ContactParameter: material.num must be positive\n";
        return false;
    }

    if ((int)cp.material.elasticModulus.size() != Nm ||
        (int)cp.material.poissonRatio.size() != Nm)
        return fail("contactPara.material length");

    for (int i = 0; i < Nm; ++i)
    {
        double E = cp.material.elasticModulus[i];
        if (!(E >= 0 && isFinite(E)))                            return fail("material.elasticModulus", i);

        double v = cp.material.poissonRatio[i];
        if (!(v > -1.0 && v < 1.0 && isFinite(v)))              return fail("material.poissonRatio", i);
    }

    /* --- Hertzian --- */
    const int Nh = cp.Hertzian.num;
    if ((int)cp.Hertzian.kR_to_kS_ratio.size() != Nh ||
        (int)cp.Hertzian.kT_to_kS_ratio.size() != Nh ||
        (int)cp.Hertzian.restitution.size() != Nh)
        return fail("contactPara.Hertzian length");
    if (!checkDirectional(cp.Hertzian.friction, Nh, "Hertzian.friction"))
        return false;

    for (int i = 0; i < Nh; ++i)
    {
        double r1 = cp.Hertzian.kR_to_kS_ratio[i];
        double r2 = cp.Hertzian.kT_to_kS_ratio[i];
        double e = cp.Hertzian.restitution[i];
        if (!(r1 >= 0 && isFinite(r1)))                         return fail("Hertzian.kR_to_kS_ratio", i);
        if (!(r2 >= 0 && isFinite(r2)))                         return fail("Hertzian.kT_to_kS_ratio", i);
        if (!(e > 0 && e <= 1 && isFinite(e)))                return fail("Hertzian.restitution", i);
    }

    /* --- Linear --- */
    const int Nl = cp.Linear.num;
    if (!checkDirectional(cp.Linear.stiffness, Nl, "Linear.stiffness")) return false;
    if (!checkDirectional(cp.Linear.dissipation, Nl, "Linear.dissipation")) return false;
    if (!checkDirectional(cp.Linear.friction, Nl, "Linear.friction")) return false;

    /* --- Bonded --- */
    const int Nb = cp.Bond.num;
    if ((int)cp.Bond.maxContactGap.size() != Nb ||
        (int)cp.Bond.multiplier.size() != Nb ||
        (int)cp.Bond.elasticModulus.size() != Nb ||
        (int)cp.Bond.kN_to_kS_ratio.size() != Nb ||
        (int)cp.Bond.tensileStrength.size() != Nb ||
        (int)cp.Bond.cohesion.size() != Nb ||
        (int)cp.Bond.frictionCoeff.size() != Nb)
        return fail("contactPara.Bond length");

    for (int i = 0; i < Nb; ++i)
    {
        if (!(cp.Bond.maxContactGap[i] >= 0 && isFinite(cp.Bond.maxContactGap[i])))
            return fail("Bond.maxContactGap", i);
        if (!(cp.Bond.multiplier[i] > 0 && isFinite(cp.Bond.multiplier[i])))
            return fail("Bond.multiplier", i);
        if (!(cp.Bond.elasticModulus[i] >= 0 && isFinite(cp.Bond.elasticModulus[i])))
            return fail("Bond.elasticModulus", i);
        if (!(cp.Bond.kN_to_kS_ratio[i] > 0 && isFinite(cp.Bond.kN_to_kS_ratio[i])))
            return fail("Bond.kN_to_kS_ratio", i);
        if (!(cp.Bond.tensileStrength[i] >= 0 && isFinite(cp.Bond.tensileStrength[i])))
            return fail("Bond.tensileStrength", i);
        if (!(cp.Bond.cohesion[i] >= 0 && isFinite(cp.Bond.cohesion[i])))
            return fail("Bond.cohesion", i);
        if (!(cp.Bond.frictionCoeff[i] >= 0 && isFinite(cp.Bond.frictionCoeff[i])))
            return fail("Bond.frictionCoeff", i);
    }

    std::cout << "ContactParameter data check PASSED\n";
    return true;
}

bool validateSpatialGrid(const HostSpatialGrid& g)
{
    std::cout << "Validating SpatialGrid data ......\n";

    if (!isFinite3(g.minBound) || !isFinite3(g.maxBound) || !isFinite3(g.cellSize))
    {
        std::cerr << "Grid: NaN/Inf in bounds or cellSize\n"; return false;
    }

    if (g.cellSize.x <= 0 || g.cellSize.y <= 0 || g.cellSize.z <= 0)
    {
        std::cerr << "Grid: cellSize must be positive\n"; return false;
    }

    if (g.gridSize.x <= 0 || g.gridSize.y <= 0 || g.gridSize.z <= 0)
    {
        std::cerr << "Grid: gridSize components must be > 0\n"; return false;
    }

    if (g.minBound.x >= g.maxBound.x || g.minBound.y >= g.maxBound.y || g.minBound.z >= g.maxBound.z)
    {
        std::cerr << "Grid: minBound must be strictly less than maxBound\n"; return false;
    }

    double3 span = make_double3(g.maxBound.x - g.minBound.x,
        g.maxBound.y - g.minBound.y,
        g.maxBound.z - g.minBound.z);

    double3 expectSpan = make_double3(g.cellSize.x * g.gridSize.x,
        g.cellSize.y * g.gridSize.y,
        g.cellSize.z * g.gridSize.z);

    double eps = 1e-9;
    if (std::fabs(span.x - expectSpan.x) > eps ||
        std::fabs(span.y - expectSpan.y) > eps ||
        std::fabs(span.z - expectSpan.z) > eps)
    {
        std::cerr << "Grid: (maxBound-minBound) inconsistent with cellSize * gridSize\n"; return false;
    }

    int expectedNum = g.gridSize.x * g.gridSize.y * g.gridSize.z + 1;
    if (g.num != 0 && g.num != expectedNum)
    {
        std::cerr << "Grid: num != gridSize.x*y*z + 1\n"; return false;
    }

    std::cout << "SpatialGrid check PASSED\n";
    return true;
}

bool validateSphereData(const HostSphere& sph, double3 minBound, double3 maxBound, const int numClump, const int numSPH, int numMaterial)
{
    const int n = sph.num;
    std::cout << "Validating Sphere data ......\n";

#define SIZECHK(vec,name)  if((int)(vec).size()!=n) return fail("size mismatch: "#name);
    SIZECHK(sph.clumpIndex, clumpIndex)
        SIZECHK(sph.materialIndex, materialIndex)
        SIZECHK(sph.bondClusterIndex, bondClusterIndex)
        SIZECHK(sph.radii, radii)
        SIZECHK(sph.state.positions, positions)
        SIZECHK(sph.state.orientations, orientations)
        SIZECHK(sph.state.velocities, velocities)
        SIZECHK(sph.state.angularVelocities, angularVelocities)
        SIZECHK(sph.state.forces, forces)
        SIZECHK(sph.state.torques, torques)
        SIZECHK(sph.state.inverseMass, inverseMass)
        SIZECHK(sph.state.inverseInertia, inverseInertia)
#undef SIZECHK

        for (int i = 0;i < n;++i)
        {
            double r = sph.radii[i];
            if (!(r > 0 && std::isfinite(r)))
                return fail("invalid radius", i);

            if (!isFinite3(sph.state.positions[i]))            return fail("positions", i);
            if (sph.state.positions[i].x < minBound.x || sph.state.positions[i].y < minBound.y || sph.state.positions[i].z < minBound.z ||
                sph.state.positions[i].x > maxBound.x || sph.state.positions[i].y > maxBound.y || sph.state.positions[i].z > maxBound.z)
            {
                std::cout << "Warning: sphere " << i << " is out of boundary\n";
            }
            if (!isFinite3(sph.state.velocities[i]))           return fail("velocities", i);
            if (!isFinite3(sph.state.angularVelocities[i]))    return fail("angularVelocities", i);
            if (!isFinite3(sph.state.forces[i]))               return fail("forces", i);
            if (!isFinite3(sph.state.torques[i]))              return fail("torques", i);

            const quaternion& q = sph.state.orientations[i];
            double qn = std::sqrt(q.q0 * q.q0 + q.q1 * q.q1 + q.q2 * q.q2 + q.q3 * q.q3);
            if (!(qn > 0.99 && qn < 1.01 && std::isfinite(qn)))
                return fail("orientation not normalized", i);

            double mInv = sph.state.inverseMass[i];
            if (!(mInv >= 0 && std::isfinite(mInv)))
            {
                return fail("inverseMass", i);
            }
            else if (mInv == 0 && sph.SPHIndex[i] >= 0)
            {
                return fail("inverseMass(SPH particle)", i);
            }

            if (mInv > 0)
            {
                const symMatrix& I = sph.state.inverseInertia[i];
                if (!(I.xx > 0 && I.yy > 0 && I.zz > 0 &&
                    std::isfinite(I.xx) && std::isfinite(I.yy) && std::isfinite(I.zz)))
                    return fail("inverseInertia", i);
            }
            if (sph.materialIndex[i] < 0 || sph.materialIndex[i] >= numMaterial) return fail("materialIndex out of range", i);
            if (sph.clumpIndex[i] >= numClump) return fail("clumpIndex out of range", i);
            if (sph.SPHIndex[i] >= numSPH) return fail("SPHIndex out of range", i);
        }

    std::cout << "Sphere check PASSED\n";
    return true;
}

bool validateSPHData(const HostSPH& SPHP)
{
    const int n = SPHP.num;
    if (n == 0) return true;
    std::cout << "Validating SPH data ......\n";
#define SIZECHK(vec,name)  if((int)(vec).size()!=n) return fail("size mismatch: "#name);
    SIZECHK(SPHP.density, density)
        SIZECHK(SPHP.pressure, pressure)
#undef SIZECHK
        if (!isFinite(SPHP.z0))
        {
            std::cerr << "z0(local water level) must be finite\n"; return false;
        }
    if (!isFinite(SPHP.H0) || SPHP.H0 < 0)
    {
        std::cerr << "H0 must be finite and non-negative\n"; return false;
    }
    if (!isFinite(SPHP.density0) || SPHP.density0 <= 0)
    {
        std::cerr << "density0 must be finite and positive\n"; return false;
    }
    if (!isFinite(SPHP.alpha) || SPHP.alpha < 0)
    {
        std::cerr << "alpha(artificial viscosity coefficient) must be finite and non-negative\n"; return false;
    }
    if (!isFinite(SPHP.beta) || SPHP.beta < 0)
    {
        std::cerr << "beta(viscosity coefficient) must be finite and non-negative\n"; return false;
    }
    if (!isFinite(SPHP.c0) || SPHP.c0 <= 0)
    {
        std::cerr << "c0(sound speed) must be finite and positive\n"; return false;
    }
    for (int i = 0;i < n;++i)
    {
        if (!isFinite(SPHP.density[i]) || SPHP.density[i] < 0)
            return fail("density", i);
    }
    std::cout << "SPH data check PASSED\n";
    return true;
}

bool validateClumpData(const HostClump& clumps, int numSphere)
{
    const int n = clumps.num;
    if (n == 0) return true;
    std::cout << "Validating Clump data ......\n";
#define SIZECHK(vec,name)  if((int)(vec).size()!=n) return fail("size mismatch: "#name);
    SIZECHK(clumps.pebbleStart, pebbleStart)
        SIZECHK(clumps.pebbleEnd, pebbleEnd)
#undef SIZECHK
        for (int i = 0; i < n; ++i)
        {
            if (clumps.pebbleStart[i] < 0 || clumps.pebbleEnd[i] < 0)
                return fail("pebbleStart/End negative", i);
            if (clumps.pebbleStart[i] >= numSphere || clumps.pebbleEnd[i] >= numSphere)
                return fail("pebbleStart/End out of range", i);
        }
    std::cout << "Clump data check PASSED\n";
    return true;
}

bool validateTriangleWall(const HostTriangleWall& w, int numMaterial)
{
    const int Nw = w.num;
    if (Nw == 0) return true;
    std::cout << "Validating TriangleWall data ......\n";

    /* =============== Wall-level =============== */
#define SIZECHK(vec,name)  if((int)(vec).size()!=Nw) return fail("size mismatch: "#name);
    SIZECHK(w.materialIndex, materialIndex)
        SIZECHK(w.state.positions, positions)
        SIZECHK(w.state.orientations, orientations)
        SIZECHK(w.state.velocities, velocities)
        SIZECHK(w.state.angularVelocities, angularVelocities)
        SIZECHK(w.state.forces, forces)
        SIZECHK(w.state.torques, torques)
        SIZECHK(w.state.inverseMass, inverseMass)
        SIZECHK(w.state.inverseInertia, inverseInertia)
#undef SIZECHK

        for (int i = 0;i < Nw;++i)
        {
            if (!isFinite3(w.state.positions[i]))            return fail("positions", i);
            if (!isFinite3(w.state.velocities[i]))           return fail("velocities", i);
            if (!isFinite3(w.state.angularVelocities[i]))    return fail("angularVelocities", i);
            if (!isFinite3(w.state.forces[i]))               return fail("forces", i);
            if (!isFinite3(w.state.torques[i]))              return fail("torques", i);

            const quaternion& q = w.state.orientations[i];
            double qn = std::sqrt(q.q0 * q.q0 + q.q1 * q.q1 + q.q2 * q.q2 + q.q3 * q.q3);
            if (!(qn > 0.99 && qn < 1.01 && std::isfinite(qn)))
                return fail("orientation not normalized", i);

            double mInv = w.state.inverseMass[i];
            if (!(mInv >= 0 && std::isfinite(mInv)))
                return fail("inverseMass", i);

            if (mInv > 0)
            {
                const symMatrix& I = w.state.inverseInertia[i];
                if (!(I.xx > 0 && I.yy > 0 && I.zz > 0 &&
                    std::isfinite(I.xx) && std::isfinite(I.yy) && std::isfinite(I.zz)))
                    return fail("inverseInertia", i);
            }
            if (w.materialIndex[i] < 0 || w.materialIndex[i] >= numMaterial) return fail("materialIndex out of range", i);
        }

    /* =============== Face =============== */
    const int Nf = w.face.num;
    if (Nf <= 0) {
        std::cerr << "face number must be >0\n";
        return false;
    }
#define CHK_LEN(vec,name) if((int)(vec).size()!=Nf) return fail("face "#name" length");
    CHK_LEN(w.face.vAIndex, vAIndex)
        CHK_LEN(w.face.vBIndex, vBIndex)
        CHK_LEN(w.face.vCIndex, vCIndex)
        CHK_LEN(w.face.face2Wall, face2Wall)
#undef CHK_LEN

        for (int i = 0;i < Nf;++i)
        {
            if (w.face.vAIndex[i] < 0 || w.face.vBIndex[i] < 0 || w.face.vCIndex[i] < 0)
                return fail("negative vertex index in face", i);
            if (w.face.vAIndex[i] == w.face.vBIndex[i] || w.face.vAIndex[i] == w.face.vCIndex[i] || w.face.vBIndex[i] == w.face.vCIndex[i])
                return fail("repeated vertex index in face", i);
            if (w.face.face2Wall[i] < 0 || w.face.face2Wall[i] >= Nw)
                return fail("face2Wall out of range", i);
        }

    /* =============== Edge =============== */
    const int Ne = w.edge.num;
    if (Ne <= 0) {
        std::cerr << "edge number must be >0\n";
        return false;
    }
#define CHK_LEN_E(vec,name) if((int)(vec).size()!=Ne) return fail("edge "#name" length");
    CHK_LEN_E(w.edge.vAIndex, vAIndex)
        CHK_LEN_E(w.edge.vBIndex, vBIndex)
        CHK_LEN_E(w.edge.edge2Wall, edge2Wall)
#undef CHK_LEN_E

        if (!w.edge.facePrefixSum.empty() &&
            (int)w.edge.facePrefixSum.size() != Ne)
            return fail("edge.facePrefixSum length");

    for (int i = 0;i < Ne;++i)
    {
        if (w.edge.vAIndex[i] < 0 || w.edge.vBIndex[i] < 0)
            return fail("negative vertex index in edge", i);
        if (w.edge.vAIndex[i] == w.edge.vBIndex[i])
            return fail("repeated vertex index in edge", i);
        if (w.edge.edge2Wall[i] < 0 || w.edge.edge2Wall[i] >= Nw)
            return fail("edge2Wall out of range", i);
        if (w.edge.edge2Face[i] < 0 || w.edge.edge2Face[i] >= Nf)
            return fail("edge2Face out of range", i);
        if (w.edge.facePrefixSum[i] < 0 || w.edge.facePrefixSum[i] > w.edge.edge2Face.size())
            return fail("edge.facePrefixSum", i);
    }

    /* =============== Vertex =============== */
    const int Nv = w.vertex.num;
    if (Nv <= 0) {
        std::cerr << "vertex number must be >0\n";
        return false;
    }
#define CHK_LEN_V(vec,name) if((int)(vec).size()!=Nv) return fail("vertex "#name" length");
    CHK_LEN_V(w.vertex.positions, positions)
        CHK_LEN_V(w.vertex.vertex2Wall, vertex2Wall)
#undef CHK_LEN_V

        if (!w.vertex.facePrefixSum.empty() &&
            (int)w.vertex.facePrefixSum.size() != Nv)
            return fail("vertex.facePrefixSum length");
    if (!w.vertex.edgePrefixSum.empty() &&
        (int)w.vertex.edgePrefixSum.size() != Nv)
        return fail("vertex.edgePrefixSum length");

    for (int i = 0;i < Nv;++i)
    {
        if (!isFinite3(w.vertex.positions[i]))
            return fail("NaN/Inf in vertex.positions", i);
        if (w.vertex.vertex2Wall[i] < 0 || w.vertex.vertex2Wall[i] >= Nw)
            return fail("vertex2Wall out of range", i);
        if (w.vertex.vertex2Face[i] < 0 || w.vertex.vertex2Face[i] >= Nf)
            return fail("vertex2Face out of range", i);
        if (w.vertex.vertex2Edge[i] < 0 || w.vertex.vertex2Edge[i] >= Ne)
            return fail("vertex2Edge out of range", i);
        if (w.vertex.facePrefixSum[i] < 0 || w.vertex.facePrefixSum[i] > w.vertex.vertex2Face.size())
            return fail("vertex.facePrefixSum", i);
        if (w.vertex.edgePrefixSum[i] < 0 || w.vertex.edgePrefixSum[i] > w.vertex.vertex2Edge.size())
            return fail("vertex.edgePrefixSum", i);
    }

    std::cout << "TriangleWall check PASSED\n";
    return true;
}

bool validateSimulationParameter(const HostSimulationParameter& p, int numMaterial)
{
    std::cout << "Validating SimulationParameter data ......\n";
    if (!isFinite3(p.domainOrigin)) { std::cerr << "domainOrigin NaN/Inf\n"; return false; }
    if (!isFinite3(p.domainSize)) { std::cerr << "domainSize   NaN/Inf\n"; return false; }
    if (!isFinite3(p.gravity)) { std::cerr << "gravity      NaN/Inf\n"; return false; }

    if (p.domainSize.x <= 0 || p.domainSize.y <= 0 || p.domainSize.z <= 0)
    {
        std::cerr << "domainSize components must be >0\n"; return false;
    }

    if (p.boundaryWallMaterialIndex < 0 || p.boundaryWallMaterialIndex >= numMaterial)
    {
        std::cerr << "boundaryWallMaterialIndex out of range\n"; return false;
    }

    if (!(p.timeMax > 0 && std::isfinite(p.timeMax)))
    {
        std::cerr << "timeMax must be >0 & finite\n"; return false;
    }

    if (!(p.timeStep > 0 && std::isfinite(p.timeStep) && p.timeStep < p.timeMax))
    {
        std::cerr << "timeStep must be >0, finite, and < timeMax\n"; return false;
    }

    if (p.nPrint <= 0)
    {
        std::cerr << "nPrint must be >0\n"; return false;
    }

    if (p.maxThreadsPerBlock <= 0)
    {
        std::cerr << "maxThreadsPerBlock must be >0\n"; return false;
    }

    if (p.deviceNumber < 0)
    {
        std::cerr << "deviceNumber must be >=0\n"; return false;
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && p.deviceNumber >= deviceCount)
    {
        std::cerr << "deviceNumber out of range (have " << deviceCount << " devices)\n"; return false;
    }

    std::cout << "SimulationParameter check PASSED\n";
    return true;
}

bool validateHostData(const HostData& h)
{
    std::cout << "------- Host Data Validation -------\n";
    bool pass = false;
    pass = validateContactParameter(h.contactPara);
    if (!pass) return false;
    pass = validateSpatialGrid(h.spatialGrids);
    if (!pass) return false;
    pass = validateSphereData(h.spheres, h.spatialGrids.minBound, h.spatialGrids.maxBound, h.clumps.num, h.SPHParticles.num, h.contactPara.material.num);
    if (!pass) return false;
    pass = validateSPHData(h.SPHParticles);
    if (!pass) return false;
    pass = validateClumpData(h.clumps, h.spheres.num);
    if (!pass) return false;
    pass = validateTriangleWall(h.triangleWalls, h.contactPara.material.num);
    if (!pass) return false;
    pass = validateSimulationParameter(h.simulation, h.contactPara.material.num);
    if (!pass) return false;
    return true;
}