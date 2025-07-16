#pragma once
#include "mySymMatrix.h"
#include <vector>
#include <algorithm>
#include <unordered_map>

struct HostDynamicState
{
    std::vector<double3>  positions;
    std::vector<quaternion> orientations;
    std::vector<double3>  velocities;
    std::vector<double3>  angularVelocities;
    std::vector<double3>  forces;
    std::vector<double3>  torques;
    std::vector<double>   inverseMass;
    std::vector<symMatrix> inverseInertia;

	HostDynamicState() = default;

    explicit HostDynamicState(int n)
    {
        positions.resize(n, make_double3(0., 0., 0.));
        orientations.resize(n, quaternion{ 1.,0.,0.,0. });
        velocities.resize(n, make_double3(0., 0., 0.));
        angularVelocities.resize(n, make_double3(0., 0., 0.));
        forces.resize(n, make_double3(0., 0., 0.));
        torques.resize(n, make_double3(0., 0., 0.));
        inverseMass.resize(n, 0.);
        inverseInertia.resize(n, symMatrix{});
    }
    void insertData(HostDynamicState state)
    {
        positions.insert(positions.end(), state.positions.begin(), state.positions.end());
		orientations.insert(orientations.end(), state.orientations.begin(), state.orientations.end());
		velocities.insert(velocities.end(), state.velocities.begin(), state.velocities.end());
		angularVelocities.insert(angularVelocities.end(), state.angularVelocities.begin(), state.angularVelocities.end());
		forces.insert(forces.end(), state.forces.begin(), state.forces.end());
		torques.insert(torques.end(), state.torques.begin(), state.torques.end());
		inverseMass.insert(inverseMass.end(), state.inverseMass.begin(), state.inverseMass.end());
		inverseInertia.insert(inverseInertia.end(), state.inverseInertia.begin(), state.inverseInertia.end());
    }
};

struct HostSphere
{
    int num = 0;
    std::vector<int> clumpIndex;
    std::vector<int> materialIndex;
    std::vector<int> bondClusterIndex;
    std::vector<double> radii;
    HostDynamicState state;
    std::vector<int> SPHIndex;

	HostSphere() = default;

    explicit HostSphere(int n)
    {
        num = n;
		clumpIndex.resize(n, -1);
		materialIndex.resize(n, 0);
		bondClusterIndex.resize(n, -1);
		radii.resize(n, 0.);
		state = HostDynamicState(n);
		SPHIndex.resize(n, -1);
    }

    void insertData(HostSphere sph)
    {
		if (sph.num <= 0) return;
		clumpIndex.insert(clumpIndex.end(), sph.clumpIndex.begin(), sph.clumpIndex.end());
		materialIndex.insert(materialIndex.end(), sph.materialIndex.begin(), sph.materialIndex.end());
		bondClusterIndex.insert(bondClusterIndex.end(), sph.bondClusterIndex.begin(), sph.bondClusterIndex.end());
		radii.insert(radii.end(), sph.radii.begin(), sph.radii.end());
		state.insertData(sph.state);
		SPHIndex.insert(SPHIndex.end(), sph.SPHIndex.begin(), sph.SPHIndex.end());
		num += sph.num;
    }

    void createBlockSample(double3 origin, double3 sampleSize, double3 velocity, double density, double spacing, double sphereRadius, int bondClusterId, int materialId)
    {
        if (spacing <= 0.)
        {
            std::cerr << "Spacing must be positive.\n";
            return;
        }
        double mass = 4. / 3. * pow(sphereRadius, 3) * pi() * density;
        double inertia = 0.4 * mass * pow(sphereRadius, 2);
        int numPX = int(sampleSize.x / spacing);
        int numPY = int(sampleSize.y / spacing);
        int numPZ = int(sampleSize.z / spacing);
        int N = numPX * numPY * numPZ;
        if (N <= 0) return;
        HostSphere sph = HostSphere(N);
        int count = 0;
        for (int x = 0; x < numPX; x++)
        {
            for (int y = 0; y < numPY; y++)
            {
                for (int z = 0; z < numPZ; z++)
                {
                    double3 pos = make_double3(0, 0, 0);
                    pos.x = origin.x + (x + 0.5) * spacing;
                    pos.y = origin.y + (y + 0.5) * spacing;
                    pos.z = origin.z + (z + 0.5) * spacing;
                    sph.radii[count] = sphereRadius;
                    sph.materialIndex[count] = materialId;
                    sph.bondClusterIndex[count] = bondClusterId;
                    sph.state.positions[count] = pos;
                    sph.state.velocities[count] = velocity;
                    if (mass > 0)
                    {
                        sph.state.inverseMass[count] = 1. / mass;
                        symMatrix inertiaTensor = make_symMatrix(inertia, inertia, inertia, 0., 0., 0.);
                        sph.state.inverseInertia[count] = inverse(inertiaTensor);
                    }
                    count++;
                }
            }
        }
        insertData(sph);
    }

    void createHEXBlockSample(double3 origin, double3 sampleSize, double3 velocity, double density, double spacing, double sphereRadius, int bondClusterId, int materialId)
    {
        if (spacing <= 0.)
        {
            std::cerr << "Spacing size must be positive.\n";
			return;
        }
		double mass = 4. / 3. * pow(sphereRadius, 3) * pi() * density;
		double inertia = 0.4 * mass * pow(sphereRadius, 2);
        int numPX = int(sampleSize.x / spacing);
        int numPY = int(sampleSize.y / sqrt(3.0) / spacing * 2.0) - 1;
        int numPZ = int(sampleSize.z / sqrt(6.0) / spacing * 3.0);

        double dx = spacing;
        double dy = spacing * sqrt(3.0) / 2.0;
        double dz = spacing * sqrt(6.0) / 3.0;
		int N = numPX * numPY * numPZ;
        if (N <= 0) return;
		HostSphere sph(N);

		double SXMax = double(2. * numPX - 1 + 2) * dx / 2. + sphereRadius;
        double SYMax = double(2. * numPY - 1 + 2. / 3.) * dy / 2. + (1. / sqrt(3.) - 0.5) * dy + sphereRadius;
		double SZMax = 0.5 * (spacing - dz) + (2. * numPZ - 1.) * dz / 2. + sphereRadius;
		origin.x += (sampleSize.x > SXMax) * (sampleSize.x - SXMax) / 2.;
        origin.y += (sampleSize.y > SYMax) * (sampleSize.y - SYMax) / 2.;
        origin.z += (sampleSize.z > SZMax) * (sampleSize.z - SZMax) / 2.;
        
        int count = 0;
        for (int x = 1; x <= numPX; x++)
        {
            for (int y = 1; y <= numPY; y++)
            {
                for (int z = 1; z <= numPZ; z++)
                {
                    double3 pos = make_double3(0, 0, 0);
                    pos.x = origin.x + double(2. * x - 1 + y % 2 + z % 2) * dx / 2.;
                    pos.y = origin.y + double(2. * y - 1 + 2. * (z % 2) / 3.) * dy / 2. + (1. / sqrt(3.) - 0.5) * dy;
                    pos.z = origin.z + 0.5 * (spacing - dz) + (2. * z - 1.) * dz / 2.;
					sph.radii[count] = sphereRadius;
					sph.materialIndex[count] = materialId;
					sph.bondClusterIndex[count] = bondClusterId;
                    sph.state.positions[count] = pos;
                    sph.state.velocities[count] = velocity;
                    if (mass > 0)
                    {
						sph.state.inverseMass[count] = 1. / mass;
						symMatrix inertiaTensor = make_symMatrix(inertia, inertia, inertia, 0., 0., 0.);
						sph.state.inverseInertia[count] = inverse(inertiaTensor);
					}
                    count++;
                }
            }
        }
        insertData(sph);
    }
};

struct HostClump
{
    int num = 0;
    std::vector<int> pebbleStart;
    std::vector<int> pebbleEnd;
    HostDynamicState state;

    HostClump() = default;

	explicit HostClump(int n)
	{
		num = n;
		pebbleStart.resize(n, 0);
		pebbleEnd.resize(n, 0);
		state = HostDynamicState(n);
	}

    void createBlockSample(HostSphere& spheres, double3 origin, double3 sampleSize, double3 velocity, double density, double spacing, double sphereRadius, int materialId)
    {
        if (spacing <= 0.)
        {
            std::cerr << "Spacing must be positive.\n";
            return;
        }
        int numPX = int(sampleSize.x / spacing);
        int numPY = int(sampleSize.y / spacing);
        int numPZ = int(sampleSize.z / spacing);
        int N = numPX * numPY * numPZ;
        if (N <= 0) return;
        
        if(num == 0) pebbleStart.push_back(0);
        else pebbleStart.push_back(pebbleEnd.back());
        pebbleEnd.push_back(pebbleStart.back() + N);
        state.insertData(HostDynamicState(1));
        state.positions.back() = origin + 0.5 * sampleSize;
        double M = sampleSize.x * sampleSize.y * sampleSize.z * density;
        if (M > 0) state.inverseMass.back() = 1. / M;
        symMatrix iT = make_symMatrix(M * (sampleSize.y * sampleSize.y + sampleSize.z * sampleSize.z) / 12., M * (sampleSize.x * sampleSize.x + sampleSize.z * sampleSize.z) / 12., M * (sampleSize.x * sampleSize.x + sampleSize.y * sampleSize.y) / 12., 0., 0., 0.);
        state.inverseInertia.back() = inverse(iT);
        state.velocities.back() = velocity;
        num++;

        HostSphere sph = HostSphere(N);
        int count = 0;
        for (int x = 0; x < numPX; x++)
        {
            for (int y = 0; y < numPY; y++)
            {
                for (int z = 0; z < numPZ; z++)
                {
                    double3 pos = make_double3(0, 0, 0);
                    pos.x = origin.x + (x + 0.5) * spacing;
                    pos.y = origin.y + (y + 0.5) * spacing;
                    pos.z = origin.z + (z + 0.5) * spacing;
                    sph.radii[count] = sphereRadius;
                    sph.materialIndex[count] = materialId;
                    sph.clumpIndex[count] = num - 1;
                    sph.state.positions[count] = pos;
                    sph.state.velocities[count] = velocity;
                    if (M > 0)
                    {
                        sph.state.inverseMass[count] = double(N) / M;
                        double inertia = 0.4 / sph.state.inverseMass[count] * pow(sphereRadius, 2);
                        symMatrix inertiaTensor = make_symMatrix(inertia, inertia, inertia, 0., 0., 0.);
                        sph.state.inverseInertia[count] = inverse(inertiaTensor);
                    }
                    count++;
                }
            }
        }
        spheres.insertData(sph);
    }
};

struct HostSPH
{
    int num = 0;
    double z0 = 0.;
    double H0 = 1.;
    double density0 = 1000.;
    double alpha = 0.1;//artificial viscosity coefficient
    double beta = 0.0;//viscosity coefficient
    double c0 = 50.;//speed of sound
    std::vector<double> density;
    std::vector<double> pressure;

	HostSPH() = default;

	explicit HostSPH(int n)
	{
		num = n;
		z0 = 0.;
		H0 = 1.;
		density0 = 1000.;
		alpha = 0.1;
		c0 = 50.;
		density.resize(n, density0);
		pressure.resize(n, 0.);
	}

	void createBlockSample(HostSphere& spheres, double3 origin, double3 sampleSize, double initialDensity, double smoothLength, double A, double B, double soundSpeed, int materialId)
	{
		if (sampleSize.x <= 0 || sampleSize.y <= 0 || sampleSize.z <= 0)
		{
			std::cerr << "Sample size must be positive.\n";
			return;
		}
		if (smoothLength <= 0)
		{
			std::cerr << "Smooth length must be positive.\n";
			return;
		}
		z0 = origin.z + sampleSize.z;
		H0 = sampleSize.z;
		density0 = initialDensity;
		alpha = A;
        beta = B;
        c0 = soundSpeed;
        double elementSzie = smoothLength / 1.2;
        int Nx = sampleSize.x > 2 * smoothLength ? int(sampleSize.x / smoothLength) - 1: 1;
        int Ny = sampleSize.y > 2 * smoothLength ? int(sampleSize.y / smoothLength) - 1: 1;
        int Nz = sampleSize.z > 2 * smoothLength ? int(sampleSize.z / smoothLength) - 1: 1;
        double spacingX = Nx > 1 ? (sampleSize.x - 2 * smoothLength) / double(Nx - 1) : 0.5 * sampleSize.x;
        double spacingY = Ny > 1 ? (sampleSize.y - 2 * smoothLength) / double(Ny - 1) : 0.5 * sampleSize.y;
        double spacingZ = Nz > 1 ? (sampleSize.z - 2 * smoothLength) / double(Nz - 1) : 0.5 * sampleSize.z;
		num = Nx * Ny * Nz;
		density.resize(num, density0);
		pressure.resize(num, 0.);

        HostSphere sph = HostSphere(num);
        int count = 0;
        for (int x = 0; x < Nx; ++x)
        {
            for (int y = 0; y < Ny; ++y)
            {
                for (int z = 0; z < Nz; ++z)
                {
					sph.radii[count] = smoothLength;
					sph.materialIndex[count] = materialId;
					sph.SPHIndex[count] = count; 
                    sph.state.positions[count].x = origin.x + x * spacingX + smoothLength;
                    sph.state.positions[count].y = origin.y + y * spacingY + smoothLength;
                    sph.state.positions[count].z = origin.z + z * spacingZ + smoothLength;
                    double mass = density0 * elementSzie * elementSzie * elementSzie;
                    sph.state.inverseMass[count] = 1. / mass;
                    double inertia = 1. / 6. * mass * elementSzie * elementSzie;
                    symMatrix inertiaTensor = make_symMatrix(inertia, inertia, inertia, 0., 0., 0.);
                    sph.state.inverseInertia[count] = inverse(inertiaTensor);
					count++;
                }
            }
        }
		spheres.insertData(sph);
	}
};

struct HostTriangleFace
{
    int num = 0;
    std::vector<int> vAIndex;
    std::vector<int> vBIndex;
    std::vector<int> vCIndex;
    std::vector<int> face2Wall;

	HostTriangleFace() = default;

	explicit HostTriangleFace(int n)
	{
		num = n;
		vAIndex.resize(n, 0);
		vBIndex.resize(n, 0);
		vCIndex.resize(n, 0);
		face2Wall.resize(n, 0);
	}
};

struct HostTriangleEdge
{
    int num = 0;
    std::vector<int> vAIndex;
    std::vector<int> vBIndex;
    std::vector<int> edge2Wall;
    std::vector<int> facePrefixSum;

	HostTriangleEdge() = default;

    explicit HostTriangleEdge(int n)
    {
        num = n;
        vAIndex.resize(n, 0);
        vBIndex.resize(n, 0);
        edge2Wall.resize(n, 0);
        facePrefixSum.resize(n, 0);
    }

    std::vector<int> edge2Face;
};

struct HostTriangleVertex
{
    int num = 0;
    std::vector<double3> positions;
    std::vector<int> vertex2Wall;
    std::vector<int> facePrefixSum;
    std::vector<int> edgePrefixSum;

	HostTriangleVertex() = default;

	explicit HostTriangleVertex(int n)
	{
		num = n;
		positions.resize(n, make_double3(0., 0., 0.));
		vertex2Wall.resize(n, 0);
		facePrefixSum.resize(n, 0);
		edgePrefixSum.resize(n, 0);
	}

    std::vector<int> vertex2Face;
    std::vector<int> vertex2Edge;
};

struct HostTriangleWall
{
    int num = 0;
    std::vector<int> materialIndex;
    HostDynamicState state;
    HostTriangleFace face;
    HostTriangleEdge edge;
    HostTriangleVertex vertex;

    HostTriangleWall() = default;

    void loadEdgeInfo()
    {
		if (face.num == 0 || vertex.num == 0)
		{
			std::cerr << "No faces or vertices to create edges.\n";
			return;
		}
        /*---------------------------------------------------------------
         edge、edge↔face、vertex↔{face,edge}
        ----------------------------------------------------------------*/
        using EdgeKey = std::pair<int, int>;  // (min,max)

        struct KeyHash
        {
            size_t operator()(const EdgeKey& k) const noexcept
            {
                return (static_cast<size_t>(k.first) << 32) ^ static_cast<size_t>(k.second);
            }
        };

        // faces
        std::unordered_map<EdgeKey, std::vector<int>, KeyHash> edgeMap;

        auto addEdge = [&](int a, int b, int faceId)
        {
            if (a > b) std::swap(a, b);
            edgeMap[{a, b}].push_back(faceId);
        };

        // {face,edge}
        std::vector<std::vector<int>> v2Face(vertex.num), v2Edge(vertex.num);

        for (int f = 0; f < face.num; ++f)
        {
            int a = face.vAIndex[f];
            int b = face.vBIndex[f];
            int c = face.vCIndex[f];

            addEdge(a, b, f);
            addEdge(b, c, f);
            addEdge(c, a, f);

            v2Face[a].push_back(f);
            v2Face[b].push_back(f);
            v2Face[c].push_back(f);
        }

        // edge
        int nE = static_cast<int>(edgeMap.size());
        edge = HostTriangleEdge(nE);
        edge.edge2Face.clear();
        edge.edge2Face.reserve(nE * 2);

        int eIdx = 0, faceCum = 0;
        for (const auto& kv : edgeMap)
        {
            int a = kv.first.first;
            int b = kv.first.second;
            edge.vAIndex[eIdx] = a;
            edge.vBIndex[eIdx] = b;

            // wallId
            int wallId = face.face2Wall[kv.second.front()];
            edge.edge2Wall[eIdx] = wallId;

            // edge2Face
            for (int f : kv.second)
                edge.edge2Face.push_back(f);

            faceCum += static_cast<int>(kv.second.size());
            edge.facePrefixSum[eIdx] = faceCum;

            // vertex -> edge
            v2Edge[a].push_back(eIdx);
            v2Edge[b].push_back(eIdx);

            ++eIdx;
        }

        /* vertex */
        int totF = 0, totE = 0;
        for (int v = 0; v < vertex.num; ++v)
        {
            totF += static_cast<int>(v2Face[v].size());
            totE += static_cast<int>(v2Edge[v].size());
            vertex.facePrefixSum[v] = totF;
            vertex.edgePrefixSum[v] = totE;
        }
        vertex.vertex2Face.resize(totF);
        vertex.vertex2Edge.resize(totE);
        int cf = 0, ce = 0;
        for (int v = 0; v < vertex.num; ++v)
        {
            for (int f : v2Face[v]) vertex.vertex2Face[cf++] = f;
            for (int e : v2Edge[v]) vertex.vertex2Edge[ce++] = e;
        }
    }

    void addBoxWall(double3 origin, double3 size, int matId)
    {
        int nV = vertex.num;
		int nE = edge.num;
		int nF = face.num;

        num += 1;
        materialIndex.push_back(matId);
		state.insertData(HostDynamicState(1));
        double3 pos = origin + 0.5 * size;
        state.positions.back() = pos;

        face.num += 12;
		face.vAIndex.push_back(nV + 0);
        face.vBIndex.push_back(nV + 1);
        face.vCIndex.push_back(nV + 2);
        face.vAIndex.push_back(nV + 0);
        face.vBIndex.push_back(nV + 2);
        face.vCIndex.push_back(nV + 3);
        face.vAIndex.push_back(nV + 0);
        face.vBIndex.push_back(nV + 4);
        face.vCIndex.push_back(nV + 1);
        face.vAIndex.push_back(nV + 1);
        face.vBIndex.push_back(nV + 4);
        face.vCIndex.push_back(nV + 5);
        face.vAIndex.push_back(nV + 1);
        face.vBIndex.push_back(nV + 5);
        face.vCIndex.push_back(nV + 2);
        face.vAIndex.push_back(nV + 2);
        face.vBIndex.push_back(nV + 5);
        face.vCIndex.push_back(nV + 6);
        face.vAIndex.push_back(nV + 2);
        face.vBIndex.push_back(nV + 7);
        face.vCIndex.push_back(nV + 3);
        face.vAIndex.push_back(nV + 2);
        face.vBIndex.push_back(nV + 6);
        face.vCIndex.push_back(nV + 7);
        face.vAIndex.push_back(nV + 3);
        face.vBIndex.push_back(nV + 7);
        face.vCIndex.push_back(nV + 4);
        face.vAIndex.push_back(nV + 0);
        face.vBIndex.push_back(nV + 3);
        face.vCIndex.push_back(nV + 4);
        face.vAIndex.push_back(nV + 4);
        face.vBIndex.push_back(nV + 6);
        face.vCIndex.push_back(nV + 5);
        face.vAIndex.push_back(nV + 4);
        face.vBIndex.push_back(nV + 7);
        face.vCIndex.push_back(nV + 6);
        for (int i = 0; i < 12; ++i)
        {
            face.face2Wall.push_back(num - 1);
        }

        vertex.num += 8;
        vertex.positions.push_back(origin + make_double3(0, 0, 0) - pos);
        vertex.positions.push_back(origin + make_double3(size.x, 0, 0) - pos);
        vertex.positions.push_back(origin + make_double3(size.x, size.y, 0) - pos);
        vertex.positions.push_back(origin + make_double3(0, size.y, 0) - pos);
        vertex.positions.push_back(origin + make_double3(0, 0, size.z) - pos);
        vertex.positions.push_back(origin + make_double3(size.x, 0, size.z) - pos);
        vertex.positions.push_back(origin + make_double3(size.x, size.y, size.z) - pos);
        vertex.positions.push_back(origin + make_double3(0, size.y, size.z) - pos);
        for (int i = 0; i < 8; ++i)
        {
            vertex.vertex2Wall.push_back(num - 1);
            vertex.facePrefixSum.push_back(nF);
            vertex.edgePrefixSum.push_back(nE);
        }

        loadEdgeInfo();
    }

    void addPlaneWall(double3 pos, double3 vAG, double3 vBG, double3 vCG, double3 vDG, int matId)
    {
        int nV = vertex.num;
        int nE = edge.num;
        int nF = face.num;

        num += 1;
        materialIndex.push_back(matId);
        state.insertData(HostDynamicState(1));
        state.positions.back() = pos;

        face.num += 2;
        face.vAIndex.push_back(nV + 0);
        face.vBIndex.push_back(nV + 1);
        face.vCIndex.push_back(nV + 2);
        face.vAIndex.push_back(nV + 0);
        face.vBIndex.push_back(nV + 2);
        face.vCIndex.push_back(nV + 3);
        for (int i = 0; i < 2; ++i)
        {
            face.face2Wall.push_back(num - 1);
        }

        vertex.num += 4;
        vertex.positions.push_back(vAG - pos);
        vertex.positions.push_back(vBG - pos);
        vertex.positions.push_back(vCG - pos);
        vertex.positions.push_back(vDG - pos);
        for (int i = 0; i < 4; ++i)
        {
            vertex.vertex2Wall.push_back(num - 1);
            vertex.facePrefixSum.push_back(nF);
            vertex.edgePrefixSum.push_back(nE);
        }

        loadEdgeInfo();
    }

    void addVerticalCylinder(double3 bottomPos, double radius, double height, int matId, int revolution = 12)
    {
        if (revolution < 3)
        {
			std::cerr << "Revolution must be at least 3.\n";
			return;
        }
        int nV = vertex.num;
        int nE = edge.num;
        int nF = face.num;

        num += 1;
        materialIndex.push_back(matId);
        state.insertData(HostDynamicState(1));
        state.positions.back() = bottomPos;

        face.num += 2 * revolution;
		for (int i = 0; i < revolution; ++i)
		{
			int a = nV + i;
			int b = nV + (i + 1) % revolution;
			int c = nV + (i + 1) % revolution + revolution;
			int d = nV + i + revolution;
			face.vAIndex.push_back(a);
			face.vBIndex.push_back(b);
			face.vCIndex.push_back(c);
			face.vAIndex.push_back(a);
			face.vBIndex.push_back(c);
			face.vCIndex.push_back(d);
			face.face2Wall.push_back(num - 1);
			face.face2Wall.push_back(num - 1);
		}

		vertex.num += 2 * revolution;
		double angleStep = 2.0 * pi() / double(revolution);
		for (int i = 0; i < revolution; ++i)
		{
			double angle = i * angleStep;
			double x = radius * cos(angle);
			double y = radius * sin(angle);
			vertex.positions.push_back(make_double3(x, y, 0));
			vertex.vertex2Wall.push_back(num - 1);
			vertex.facePrefixSum.push_back(nF);
			vertex.edgePrefixSum.push_back(nE);
		}
		for (int i = 0; i < revolution; ++i)
		{
			double angle = i * angleStep;
			double x = radius * cos(angle);
			double y = radius * sin(angle);
			vertex.positions.push_back(make_double3(x, y, height));
			vertex.vertex2Wall.push_back(num - 1);
			vertex.facePrefixSum.push_back(nF);
			vertex.edgePrefixSum.push_back(nE);
		}

		loadEdgeInfo();
    }
};

struct HostBasicInteraction
{
    int capacity = 0;
    int num = 0;
    std::vector<int> objectPointed;
    std::vector<int> objectPointing;
    std::vector<double> normalOverlap;
    std::vector<double3> contactNormal;
    std::vector<double3> contactPoint;
    std::vector<double3> slidingSpring;
    std::vector<double3> rollingSpring;
    std::vector<double3> torsionSpring;
    std::vector<double3> contactForce;
    std::vector<double3> contactTorque;

    HostBasicInteraction() = default;

    explicit HostBasicInteraction(int n) : capacity(n)
    {
        num = 0;
        objectPointed.resize(n, 0);
        objectPointing.resize(n, 0);
        normalOverlap.resize(n, 0);
        contactNormal.resize(n, make_double3(0., 0., 0.));
        contactPoint.resize(n, make_double3(0., 0., 0.));
        slidingSpring.resize(n, make_double3(0., 0., 0.));
        rollingSpring.resize(n, make_double3(0., 0., 0.));
        torsionSpring.resize(n, make_double3(0., 0., 0.));
        contactForce.resize(n, make_double3(0., 0., 0.));
        contactTorque.resize(n, make_double3(0., 0., 0.));
    }
};

struct HostBondedInteraction
{
    int num = 0;
    std::vector<int> objectPointed;
    std::vector<int> objectPointing;
    std::vector<int> isBonded;
    std::vector<double3> contactNormal;
    std::vector<double3> contactPoint;
    std::vector<double> normalForce;
    std::vector<double> torsionTorque;
    std::vector<double3> shearForce;
    std::vector<double3> bendingTorque;

    HostBondedInteraction() = default;

    explicit HostBondedInteraction(int n) : num(n)
    {
        num = 0;
        objectPointed.resize(n, 0);
        objectPointing.resize(n, 0);
        isBonded.resize(n, 0);
        contactNormal.resize(n, make_double3(0., 0., 0.));
        contactPoint.resize(n, make_double3(0., 0., 0.));
        normalForce.resize(n, 0.);
        torsionTorque.resize(n, 0.);
        shearForce.resize(n, make_double3(0., 0., 0.));
        bendingTorque.resize(n, make_double3(0., 0., 0.));
    }
};

struct HostMaterialProperty
{
    int num = 0;
    std::vector<double> elasticModulus;
    std::vector<double> poissonRatio;
};

struct HostDirectionalTerms
{
    std::vector<double> normal;
    std::vector<double> sliding;
    std::vector<double> rolling;
    std::vector<double> torsion;
};

struct HostHertzianContactModel
{
    int num = 0;
    std::vector<double> kR_to_kS_ratio;
    std::vector<double> kT_to_kS_ratio;
    std::vector<double> restitution;
    HostDirectionalTerms friction;
};

struct HostLinearContactModel
{
    int num = 0;
    HostDirectionalTerms stiffness;
    HostDirectionalTerms dissipation;
    HostDirectionalTerms friction;
};

struct HostBondedContactModel
{
    int num = 0;
    std::vector<double> maxContactGap;
    std::vector<double> multiplier;
    std::vector<double> elasticModulus;
    std::vector<double> kN_to_kS_ratio;
    std::vector<double> tensileStrength;
    std::vector<double> cohesion;
    std::vector<double> frictionCoeff;
};

struct HostContactParameter
{
    HostMaterialProperty material;
    HostHertzianContactModel Hertzian;
    HostLinearContactModel Linear;
    HostBondedContactModel Bond;

	HostContactParameter() = default;

	explicit HostContactParameter(int nMat)
	{
		material.num = nMat;
		material.elasticModulus.resize(nMat, 0.);
		material.poissonRatio.resize(nMat, 0.);

		int nC = nMat * (nMat + 1) / 2; // number of contact pairs
		Hertzian.num = nC;
        Hertzian.kR_to_kS_ratio.resize(nC, 1.);
		Hertzian.kT_to_kS_ratio.resize(nC, 1.);
		Hertzian.restitution.resize(nC, 1.);
		Hertzian.friction.normal.resize(nC, 0.);
		Hertzian.friction.sliding.resize(nC, 0.);
		Hertzian.friction.rolling.resize(nC, 0.);
		Hertzian.friction.torsion.resize(nC, 0.);
		Linear.num = nC;
		Linear.stiffness.normal.resize(nC, 0.);
		Linear.stiffness.sliding.resize(nC, 0.);
		Linear.stiffness.rolling.resize(nC, 0.);
		Linear.stiffness.torsion.resize(nC, 0.);
		Linear.dissipation.normal.resize(nC, 0.);
		Linear.dissipation.sliding.resize(nC, 0.);
		Linear.dissipation.rolling.resize(nC, 0.);
		Linear.dissipation.torsion.resize(nC, 0.);
		Linear.friction.normal.resize(nC, 0.);
		Linear.friction.sliding.resize(nC, 0.);
		Linear.friction.rolling.resize(nC, 0.);
		Linear.friction.torsion.resize(nC, 0.);
		Bond.num = nC;
		Bond.maxContactGap.resize(nC, 0.);
		Bond.multiplier.resize(nC, 1.);
		Bond.elasticModulus.resize(nC, 0.);
		Bond.kN_to_kS_ratio.resize(nC, 1.);
		Bond.tensileStrength.resize(nC, 0.);
		Bond.cohesion.resize(nC, 0.);
		Bond.frictionCoeff.resize(nC, 0.);
	}

    int getContactParameterIndex(int mA, int mB)
    {
        if (material.num < 2) return 0;
        int i = mA;
        int j = mB;
        if (mA > mB)
        {
            i = mB;
            j = mA;
        }
        int index = (i * (2 * material.num - i + 1)) / 2 + j - i;
        return index;
    }
};

struct HostSpatialGrid
{
    int num = 0;
    double3 minBound = make_double3(0., 0., 0.);
    double3 maxBound = make_double3(1., 1., 1.);
    double3 cellSize = make_double3(1., 1., 1.);
    int3 gridSize = make_int3(1, 1, 1);
};

struct HostData
{
    HostSphere spheres;
    HostSPH SPHParticles;
    HostClump clumps;
    HostTriangleWall triangleWalls;
    HostBasicInteraction sphSphInteract;
    HostBondedInteraction sphSphBondedInteract;
    HostBasicInteraction faceSphInteract;
    HostBasicInteraction edgeSphInteract;
    HostBasicInteraction vertexSphInteract;
    HostContactParameter contactPara;
    HostSpatialGrid spatialGrids;
};