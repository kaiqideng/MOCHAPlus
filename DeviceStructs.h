#pragma once
#include "cuda_mem_utils.h"
#include "mySymMatrix.h"
#include "HostStructs.h"

struct DynamicState {
    double3* positions{ nullptr };
    quaternion* orientations{ nullptr };
    double3* velocities{ nullptr };
    double3* angularVelocities{ nullptr };
    double3* forces{ nullptr };
    double3* torques{ nullptr };
    double* inverseMass{ nullptr };
    symMatrix* inverseInertia{ nullptr };

    void alloc(int n)
    {
        CUDA_ALLOC(positions, n, InitMode::NONE);
        CUDA_ALLOC(orientations, n, InitMode::NONE);
        CUDA_ALLOC(velocities, n, InitMode::NONE);
        CUDA_ALLOC(angularVelocities, n, InitMode::NONE);
        CUDA_ALLOC(forces, n, InitMode::NONE);
        CUDA_ALLOC(torques, n, InitMode::NONE);
        CUDA_ALLOC(inverseMass, n, InitMode::NONE);
        CUDA_ALLOC(inverseInertia, n, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(positions); 
        CUDA_FREE(orientations);
        CUDA_FREE(velocities); 
        CUDA_FREE(angularVelocities);
        CUDA_FREE(forces);     
        CUDA_FREE(torques);
        CUDA_FREE(inverseMass);
        CUDA_FREE(inverseInertia);
    }
    void copy(int n, const HostDynamicState& h)
    {
        release();
        if (n == 0) return;
        alloc(n);
        cuda_copy(positions, h.positions.data(), size_t(n), CopyDir::H2D);
        cuda_copy(orientations, h.orientations.data(), size_t(n), CopyDir::H2D);
        cuda_copy(velocities, h.velocities.data(), size_t(n), CopyDir::H2D);
        cuda_copy(angularVelocities, h.angularVelocities.data(), size_t(n), CopyDir::H2D);
        cuda_copy(forces, h.forces.data(), size_t(n), CopyDir::H2D);
        cuda_copy(torques, h.torques.data(), size_t(n), CopyDir::H2D);
        cuda_copy(inverseMass, h.inverseMass.data(), size_t(n), CopyDir::H2D);
        cuda_copy(inverseInertia, h.inverseInertia.data(), size_t(n), CopyDir::H2D);
    }
    void upload(int n, HostDynamicState& h)
    {
        if (n == 0) return;
        cuda_copy(h.positions.data(), positions, size_t(n), CopyDir::D2H);
        cuda_copy(h.orientations.data(), orientations, size_t(n), CopyDir::D2H);
        cuda_copy(h.velocities.data(), velocities, size_t(n), CopyDir::D2H);
        cuda_copy(h.angularVelocities.data(), angularVelocities, size_t(n), CopyDir::D2H);
        cuda_copy(h.forces.data(), forces, size_t(n), CopyDir::D2H);
        cuda_copy(h.torques.data(), torques, size_t(n), CopyDir::D2H);
        cuda_copy(h.inverseMass.data(), inverseMass, size_t(n), CopyDir::D2H);
        cuda_copy(h.inverseInertia.data(), inverseInertia, size_t(n), CopyDir::D2H);
    }
    void download(int n, HostDynamicState& h)
    {
        if (n == 0) return;
        cuda_copy(positions, h.positions.data(), size_t(n), CopyDir::H2D);
        cuda_copy(orientations, h.orientations.data(), size_t(n), CopyDir::H2D);
        cuda_copy(velocities, h.velocities.data(), size_t(n), CopyDir::H2D);
        cuda_copy(angularVelocities, h.angularVelocities.data(), size_t(n), CopyDir::H2D);
        cuda_copy(forces, h.forces.data(), size_t(n), CopyDir::H2D);
        cuda_copy(torques, h.torques.data(), size_t(n), CopyDir::H2D);
        cuda_copy(inverseMass, h.inverseMass.data(), size_t(n), CopyDir::H2D);
        cuda_copy(inverseInertia, h.inverseInertia.data(), size_t(n), CopyDir::H2D);
    }
};

struct ObjectHash {
    int* value{ nullptr };
    int* aux{ nullptr };
    int* index{ nullptr };

    void alloc(int n)
    {
        CUDA_ALLOC(value, n, InitMode::NEG_ONE);
        CUDA_ALLOC(aux, n, InitMode::NEG_ONE);
        CUDA_ALLOC(index, n, InitMode::NEG_ONE);
    }
    void release()
    { 
        CUDA_FREE(value); 
        CUDA_FREE(aux); 
        CUDA_FREE(index); 
    }
    void reset(int n)
    {
        CUDA_CHECK(cudaMemset(value, 0xFFFFFFFF, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(aux, 0xFFFFFFFF, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(index, 0xFFFFFFFF, n * sizeof(int)));
    }
};

struct NeighborPrefix {
    int* count{ nullptr };
    int* prefixSum{ nullptr };

    void alloc(int n)
    {
        CUDA_ALLOC(count, n, InitMode::ZERO);
        CUDA_ALLOC(prefixSum, n, InitMode::ZERO);
    }
    void release()
    { 
        CUDA_FREE(count);
        CUDA_FREE(prefixSum); 
    }
};

struct InteractionRange {
    int* start{ nullptr };
    int* end{ nullptr };

    void alloc(int n)
    {
        CUDA_ALLOC(start, n, InitMode::NEG_ONE);
        CUDA_ALLOC(end, n, InitMode::NEG_ONE);
    }
    void release() 
    { 
        CUDA_FREE(start); 
        CUDA_FREE(end); 
    }
    void reset(int n)
    {
        CUDA_CHECK(cudaMemset(start, 0xFFFFFFFF, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(end, 0xFFFFFFFF, n * sizeof(int)));
    }
};

struct Sphere {
    int      num{ 0 };
    int* clumpIndex{ nullptr };
    int* materialIndex{ nullptr };
    int* bondClusterIndex{ nullptr };
    double* radii{ nullptr };

    DynamicState    state;
    ObjectHash      hash;
    NeighborPrefix  neighbor;
    InteractionRange sphereRange, faceRange, edgeRange, vertexRange;

    int* SPHIndex{ nullptr };
    
    void alloc(int n)
    {
        num = n;
        CUDA_ALLOC(clumpIndex, n, InitMode::NONE);
        CUDA_ALLOC(materialIndex, n, InitMode::NONE);
        CUDA_ALLOC(bondClusterIndex, n, InitMode::NONE);
        CUDA_ALLOC(radii, n, InitMode::NONE);

        state.alloc(n);
        hash.alloc(n);
        neighbor.alloc(n);
        sphereRange.alloc(n);
        faceRange.alloc(n);
        edgeRange.alloc(n);
        vertexRange.alloc(n);

        CUDA_ALLOC(SPHIndex, n, InitMode::NEG_ONE);
    }
    void release()
    {
        CUDA_FREE(clumpIndex);
        CUDA_FREE(materialIndex); 
        CUDA_FREE(bondClusterIndex);
        CUDA_FREE(radii);
        state.release();
        hash.release(); 
        neighbor.release(); 
        sphereRange.release();
        faceRange.release(); 
        edgeRange.release();
        vertexRange.release();
        CUDA_FREE(SPHIndex);
        num = 0;
    }
    void copy(const HostSphere& h)
    {
        release();
        alloc(h.num);
        cuda_copy(clumpIndex, h.clumpIndex.data(), size_t(num), CopyDir::H2D);
        cuda_copy(materialIndex, h.materialIndex.data(), size_t(num), CopyDir::H2D);
        cuda_copy(bondClusterIndex, h.bondClusterIndex.data(), size_t(num), CopyDir::H2D);
        cuda_copy(radii, h.radii.data(), size_t(num), CopyDir::H2D);
        state.copy(num, h.state);
        cuda_copy(SPHIndex, h.SPHIndex.data(), size_t(num), CopyDir::H2D);
    }
    void uploadState(HostSphere& h)
    {
        state.upload(num, h.state);
    }
    void downloadState(HostSphere& h)
	{
		if (num < h.num) return;
		state.download(h.num, h.state);
	}
};

struct SPH {
    int num{ 0 };
	double z0{ 0 };
    double H0{ 0 };
    double density0{ 0 };
    double alpha{ 0 };
    double beta{ 0 };
    double c0{ 0 };
    double* density{ nullptr };
    double* pressure{ nullptr };
    double* effectiveVolume{ nullptr };
    double3* XSPHVariant{ nullptr };
    symMatrix* SPSStress{ nullptr };

    void alloc(int n)
    {
        num = n;
        CUDA_ALLOC(density, n, InitMode::NONE);
        CUDA_ALLOC(pressure, n, InitMode::NONE);
        CUDA_ALLOC(effectiveVolume, n, InitMode::ZERO);
		CUDA_ALLOC(XSPHVariant, n, InitMode::ZERO);
		CUDA_ALLOC(SPSStress, n, InitMode::ZERO);
    }
    void release()
    {
        CUDA_FREE(density);
        CUDA_FREE(pressure);
        CUDA_FREE(effectiveVolume);
		CUDA_FREE(XSPHVariant);
		CUDA_FREE(SPSStress);
        num = 0;
    }
    void copy(const HostSPH& h)
    {
        if (h.num == 0) return;
        release();
        alloc(h.num);
        z0 = h.z0;
        H0 = h.H0;
        density0 = h.density0;
        alpha = h.alpha;
        beta = h.beta;
        c0 = h.c0;
        cuda_copy(density, h.density.data(), size_t(num), CopyDir::H2D);
        cuda_copy(pressure, h.pressure.data(), size_t(num), CopyDir::H2D);
    }
    void upload(HostSPH& h)
    {
        if (num == 0) return;
        cuda_copy(h.density.data(), density, size_t(num), CopyDir::D2H);
        cuda_copy(h.pressure.data(), pressure, size_t(num), CopyDir::D2H);
    }
};

struct Clump {
    int num{ 0 };
    int* pebbleStart{ nullptr };
    int* pebbleEnd{ nullptr };
    DynamicState state;

    void alloc(int n)
    {
        num = n;
        CUDA_ALLOC(pebbleStart, n, InitMode::NONE);
        CUDA_ALLOC(pebbleEnd, n, InitMode::NONE);
        state.alloc(n);
    }
    void release()
    {
        CUDA_FREE(pebbleStart);
        CUDA_FREE(pebbleEnd);
        state.release();
        num = 0;
    }
    void copy(const HostClump& h)
    {
        if (h.num == 0) return;
        release();
        alloc(h.num);
        cuda_copy(pebbleStart, h.pebbleStart.data(), size_t(num), CopyDir::H2D);
        cuda_copy(pebbleEnd, h.pebbleEnd.data(), size_t(num), CopyDir::H2D);
        state.copy(num, h.state);
    }
    void uploadState(HostClump& h)
	{
		state.upload(num, h.state);
	}
};

struct TriangleFace {
    int  num{ 0 };
    int* vAIndex{ nullptr };
    int* vBIndex{ nullptr };
    int* vCIndex{ nullptr };
    NeighborPrefix neighbor;
    int* face2Wall{ nullptr };

    void alloc(int n)
    {
        num = n;
        CUDA_ALLOC(vAIndex, n, InitMode::NONE);
        CUDA_ALLOC(vBIndex, n, InitMode::NONE);
        CUDA_ALLOC(vCIndex, n, InitMode::NONE);
        neighbor.alloc(n);
        CUDA_ALLOC(face2Wall, n, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(vAIndex); 
        CUDA_FREE(vBIndex); 
        CUDA_FREE(vCIndex);
        neighbor.release();
        CUDA_FREE(face2Wall);
        num = 0;
    }
    void copy(const HostTriangleFace& h)
    {
        release();
        alloc(h.num);
        cuda_copy(vAIndex, h.vAIndex.data(), num, CopyDir::H2D);
        cuda_copy(vBIndex, h.vBIndex.data(), num, CopyDir::H2D);
        cuda_copy(vCIndex, h.vCIndex.data(), num, CopyDir::H2D);
        cuda_copy(face2Wall, h.face2Wall.data(), num, CopyDir::H2D);
    }
};

/* ---------- Edge list ---------- */
struct TriangleEdge {
    int  num{ 0 };
    int* vAIndex{ nullptr };
    int* vBIndex{ nullptr };
    NeighborPrefix neighbor;
    int* edge2Wall{ nullptr };
    int* facePrefixSum{ nullptr };   // size = num
    int* edge2Face{ nullptr };   // size = facePrefixSum[num-1]

    void alloc(int nEdge, int nEdge2Face)
    {
        num = nEdge;
        CUDA_ALLOC(vAIndex, nEdge, InitMode::NONE);
        CUDA_ALLOC(vBIndex, nEdge, InitMode::NONE);
        neighbor.alloc(nEdge);
        CUDA_ALLOC(edge2Wall, nEdge, InitMode::NONE);
        CUDA_ALLOC(facePrefixSum, nEdge, InitMode::NONE);
        CUDA_ALLOC(edge2Face, nEdge2Face, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(vAIndex); 
        CUDA_FREE(vBIndex);
        neighbor.release();
        CUDA_FREE(edge2Wall);
        CUDA_FREE(facePrefixSum);
        CUDA_FREE(edge2Face);
        num = 0;
    }
    void copy(const HostTriangleEdge& h)
    {
        release();
        alloc(h.num, (int)h.edge2Face.size());
        cuda_copy(vAIndex, h.vAIndex.data(), num, CopyDir::H2D);
        cuda_copy(vBIndex, h.vBIndex.data(), num, CopyDir::H2D);
        cuda_copy(edge2Wall, h.edge2Wall.data(), num, CopyDir::H2D);
        cuda_copy(facePrefixSum, h.facePrefixSum.data(), num, CopyDir::H2D);
        cuda_copy(edge2Face, h.edge2Face.data(), h.edge2Face.size(), CopyDir::H2D);
    }
};

/* ---------- Vertex list ---------- */
struct TriangleVertex {
    int      num{ 0 };
    double3* positions{ nullptr };
    NeighborPrefix neighbor;
    int* vertex2Wall{ nullptr };
    int* facePrefixSum{ nullptr };     // size = num
    int* edgePrefixSum{ nullptr };     // size = num
    int* vertex2Face{ nullptr };     // size = facePrefixSum[num-1]
    int* vertex2Edge{ nullptr };     // size = edgePrefixSum[num-1]

    void alloc(int nVertex, int nV2Face, int nV2Edge)
    {
        num = nVertex;
        CUDA_ALLOC(positions, nVertex, InitMode::NONE);
        neighbor.alloc(nVertex);
        CUDA_ALLOC(vertex2Wall, nVertex, InitMode::NONE);
        CUDA_ALLOC(facePrefixSum, nVertex, InitMode::NONE);
        CUDA_ALLOC(edgePrefixSum, nVertex, InitMode::NONE);
        CUDA_ALLOC(vertex2Face, nV2Face, InitMode::NONE);
        CUDA_ALLOC(vertex2Edge, nV2Edge, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(positions);
        neighbor.release();
        CUDA_FREE(vertex2Wall);
        CUDA_FREE(facePrefixSum);
        CUDA_FREE(edgePrefixSum);
        CUDA_FREE(vertex2Face);  
        CUDA_FREE(vertex2Edge);
        num = 0;
    }
    void copy(const HostTriangleVertex& h)
    {
        release();
        alloc(h.num, (int)h.vertex2Face.size(), (int)h.vertex2Edge.size());
        cuda_copy(positions, h.positions.data(), num, CopyDir::H2D);
        cuda_copy(vertex2Wall, h.vertex2Wall.data(), num, CopyDir::H2D);
        cuda_copy(facePrefixSum, h.facePrefixSum.data(), num, CopyDir::H2D);
        cuda_copy(edgePrefixSum, h.edgePrefixSum.data(), num, CopyDir::H2D);
        cuda_copy(vertex2Face, h.vertex2Face.data(), h.vertex2Face.size(), CopyDir::H2D);
        cuda_copy(vertex2Edge, h.vertex2Edge.data(), h.vertex2Edge.size(), CopyDir::H2D);
    }
};

/* ---------- Whole Triangle Wall ---------- */
struct TriangleWall {
    int num{ 0 };            
    int* materialIndex{ nullptr };     // size = num
    DynamicState state;               
    TriangleFace   face;
    TriangleEdge   edge;
    TriangleVertex vertex;

    void alloc(int nWall)
    {
        num = nWall;
        CUDA_ALLOC(materialIndex, nWall, InitMode::NONE);
        state.alloc(nWall);                // per‑wall motion
    }
    void release()
    {
        CUDA_FREE(materialIndex);
        state.release();
        face.release();
        edge.release();
        vertex.release();
        num = 0;
    }
    void copy(const HostTriangleWall& h)
    {
        release();
        alloc(h.num);
        cuda_copy(materialIndex, h.materialIndex.data(), num, CopyDir::H2D);
        state.copy(num, h.state);
        face.copy(h.face);
        edge.copy(h.edge);
        vertex.copy(h.vertex);
    }
    void uploadState(HostTriangleWall& h)
    {
        state.upload(num, h.state);
    }
    void downloadState(HostTriangleWall& h)
    {
        if (num < h.num) return;
        state.download(h.num, h.state);
    }
};

struct BasicInteractionPrev {
    int capacity{ 0 }, num{ 0 };
    int* objectPointed{ nullptr };
    int* objectPointing{ nullptr };
    double3* contactNormal{ nullptr };
    double3* slidingSpring{ nullptr };
    double3* rollingSpring{ nullptr };
    double3* torsionSpring{ nullptr };

    void alloc(int cap)
    {
        capacity = cap; num = 0;
        CUDA_ALLOC(objectPointed, cap, InitMode::NONE);
        CUDA_ALLOC(objectPointing, cap, InitMode::NONE);
        CUDA_ALLOC(contactNormal, cap, InitMode::NONE);
        CUDA_ALLOC(slidingSpring, cap, InitMode::NONE);
        CUDA_ALLOC(rollingSpring, cap, InitMode::NONE);
        CUDA_ALLOC(torsionSpring, cap, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(objectPointed);
        CUDA_FREE(objectPointing);
        CUDA_FREE(contactNormal);
        CUDA_FREE(slidingSpring);
        CUDA_FREE(rollingSpring);
        CUDA_FREE(torsionSpring);
        capacity = num = 0;
    }
};

struct BasicInteraction {
    int capacity{ 0 }, num{ 0 };
    int* objectPointed{ nullptr };
    int* objectPointing{ nullptr };
    double* normalOverlap{ nullptr };
    double3* contactNormal{ nullptr };
    double3* contactPoint{ nullptr };
    double3* slidingSpring{ nullptr };
    double3* rollingSpring{ nullptr };
    double3* torsionSpring{ nullptr };
    double3* contactForce{ nullptr };
    double3* contactTorque{ nullptr };
    BasicInteractionPrev prev;
    ObjectHash hash;

    void allocCurr(int cap)
    {
        capacity = cap; num = 0;
        CUDA_ALLOC(objectPointed, cap, InitMode::NONE);
        CUDA_ALLOC(objectPointing, cap, InitMode::NONE);
        CUDA_ALLOC(normalOverlap, cap, InitMode::NONE);
        CUDA_ALLOC(contactNormal, cap, InitMode::NONE);
        CUDA_ALLOC(contactPoint, cap, InitMode::NONE);
        CUDA_ALLOC(slidingSpring, cap, InitMode::NONE);
        CUDA_ALLOC(rollingSpring, cap, InitMode::NONE);
        CUDA_ALLOC(torsionSpring, cap, InitMode::NONE);
        CUDA_ALLOC(contactForce, cap, InitMode::NONE);
        CUDA_ALLOC(contactTorque, cap, InitMode::NONE);
        hash.alloc(cap);
    }
    void alloc(int cap)
    {
        allocCurr(cap);
        prev.alloc(cap);
    }
    void releaseCurr()
    {
        CUDA_FREE(objectPointed); 
        CUDA_FREE(objectPointing);
        CUDA_FREE(normalOverlap); 
        CUDA_FREE(contactNormal);
        CUDA_FREE(contactPoint); 
        CUDA_FREE(slidingSpring);
        CUDA_FREE(rollingSpring);
        CUDA_FREE(torsionSpring);
        CUDA_FREE(contactForce); 
        CUDA_FREE(contactTorque);
        hash.release();
        capacity = num = 0;
    }
    void release()
    {
        releaseCurr();
        prev.release();
    }
    void copy(const HostBasicInteraction& h)
    {
        release();
        alloc(h.capacity);
        num = h.num;
        cuda_copy(objectPointed, h.objectPointed.data(), num, CopyDir::H2D);
        cuda_copy(objectPointing, h.objectPointing.data(), num, CopyDir::H2D);
        cuda_copy(normalOverlap, h.normalOverlap.data(), num, CopyDir::H2D);
        cuda_copy(contactNormal, h.contactNormal.data(), num, CopyDir::H2D);
        cuda_copy(contactPoint, h.contactPoint.data(), num, CopyDir::H2D);
        cuda_copy(slidingSpring, h.slidingSpring.data(), num, CopyDir::H2D);
        cuda_copy(rollingSpring, h.rollingSpring.data(), num, CopyDir::H2D);
        cuda_copy(torsionSpring, h.torsionSpring.data(), num, CopyDir::H2D);
        cuda_copy(contactForce, h.contactForce.data(), num, CopyDir::H2D);
        cuda_copy(contactTorque, h.contactTorque.data(), num, CopyDir::H2D);
    }
    void upload(HostBasicInteraction& h)
    {
        if (num > h.capacity)
        {
            h = HostBasicInteraction(num);
        }
        h.num = num;
        cuda_copy(h.objectPointed.data(), objectPointed, num, CopyDir::D2H);
        cuda_copy(h.objectPointing.data(), objectPointing, num, CopyDir::D2H);
        cuda_copy(h.normalOverlap.data(), normalOverlap, num, CopyDir::D2H);
        cuda_copy(h.contactNormal.data(), contactNormal, num, CopyDir::D2H);
        cuda_copy(h.contactPoint.data(), contactPoint, num, CopyDir::D2H);
        cuda_copy(h.slidingSpring.data(), slidingSpring, num, CopyDir::D2H);
        cuda_copy(h.rollingSpring.data(), rollingSpring, num, CopyDir::D2H);
        cuda_copy(h.torsionSpring.data(), torsionSpring, num, CopyDir::D2H);
        cuda_copy(h.contactForce.data(), contactForce, num, CopyDir::D2H);
        cuda_copy(h.contactTorque.data(), contactTorque, num, CopyDir::D2H);
    }
    void setNum(int nCurrentInteraction)
    {
        if (nCurrentInteraction > capacity)
        {
            releaseCurr();
            allocCurr(nCurrentInteraction);
        }
        num = nCurrentInteraction;
    }
    void copy2Prev()
    {
        if (num > prev.capacity)
        {
            prev.release();
            prev.alloc(num);
        }
        prev.num = num;
        cuda_copy(prev.objectPointed, objectPointed, num, CopyDir::D2D);
        cuda_copy(prev.objectPointing, objectPointing, num, CopyDir::D2D);
        cuda_copy(prev.contactNormal, contactNormal, num, CopyDir::D2D);
        cuda_copy(prev.slidingSpring, slidingSpring, num, CopyDir::D2D);
        cuda_copy(prev.rollingSpring, rollingSpring, num, CopyDir::D2D);
        cuda_copy(prev.torsionSpring, torsionSpring, num, CopyDir::D2D);
    }
    void setHash()
    {
        cuda_copy(hash.value, objectPointing, num, CopyDir::D2D);
    }
};

struct BondedInteraction {
    int num{ 0 };
    int* objectPointed{ nullptr };
    int* objectPointing{ nullptr };
    int* isBonded{ nullptr };
    double3* contactNormal{ nullptr };
    double3* contactPoint{ nullptr };
    double* normalForce{ nullptr };
    double* torsionTorque{ nullptr };
    double3* shearForce{ nullptr };
    double3* bendingTorque{ nullptr };

    void alloc(int n)
    {
        num = n;
        CUDA_ALLOC(objectPointed, n, InitMode::NONE);
        CUDA_ALLOC(objectPointing, n, InitMode::NONE);
        CUDA_ALLOC(isBonded, n, InitMode::NONE);
        CUDA_ALLOC(contactNormal, n, InitMode::NONE);
        CUDA_ALLOC(contactPoint, n, InitMode::NONE);
        CUDA_ALLOC(normalForce, n, InitMode::NONE);
        CUDA_ALLOC(torsionTorque, n, InitMode::NONE);
        CUDA_ALLOC(shearForce, n, InitMode::NONE);
        CUDA_ALLOC(bendingTorque, n, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(objectPointed); 
        CUDA_FREE(objectPointing);
        CUDA_FREE(isBonded); 
        CUDA_FREE(contactNormal);
        CUDA_FREE(contactPoint);
        CUDA_FREE(normalForce); 
        CUDA_FREE(torsionTorque); 
        CUDA_FREE(shearForce); 
        CUDA_FREE(bendingTorque); 
        num = 0;
    }
	void copy(const HostBondedInteraction& h)
	{
		release();
        if (h.num == 0) return;
		alloc(h.num);
		cuda_copy(objectPointed, h.objectPointed.data(), num, CopyDir::H2D);
		cuda_copy(objectPointing, h.objectPointing.data(), num, CopyDir::H2D);
		cuda_copy(isBonded, h.isBonded.data(), num, CopyDir::H2D);
		cuda_copy(contactNormal, h.contactNormal.data(), num, CopyDir::H2D);
		cuda_copy(contactPoint, h.contactPoint.data(), num, CopyDir::H2D);
		cuda_copy(normalForce, h.normalForce.data(), num, CopyDir::H2D);
		cuda_copy(torsionTorque, h.torsionTorque.data(), num, CopyDir::H2D);
		cuda_copy(shearForce, h.shearForce.data(), num, CopyDir::H2D);
		cuda_copy(bendingTorque, h.bendingTorque.data(), num, CopyDir::H2D);
	}
    void setNumBonds(int n)
    {
        release();
        alloc(n);
    }
    void upload(HostBondedInteraction& h)
    {
		if (num == 0) return;
        if (num > h.num)
        {
            h = HostBondedInteraction(num);
        }
        h.num = num;
        cuda_copy(h.objectPointed.data(), objectPointed, num, CopyDir::D2H);
        cuda_copy(h.objectPointing.data(), objectPointing, num, CopyDir::D2H);
        cuda_copy(h.isBonded.data(), isBonded, num, CopyDir::D2H);
        cuda_copy(h.contactNormal.data(), contactNormal, num, CopyDir::D2H);
        cuda_copy(h.contactPoint.data(), contactPoint, num, CopyDir::D2H);
        cuda_copy(h.normalForce.data(), normalForce, num, CopyDir::D2H);
        cuda_copy(h.torsionTorque.data(), torsionTorque, num, CopyDir::D2H);
        cuda_copy(h.shearForce.data(), shearForce, num, CopyDir::D2H);
        cuda_copy(h.bendingTorque.data(), bendingTorque, num, CopyDir::D2H);
    }
};

struct MaterialProperty {
    int     num{ 0 };
    double* elasticModulus{ nullptr };
    double* poissonRatio{ nullptr };

    void alloc(int n)
    {
        num = n;
        CUDA_ALLOC(elasticModulus, n, InitMode::NONE);
        CUDA_ALLOC(poissonRatio, n, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(elasticModulus); 
        CUDA_FREE(poissonRatio);
        num = 0;
    }
    void copy(const HostMaterialProperty& h)
    {
        release();
        alloc(h.num);
        cuda_copy(elasticModulus, h.elasticModulus.data(), num, CopyDir::H2D);
        cuda_copy(poissonRatio, h.poissonRatio.data(), num, CopyDir::H2D);
    }
};

struct DirectionalTerms {
    double* normal{ nullptr };
    double* sliding{ nullptr };
    double* rolling{ nullptr };
    double* torsion{ nullptr };

    void alloc(int n)
    {
        CUDA_ALLOC(normal, n, InitMode::NONE);
        CUDA_ALLOC(sliding, n, InitMode::NONE);
        CUDA_ALLOC(rolling, n, InitMode::NONE);
        CUDA_ALLOC(torsion, n, InitMode::NONE);
    }
    void release() 
    {
        CUDA_FREE(normal);
        CUDA_FREE(sliding);
        CUDA_FREE(rolling);
        CUDA_FREE(torsion); 
    }
    void copy(int n, const HostDirectionalTerms& h)
    {
        release();
        alloc(n);
        cuda_copy(normal, h.normal.data(), n, CopyDir::H2D);
        cuda_copy(sliding, h.sliding.data(), n, CopyDir::H2D);
        cuda_copy(rolling, h.rolling.data(), n, CopyDir::H2D);
        cuda_copy(torsion, h.torsion.data(), n, CopyDir::H2D);
    }
};

struct HertzianContactModel {
    int     num{ 0 };                // = (m+1)*m/2  (m = number of Materials)
    double* kR_to_kS_ratio{ nullptr };
    double* kT_to_kS_ratio{ nullptr };
    double* restitution{ nullptr };
    DirectionalTerms friction;

    void alloc(int nPair)
    {
        num = nPair;
        CUDA_ALLOC(kR_to_kS_ratio, nPair, InitMode::NONE);
        CUDA_ALLOC(kT_to_kS_ratio, nPair, InitMode::NONE);
        CUDA_ALLOC(restitution, nPair, InitMode::NONE);
        friction.alloc(nPair);
    }
    void release()
    {
        CUDA_FREE(kR_to_kS_ratio);
        CUDA_FREE(kT_to_kS_ratio);
        CUDA_FREE(restitution);
        friction.release(); 
        num = 0;
    }
    void copy(const HostHertzianContactModel& h)
    {
        release();
        alloc(h.num);
        cuda_copy(kR_to_kS_ratio, h.kR_to_kS_ratio.data(), num, CopyDir::H2D);
        cuda_copy(kT_to_kS_ratio, h.kT_to_kS_ratio.data(), num, CopyDir::H2D);
        cuda_copy(restitution, h.restitution.data(), num, CopyDir::H2D);
        friction.copy(num, h.friction);
    }
};

struct LinearContactModel {
    int     num{ 0 };
    DirectionalTerms stiffness;
    DirectionalTerms dissipation;
    DirectionalTerms friction;

    void alloc(int nPair)
    {
        num = nPair;
        stiffness.alloc(nPair);
        dissipation.alloc(nPair);
        friction.alloc(nPair);
    }
    void release()
    {
        stiffness.release();
        dissipation.release();
        friction.release(); 
        num = 0;
    }
    void copy(const HostLinearContactModel& h)
    {
        release();
        alloc(h.num);
		stiffness.copy(num, h.stiffness);
        dissipation.copy(num, h.dissipation);
        friction.copy(num, h.friction);
    }
};

struct BondedContactModel {
    int     num{ 0 };
    double* maxContactGap{ nullptr };
    double* multiplier{ nullptr };
    double* elasticModulus{ nullptr };
    double* kN_to_kS_ratio{ nullptr };
    double* tensileStrength{ nullptr };
    double* cohesion{ nullptr };
    double* frictionCoeff{ nullptr };

    void alloc(int nPair)
    {
        num = nPair;
        CUDA_ALLOC(maxContactGap, nPair, InitMode::NONE);
        CUDA_ALLOC(multiplier, nPair, InitMode::NONE);
        CUDA_ALLOC(elasticModulus, nPair, InitMode::NONE);
        CUDA_ALLOC(kN_to_kS_ratio, nPair, InitMode::NONE);
        CUDA_ALLOC(tensileStrength, nPair, InitMode::NONE);
        CUDA_ALLOC(cohesion, nPair, InitMode::NONE);
        CUDA_ALLOC(frictionCoeff, nPair, InitMode::NONE);
    }
    void release()
    {
        CUDA_FREE(maxContactGap); 
        CUDA_FREE(multiplier); 
        CUDA_FREE(elasticModulus); 
        CUDA_FREE(kN_to_kS_ratio);
        CUDA_FREE(tensileStrength); 
        CUDA_FREE(cohesion); 
        CUDA_FREE(frictionCoeff);
        num = 0;
    }
    void copy(const HostBondedContactModel& h)
    {
        release();
        alloc(h.num);
        cuda_copy(maxContactGap, h.maxContactGap.data(), num, CopyDir::H2D);
        cuda_copy(multiplier, h.multiplier.data(), num, CopyDir::H2D);
        cuda_copy(elasticModulus, h.elasticModulus.data(), num, CopyDir::H2D);
        cuda_copy(kN_to_kS_ratio, h.kN_to_kS_ratio.data(), num, CopyDir::H2D);
        cuda_copy(tensileStrength, h.tensileStrength.data(), num, CopyDir::H2D);
        cuda_copy(cohesion, h.cohesion.data(), num, CopyDir::H2D);
        cuda_copy(frictionCoeff, h.frictionCoeff.data(), num, CopyDir::H2D);
    }
};

struct ContactParameter
{
    MaterialProperty material;
    HertzianContactModel Hertzian;
    LinearContactModel Linear;
    BondedContactModel Bond;

    void release()
    {
        material.release();
        Hertzian.release();
        Linear.release();
        Bond.release();
    }
    void copy(const HostContactParameter& h)
    {
        material.copy(h.material);
        Hertzian.copy(h.Hertzian);
        Linear.copy(h.Linear);
        Bond.copy(h.Bond);
    }
    int __device__ getContactParameterIndex(int mA, int mB) const
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

struct BoundaryWall {
    int      numSpring{ 0 };
    int materialIndex{ 0 };
    double3* slidingSpring{ nullptr };
    double3* rollingSpring{ nullptr };
    double3* torsionSpring{ nullptr };

    void alloc(int n)
    {
        numSpring = n;
        CUDA_ALLOC(slidingSpring, n, InitMode::ZERO);
        CUDA_ALLOC(rollingSpring, n, InitMode::ZERO);
        CUDA_ALLOC(torsionSpring, n, InitMode::ZERO);
    }
    void release()
    {
        CUDA_FREE(slidingSpring);
        CUDA_FREE(rollingSpring);
        CUDA_FREE(torsionSpring);
        numSpring = 0;
    }
};

struct SpatialGrid {
    int      num{ 0 }; 
    double3  minBound{ make_double3(0,0,0) };
    double3  maxBound{ make_double3(1.,1.,1.) };
    double3  cellSize{ make_double3(1.,1.,1.) };
    int3     gridSize{ make_int3(1, 1, 1) }; // x * y * z + 1 = num

    int* cellStart{ nullptr };
    int* cellEnd{ nullptr };

    void alloc(int nCell)
    {
        num = nCell;
        CUDA_ALLOC(cellStart, nCell, InitMode::NEG_ONE);
        CUDA_ALLOC(cellEnd, nCell, InitMode::NEG_ONE);
    }
    void release()
    {
        CUDA_FREE(cellStart); 
        CUDA_FREE(cellEnd);
        num = 0;
    }
    void copy(const HostSpatialGrid& h)
    {
        release();
        alloc(h.num);
        minBound = h.minBound;
        maxBound = h.maxBound;
        cellSize = h.cellSize;
        gridSize = h.gridSize;
    }
    void resetCellStartEnd()
    {
        CUDA_CHECK(cudaMemset(cellStart, 0xFFFFFFFF, num * sizeof(int)));
        CUDA_CHECK(cudaMemset(cellEnd, 0xFFFFFFFF, num * sizeof(int)));
    }
};

struct DeviceData
{
    Sphere              spheres;
    SPH                 SPHParticles;
    Clump               clumps;
    TriangleWall        triangleWalls;
    BasicInteraction    sphSphInteract;
    BondedInteraction   sphSphBondedInteract;
    BasicInteraction    faceSphInteract;
    BasicInteraction    edgeSphInteract;
    BasicInteraction    vertexSphInteract;
    BoundaryWall  boundaryWallX;
    BoundaryWall  boundaryWallY;
    BoundaryWall  boundaryWallZ;
    ContactParameter    contactPara;
    SpatialGrid         spatialGrids;

    void copyFromHost(const HostData& h)
    {
        spheres.copy(h.spheres);
        SPHParticles.copy(h.SPHParticles);
        clumps.copy(h.clumps);
        triangleWalls.copy(h.triangleWalls);
        sphSphInteract.copy(h.sphSphInteract);
        sphSphBondedInteract.copy(h.sphSphBondedInteract);
        faceSphInteract.copy(h.faceSphInteract);
        edgeSphInteract.copy(h.edgeSphInteract);
        vertexSphInteract.copy(h.vertexSphInteract);
        contactPara.copy(h.contactPara);
        spatialGrids.copy(h.spatialGrids);
    }

    void release()
    {
        spheres.release();
        SPHParticles.release();
        clumps.release();
        triangleWalls.release();
        sphSphInteract.release();
        sphSphBondedInteract.release();
        faceSphInteract.release();
        edgeSphInteract.release();
        vertexSphInteract.release();
        contactPara.release();
        spatialGrids.release();

        boundaryWallX.release();
        boundaryWallY.release();
        boundaryWallZ.release();
    }
};