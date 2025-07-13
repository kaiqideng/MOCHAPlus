#include "NeighborSearch.cuh"

inline void computeGPUParameter(int& gridSize, int& blockSize,
    int nElements,
    int maxThreadsPerBlock)
{
    if (nElements == 0)
    {
        gridSize = int(1);
        blockSize = int(1);
        return;
    }
    blockSize = maxThreadsPerBlock < nElements ? maxThreadsPerBlock : nElements;
    gridSize = (nElements + blockSize - 1) / blockSize;
}

void sortKeyValuePairs(int* keys, int* values, int num)
{
    if (num < 1) return;
    thrust::sort_by_key(thrust::device_ptr<int>(keys),
        thrust::device_ptr<int>(keys + num),
        thrust::device_ptr<int>(values));
}

void inclusiveScan(int* prefixSum, int* count, int num)
{
    if (num < 1) return;
    thrust::inclusive_scan(thrust::device_ptr<int>(count),
        thrust::device_ptr<int>(count + num),
        thrust::device_ptr<int>(prefixSum));
}

__global__ void calculateParticleHash(Sphere sph,
    SpatialGrid SG)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sph.num) return;
    double3 pos = sph.state.positions[idx];
    if (SG.minBound.x <= pos.x && pos.x < SG.maxBound.x &&
        SG.minBound.y <= pos.y && pos.y < SG.maxBound.y &&
        SG.minBound.z <= pos.z && pos.z < SG.maxBound.z)
    {
        int3 gridPosition = calculateGridPosition(pos, SG.minBound, SG.cellSize);
        sph.hash.value[idx] = calculateHash(gridPosition, SG.gridSize);
    }
    else
    {
        sph.hash.value[idx] = SG.num - 1;
    }
}

__global__ void setInitialIndices(int* initialIndices,
    int numObjects)
{
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    initialIndices[indices] = indices;
}

__global__ void setHashAux(int* hashAux,
    int* hash,
    int numObjects)
{
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    if (indices == 0) hashAux[0] = hash[numObjects - 1];
    if (indices > 0)  hashAux[indices] = hash[indices - 1];
}

__global__ void findStartAndEnd(int* start, int* end,
    int* hash,
    int* hashAux,
    int numObjects)
{
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    if (indices == 0 || hash[indices] != hashAux[indices])
    {
        start[hash[indices]] = indices;
        end[hashAux[indices]] = indices;
    }
    if (indices == numObjects - 1) end[hash[indices]] = numObjects;
}

void buildHashSpans(int* start, int* end, int* sortedIndexes, int* hash, int* hashAux,
    int numObjects,
    int maxThreadsPerBlock)
{
    if (numObjects < 1) return;

    int grid = 1, block = 1;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);

    setInitialIndices << <grid, block >> > (sortedIndexes, numObjects);
    //cudaDeviceSynchronize();

    sortKeyValuePairs(hash, sortedIndexes, numObjects);

    setHashAux << <grid, block >> > (hashAux, hash, numObjects);
    //cudaDeviceSynchronize();

    findStartAndEnd << <grid, block >> > (start, end, hash, hashAux, numObjects);
    //cudaDeviceSynchronize();
}

__global__ void setSphereSphereInteractions(BasicInteraction I, Sphere sph,
    SpatialGrid SG,
    ContactParameter CP,
    int flag)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= sph.num)  return;

    sph.neighbor.count[idxA] = 0;
    if (flag == 0) sph.neighbor.prefixSum[idxA] = 0;
    int count = 0;
    int base = (idxA == 0 ? 0 : sph.neighbor.prefixSum[idxA - 1]);
    double3 posA = sph.state.positions[idxA];
    double radA = sph.radii[idxA];
    int3 gridPositionA = calculateGridPosition(posA, SG.minBound, SG.cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, SG.gridSize);
                if (hashB < 0 || hashB >= SG.num)
                {
                    continue;
                }
                int startIndex = SG.cellStart[hashB];
                int endIndex = SG.cellEnd[hashB];
                if (startIndex == 0xFFFFFFFF)
                {
                    continue;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = sph.hash.index[i];
                    if (idxA >= idxB || 
                        (sph.clumpIndex[idxA] >= 0 && sph.clumpIndex[idxB] >= 0 && sph.clumpIndex[idxA] == sph.clumpIndex[idxB]) ||
                        (sph.state.inverseMass[idxA] == 0 && sph.state.inverseMass[idxB] == 0))
                    {
                        continue;
                    }
                    double3 posB = sph.state.positions[idxB];
                    double radB = sph.radii[idxB];
                    double sphereOverlap = radA + radB - length(posA - posB);
                    int iMA = sph.materialIndex[idxA];
                    int iMB = sph.materialIndex[idxB];
                    int iCP = CP.getContactParameterIndex(iMA, iMB);
                    double maxContactGap = CP.Bond.maxContactGap[iCP];
                    if (sphereOverlap + maxContactGap >= 0.)
                    {
                        if (flag == 0)
                        {
                            count++;
                        }
                        else
                        {
                            int offset = atomicAdd(&sph.neighbor.count[idxA], 1);
                            int posWrite = base + offset;
                            I.objectPointed[posWrite] = idxA;
                            I.objectPointing[posWrite] = idxB;
                            I.normalOverlap[posWrite] = sphereOverlap;
							double3 contactNormal = normalize(posA - posB);
                            I.contactNormal[posWrite] = contactNormal;
                            I.contactPoint[posWrite] = posB + (radB - 0.5 * sphereOverlap) * contactNormal;
                            I.slidingSpring[posWrite] = make_double3(0, 0, 0);
                            I.rollingSpring[posWrite] = make_double3(0, 0, 0);
                            I.torsionSpring[posWrite] = make_double3(0, 0, 0);
                            I.contactForce[posWrite] = make_double3(0, 0, 0);
                            I.contactTorque[posWrite] = make_double3(0, 0, 0);
                            if (sph.sphereRange.start[idxB] != 0xFFFFFFFF)
                            {
                                for (int j = sph.sphereRange.start[idxB]; j < sph.sphereRange.end[idxB]; j++)
                                {
                                    int iContactPrev = I.hash.index[j];
                                    int idxAPrev = I.prev.objectPointed[iContactPrev];
                                    if (idxA == idxAPrev)
                                    {
                                        I.slidingSpring[posWrite] = I.prev.slidingSpring[iContactPrev];
                                        I.rollingSpring[posWrite] = I.prev.rollingSpring[iContactPrev];
                                        I.torsionSpring[posWrite] = I.prev.torsionSpring[iContactPrev];
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (flag == 0)
    {
        sph.neighbor.count[idxA] = count;
    }
}

__global__ void setFaceSphereInteractions(BasicInteraction faceSphI, TriangleWall TW,
    BasicInteraction edgeSphI,
    BasicInteraction vertexSphI,
    Sphere sph,
    SpatialGrid SG,
    int flag)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= TW.face.num)  return;

    double3 vA = TW.vertex.positions[TW.face.vAIndex[idxA]];
    double3 vB = TW.vertex.positions[TW.face.vBIndex[idxA]];
    double3 vC = TW.vertex.positions[TW.face.vCIndex[idxA]];
    int iw = TW.face.face2Wall[idxA];
    quaternion oriW = TW.state.orientations[iw];
    double3 posW = TW.state.positions[iw];
    vA = rotateVectorByQuaternion(oriW, vA) + posW;
    vB = rotateVectorByQuaternion(oriW, vB) + posW;
    vC = rotateVectorByQuaternion(oriW, vC) + posW;
    TW.face.neighbor.count[idxA] = 0;
    int count = 0;
    int base = (idxA == 0 ? 0 : TW.face.neighbor.prefixSum[idxA - 1]);
    int3 minSearch = make_int3(0, 0, 0);
    int3 maxSearch = make_int3(0, 0, 0);
    initializeFaceSearchRange(minSearch, maxSearch, SG.minBound, SG.cellSize, vA, vB, vC);
    fixSearchRange(minSearch, maxSearch, SG.gridSize);

    for (int z = minSearch.z; z <= maxSearch.z; z++)
    {
        for (int y = minSearch.y; y <= maxSearch.y; y++)
        {
            for (int x = minSearch.x; x <= maxSearch.x; x++)
            {
                int hashB = calculateHash(make_int3(x, y, z), SG.gridSize);
                int startIndex = SG.cellStart[static_cast<size_t>(hashB)];
                int endIndex = SG.cellEnd[static_cast<size_t>(hashB)];
                if (startIndex == 0xFFFFFFFF)
                {
                    continue;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = sph.hash.index[i];
                    double3 posB = sph.state.positions[idxB];
                    double radB = sph.radii[idxB];
                    double normalOverlap = 0;
                    double3 contactNormal = make_double3(0., 0., 0.), contactPoint = make_double3(0., 0., 0.);
                    bool faceContact = getFaceSphereContactInfo(normalOverlap, contactNormal, contactPoint, vA, vB, vC, radB, posB);
                    if (faceContact)
                    {
                        if (flag == 0)
                        {
                            count++;
                        }
                        else
                        {
                            int offset = atomicAdd(&TW.face.neighbor.count[idxA], 1);
                            int posWrite = base + offset;
                            faceSphI.objectPointed[posWrite] = idxA;
                            faceSphI.objectPointing[posWrite] = idxB;
                            faceSphI.normalOverlap[posWrite] = normalOverlap;
                            faceSphI.contactNormal[posWrite] = contactNormal;
                            faceSphI.contactPoint[posWrite] = contactPoint;
                            faceSphI.contactForce[posWrite] = make_double3(0, 0, 0);
                            faceSphI.contactTorque[posWrite] = make_double3(0, 0, 0);
							faceSphI.slidingSpring[posWrite] = make_double3(0, 0, 0);
							faceSphI.rollingSpring[posWrite] = make_double3(0, 0, 0);
							faceSphI.torsionSpring[posWrite] = make_double3(0, 0, 0);
                            bool set = setFace2SphereSpring(faceSphI, posWrite, idxA, idxB, contactNormal, TW.face.face2Wall, sph.faceRange);
                            if (set) continue;
                            set = setFaceEdgeVertex2SphereSpring(faceSphI, posWrite, idxB, iw, contactNormal, TW.edge.edge2Wall, sph.edgeRange, edgeSphI);
                            if (set) continue;
                            set = setFaceEdgeVertex2SphereSpring(faceSphI, posWrite, idxB, iw, contactNormal, TW.vertex.vertex2Wall, sph.vertexRange, vertexSphI);
                        }
                    }
                }
            }
        }
    }
    if (flag == 0)
    {
        TW.face.neighbor.count[idxA] = count;
    }
}

__global__ void setEdgeSphereInteractions(BasicInteraction edgeSphI, TriangleWall TW,
    BasicInteraction faceSphI,
    BasicInteraction vertexSphI,
    Sphere sph,
    SpatialGrid SG,
    int flag)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= TW.edge.num)  return;

    double3 vA = TW.vertex.positions[TW.edge.vAIndex[idxA]];
    double3 vB = TW.vertex.positions[TW.edge.vBIndex[idxA]];
    int iw = TW.edge.edge2Wall[idxA];
    quaternion oriW = TW.state.orientations[iw];
    double3 posW = TW.state.positions[iw];
    vA = rotateVectorByQuaternion(oriW, vA) + posW;
    vB = rotateVectorByQuaternion(oriW, vB) + posW;
    int edge2FaceStart = (idxA == 0 ? 0 : TW.edge.facePrefixSum[idxA - 1]);
    int edge2FaceEnd = TW.edge.facePrefixSum[idxA];
    TW.edge.neighbor.count[idxA] = 0;
    int count = 0;
    int base = (idxA == 0 ? 0 : TW.edge.neighbor.prefixSum[idxA - 1]);
    int3 minSearch = make_int3(0, 0, 0);
    int3 maxSearch = make_int3(0, 0, 0);
    initializeEdgeSearchRange(minSearch, maxSearch, SG.minBound, SG.cellSize, vA, vB);
    fixSearchRange(minSearch, maxSearch, SG.gridSize);
    for (int z = minSearch.z; z <= maxSearch.z; z++)
    {
        for (int y = minSearch.y; y <= maxSearch.y; y++)
        {
            for (int x = minSearch.x; x <= maxSearch.x; x++)
            {
                int hashB = calculateHash(make_int3(x, y, z), SG.gridSize);
                int startIndex = SG.cellStart[static_cast<size_t>(hashB)];
                int endIndex = SG.cellEnd[static_cast<size_t>(hashB)];
                if (startIndex == 0xFFFFFFFF)
                {
                    continue;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = sph.hash.index[i];
                    double3 posB = sph.state.positions[idxB];
                    double radB = sph.radii[idxB];
                    double normalOverlap = 0;
                    double3 contactNormal = make_double3(0., 0., 0.), contactPoint = make_double3(0., 0., 0.);
                    bool edgeContact = getEdgeSphereContactInfo(normalOverlap, contactNormal, contactPoint,
                        vA,
                        vB,
                        radB,
                        posB);
                    if (edgeContact)
                    {
                        for (int j = edge2FaceStart; j < edge2FaceEnd; j++)
                        {
                            int faceIndex = TW.edge.edge2Face[j];
                            int faceNeighborStart = (faceIndex == 0 ? 0 : TW.face.neighbor.prefixSum[faceIndex - 1]);
                            int faceNeighborEnd = TW.face.neighbor.prefixSum[faceIndex];
                            for (int k = faceNeighborStart; k < faceNeighborEnd; k++)
                            {
                                if (idxB == faceSphI.objectPointing[k])
                                {
                                    goto nextParticle;
                                }
                            }
                        }
                        if (flag == 0)
                        {
                            count++;
                        }
                        else
                        {
                            int offset = atomicAdd(&TW.edge.neighbor.count[idxA], 1);
                            int posWrite = base + offset;
                            edgeSphI.objectPointed[posWrite] = idxA;
                            edgeSphI.objectPointing[posWrite] = idxB;
                            edgeSphI.normalOverlap[posWrite] = normalOverlap;
                            edgeSphI.contactNormal[posWrite] = contactNormal;
                            edgeSphI.contactPoint[posWrite] = contactPoint;
                            edgeSphI.contactForce[posWrite] = make_double3(0, 0, 0);
                            edgeSphI.contactTorque[posWrite] = make_double3(0, 0, 0);
							edgeSphI.slidingSpring[posWrite] = make_double3(0, 0, 0);
							edgeSphI.rollingSpring[posWrite] = make_double3(0, 0, 0);
							edgeSphI.torsionSpring[posWrite] = make_double3(0, 0, 0);
                            bool set = setEdge2SphereSpring(edgeSphI, posWrite, idxA, idxB, contactNormal, TW.edge.edge2Wall, sph.edgeRange);
                            if (set) continue;
                            set = setFaceEdgeVertex2SphereSpring(edgeSphI, posWrite, idxB, iw, contactNormal, TW.face.face2Wall, sph.faceRange, faceSphI);
                            if (set) continue;
                            set = setFaceEdgeVertex2SphereSpring(edgeSphI, posWrite, idxB, iw, contactNormal, TW.vertex.vertex2Wall, sph.vertexRange, vertexSphI);
                        }
                    }
                nextParticle:;
                }
            }
        }
    }
    if (flag == 0)
    {
        TW.edge.neighbor.count[idxA] = count;
    }
}

__global__ void setVertexSphereInteractions(BasicInteraction vertexSphI, TriangleWall TW,
    BasicInteraction faceSphI,
    BasicInteraction edgeSphI,
    Sphere sph,
    SpatialGrid SG,
    int flag)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= TW.vertex.num)  return;

    double3 v = TW.vertex.positions[idxA];
    int iw = TW.vertex.vertex2Wall[idxA];
    v = rotateVectorByQuaternion(TW.state.orientations[iw], v) + TW.state.positions[iw];
    int vertex2FaceStart = (idxA == 0 ? 0 : TW.vertex.facePrefixSum[idxA - 1]);
    int vertex2FaceEnd = TW.vertex.facePrefixSum[idxA];
    int vertex2EdgeStart = (idxA == 0 ? 0 : TW.vertex.edgePrefixSum[idxA - 1]);
    int vertex2EdgeEnd = TW.vertex.edgePrefixSum[idxA];
    TW.vertex.neighbor.count[idxA] = 0;
    int count = 0;
    int base = (idxA == 0 ? 0 : TW.vertex.neighbor.prefixSum[idxA - 1]);
    int3 minSearch = calculateGridPosition(v, SG.minBound, SG.cellSize);
    int3 maxSearch = calculateGridPosition(v, SG.minBound, SG.cellSize);
    fixSearchRange(minSearch, maxSearch, SG.gridSize);
    for (int z = minSearch.z; z <= maxSearch.z; z++)
    {
        for (int y = minSearch.y; y <= maxSearch.y; y++)
        {
            for (int x = minSearch.x; x <= maxSearch.x; x++)
            {
                int hashB = calculateHash(make_int3(x, y, z), SG.gridSize);
                int startIndex = SG.cellStart[static_cast<size_t>(hashB)];
                int endIndex = SG.cellEnd[static_cast<size_t>(hashB)];
                if (startIndex == 0xFFFFFFFF)
                {
                    continue;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = sph.hash.index[i];
                    double3 posB = sph.state.positions[idxB];
                    double radB = sph.radii[idxB];
                    double normalOverlap = 0;
                    double3 contactNormal = make_double3(0., 0., 0.), contactPoint = make_double3(0., 0., 0.);
                    bool vertexContact = getVertexSphereContactInfo(normalOverlap, contactNormal, contactPoint, v, radB, posB);
                    if (vertexContact)
                    {
                        for (int j = vertex2FaceStart; j < vertex2FaceEnd; j++)
                        {
                            int faceIndex = TW.vertex.vertex2Face[j];
                            int faceNeighborStart = (faceIndex == 0 ? 0 : TW.face.neighbor.prefixSum[faceIndex - 1]);
                            int faceNeighborEnd = TW.face.neighbor.prefixSum[faceIndex];
                            for (int k = faceNeighborStart; k < faceNeighborEnd; k++)
                            {
                                if (idxB == faceSphI.objectPointing[k])
                                {
                                    goto nextParticle;
                                }
                            }
                        }
                        for (int j = vertex2EdgeStart; j < vertex2EdgeEnd; j++)
                        {
                            int edgeIndex = TW.vertex.vertex2Edge[j];
                            int edgeNeighborStart = (edgeIndex == 0 ? 0 : TW.edge.neighbor.prefixSum[edgeIndex - 1]);
                            int edgeNeighborEnd = TW.edge.neighbor.prefixSum[edgeIndex];
                            for (int k = edgeNeighborStart; k < edgeNeighborEnd; k++)
                            {
                                if (idxB == edgeSphI.objectPointing[k])
                                {
                                    goto nextParticle;
                                }

                            }
                        }

                        if (flag == 0)
                        {
                            count++;
                        }
                        else
                        {
                            int offset = atomicAdd(&TW.vertex.neighbor.count[idxA], 1);
                            int posWrite = base + offset;
                            vertexSphI.objectPointed[posWrite] = idxA;
                            vertexSphI.objectPointing[posWrite] = idxB;
                            vertexSphI.normalOverlap[posWrite] = normalOverlap;
                            vertexSphI.contactNormal[posWrite] = contactNormal;
                            vertexSphI.contactPoint[posWrite] = contactPoint;
                            vertexSphI.contactForce[posWrite] = make_double3(0, 0, 0);
                            vertexSphI.contactTorque[posWrite] = make_double3(0, 0, 0);
							vertexSphI.slidingSpring[posWrite] = make_double3(0, 0, 0);
							vertexSphI.rollingSpring[posWrite] = make_double3(0, 0, 0);
							vertexSphI.torsionSpring[posWrite] = make_double3(0, 0, 0);
                            bool set = setFaceEdgeVertex2SphereSpring(vertexSphI, posWrite, idxB, iw, contactNormal, TW.face.face2Wall, sph.faceRange, faceSphI);
                            if (set) continue;
                            set = setFaceEdgeVertex2SphereSpring(vertexSphI, posWrite, idxB, iw, contactNormal, TW.edge.edge2Wall, sph.edgeRange, edgeSphI);
                        }
                    }
                nextParticle:;
                }
            }
        }
    }
    if (flag == 0)
    {
        TW.vertex.neighbor.count[idxA] = count;
    }
}

__global__ void setSphereSphereBondedInteractions(BondedInteraction bondedI,
    BasicInteraction sphSphI,
    Sphere sph,
    ContactParameter CP)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sphSphI.num) return;

    int idxA = sphSphI.objectPointed[idx];
    int idxB = sphSphI.objectPointing[idx];
    bondedI.objectPointed[idx] = idxA;
    bondedI.objectPointing[idx] = idxB;
    bondedI.contactNormal[idx] = sphSphI.contactNormal[idx];
    bondedI.contactPoint[idx] = sphSphI.contactPoint[idx];
    bondedI.isBonded[idx] = 0;
    bondedI.normalForce[idx] = 0.;
    bondedI.torsionTorque[idx] = 0.;
    bondedI.shearForce[idx] = make_double3(0., 0., 0.);
    bondedI.bendingTorque[idx] = make_double3(0., 0., 0.);
    int iCP = CP.getContactParameterIndex(sph.materialIndex[idxA], sph.materialIndex[idxB]);
    if (CP.Bond.elasticModulus[iCP] > 0. && 
        sph.bondClusterIndex[idxA] >= 0 &&
		sph.bondClusterIndex[idxB] >= 0 &&
        sph.bondClusterIndex[idxA] == sph.bondClusterIndex[idxB])
    {
        bondedI.isBonded[idx] = 1;
    }
}

void setBondedInteractions(DeviceData& d, int maxThreadsPerBlock)
{
    d.sphSphBondedInteract.setNumBonds(d.sphSphInteract.num);

    int grid = 1, block = 1;
    int numObjects = 0;

    numObjects = d.sphSphBondedInteract.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    setSphereSphereBondedInteractions << <grid, block >> > (d.sphSphBondedInteract,
        d.sphSphInteract,
        d.spheres,
        d.contactPara);
    //cudaDeviceSynchronize();
}

void updateSphereGridHash(Sphere& sph, SpatialGrid& SG, int maxThreadsPerBlock)
{
    int grid = 1, block = 1;
    int numObjects = 0;

    SG.resetCellStartEnd();

    sph.hash.reset(sph.num);

    numObjects = sph.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    calculateParticleHash << <grid, block >> > (sph, SG);
    //cudaDeviceSynchronize();
    buildHashSpans(SG.cellStart, SG.cellEnd, sph.hash.index, sph.hash.value, sph.hash.aux,
        sph.num,
        maxThreadsPerBlock);
}

void triangleWallNeighborSearch(DeviceData& d, int maxThreadsPerBlock)
{
    int grid = 1, block = 1;
    int numObjects = 0;
    //Face
    d.faceSphInteract.copy2Prev();
    numObjects = d.triangleWalls.face.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    for (int flag = 0; flag < 2; flag++)
    {
        setFaceSphereInteractions << <grid, block >> > (d.faceSphInteract, d.triangleWalls,
            d.edgeSphInteract,
            d.vertexSphInteract,
            d.spheres,
            d.spatialGrids,
            flag);
        //cudaDeviceSynchronize();
        if (flag == 0)
        {
            int nf = 0;
            inclusiveScan(d.triangleWalls.face.neighbor.prefixSum, d.triangleWalls.face.neighbor.count, d.triangleWalls.face.num);
            cudaMemcpy(&nf, d.triangleWalls.face.neighbor.prefixSum + d.triangleWalls.face.num - 1, sizeof(int), cudaMemcpyDeviceToHost);
            d.faceSphInteract.setNum(nf);
        }
    }
    //EDGE
    d.edgeSphInteract.copy2Prev();
    numObjects = d.triangleWalls.edge.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    for (int flag = 0; flag < 2; flag++)
    {
        setEdgeSphereInteractions << <grid, block >> > (d.edgeSphInteract, d.triangleWalls,
            d.faceSphInteract,
            d.vertexSphInteract,
            d.spheres,
            d.spatialGrids,
            flag);
        //cudaDeviceSynchronize();
        if (flag == 0)
        {
            int ne = 0;
            inclusiveScan(d.triangleWalls.edge.neighbor.prefixSum, d.triangleWalls.edge.neighbor.count, d.triangleWalls.edge.num);
            cudaMemcpy(&ne, d.triangleWalls.edge.neighbor.prefixSum + d.triangleWalls.edge.num - 1, sizeof(int), cudaMemcpyDeviceToHost);
            d.edgeSphInteract.setNum(ne);
        }
    }
    //VERTEX
    d.vertexSphInteract.copy2Prev();
    numObjects = d.triangleWalls.vertex.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    for (int flag = 0; flag < 2; flag++)
    {
        setVertexSphereInteractions << <grid, block >> > (d.vertexSphInteract, d.triangleWalls,
            d.faceSphInteract,
            d.edgeSphInteract,
            d.spheres,
            d.spatialGrids,
            flag);
        //cudaDeviceSynchronize();
        if (flag == 0)
        {
            int nv = 0;
            inclusiveScan(d.triangleWalls.vertex.neighbor.prefixSum, d.triangleWalls.vertex.neighbor.count, d.triangleWalls.vertex.num);
            cudaMemcpy(&nv, d.triangleWalls.vertex.neighbor.prefixSum + d.triangleWalls.vertex.num - 1, sizeof(int), cudaMemcpyDeviceToHost);
            d.vertexSphInteract.setNum(nv);
        }
    }
	//Calculate the hash for face, edge and vertex interactions
    d.spheres.faceRange.reset(d.spheres.num);
    d.faceSphInteract.hash.reset(d.faceSphInteract.capacity);
    d.faceSphInteract.setHash();
    buildHashSpans(d.spheres.faceRange.start, d.spheres.faceRange.end, d.faceSphInteract.hash.index, d.faceSphInteract.hash.value, d.faceSphInteract.hash.aux,
        d.faceSphInteract.num,
        maxThreadsPerBlock);

    d.spheres.edgeRange.reset(d.spheres.num);
    d.edgeSphInteract.hash.reset(d.edgeSphInteract.capacity);
    d.edgeSphInteract.setHash();
    buildHashSpans(d.spheres.edgeRange.start, d.spheres.edgeRange.end, d.edgeSphInteract.hash.index, d.edgeSphInteract.hash.value, d.edgeSphInteract.hash.aux,
        d.edgeSphInteract.num,
        maxThreadsPerBlock);

    d.spheres.vertexRange.reset(d.spheres.num);
    d.vertexSphInteract.hash.reset(d.vertexSphInteract.capacity);
    d.vertexSphInteract.setHash();
    buildHashSpans(d.spheres.vertexRange.start, d.spheres.vertexRange.end, d.vertexSphInteract.hash.index, d.vertexSphInteract.hash.value, d.vertexSphInteract.hash.aux,
        d.vertexSphInteract.num,
        maxThreadsPerBlock);
}

void neighborSearch(DeviceData& d, int maxThreadsPerBlock, int iStep)
{
    int grid = 1, block = 1;
    int numObjects = 0;

    updateSphereGridHash(d.spheres, d.spatialGrids, maxThreadsPerBlock);

	if (d.triangleWalls.num > 0) triangleWallNeighborSearch(d, maxThreadsPerBlock);

    if (iStep % 30 == 0)
    {
        d.sphSphInteract.copy2Prev();
        numObjects = d.spheres.num;
        computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
        for (int flag = 0; flag < 2; flag++)
        {
            setSphereSphereInteractions << <grid, block >> > (d.sphSphInteract, d.spheres,
                d.spatialGrids,
                d.contactPara,
                flag);
            //cudaDeviceSynchronize();
            if (flag == 0)
            {
                int n = 0;
                inclusiveScan(d.spheres.neighbor.prefixSum, d.spheres.neighbor.count, d.spheres.num);
                cudaMemcpy(&n, d.spheres.neighbor.prefixSum + d.spheres.num - 1, sizeof(int), cudaMemcpyDeviceToHost);
                d.sphSphInteract.setNum(n);
            }
        }
        d.spheres.sphereRange.reset(d.spheres.num);
        d.sphSphInteract.hash.reset(d.sphSphInteract.capacity);
        d.sphSphInteract.setHash();
        buildHashSpans(d.spheres.sphereRange.start, d.spheres.sphereRange.end, d.sphSphInteract.hash.index, d.sphSphInteract.hash.value, d.sphSphInteract.hash.aux,
            d.sphSphInteract.num,
            maxThreadsPerBlock);
    }
}