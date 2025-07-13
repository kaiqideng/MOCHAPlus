#pragma once
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "DeviceStructs.h"
constexpr auto EPS_DOT = 1e-10;

void computeGPUParameter(int& grid, int& block, int nElements, int maxThreadsPerBlock);

void sortKeyValuePairs(int* keys, int* values, int num);

void inclusiveScan(int* prefixSum, int* count, int num);

__device__ __forceinline__ int3 calculateGridPosition(double3 position, double3 minBoundary, double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
        int((position.y - minBoundary.y) / cellSize.y),
        int((position.z - minBoundary.z) / cellSize.z));
}

__device__ __forceinline__ int calculateHash(int3 gridPosition, int3 gridSize)
{
    return gridPosition.z * gridSize.y * gridSize.x + gridPosition.y * gridSize.x + gridPosition.x;
}

__device__ __forceinline__ bool getFaceSphereContactInfo(double& normalOverlap, double3& contactNormal, double3& contactPoint,
    double3 vA,
    double3 vB,
    double3 vC,
    double rad,
    double3 pos)
{
    bool faceContact = false;
    double3 wallNormal = normalize(cross(vB - vA, vC - vB));
    contactNormal = -wallNormal;
    normalOverlap = rad - dot(pos - vA, wallNormal);
    if (normalOverlap > rad)
    {
        contactNormal = wallNormal;
        normalOverlap = rad - dot(pos - vA, -wallNormal);
    }
    contactPoint = pos + (rad - normalOverlap) * contactNormal;
    double3 AB = vB - vA;
    double3 BC = vC - vB;
    double3 CA = vA - vC;
    double3 Q = contactPoint;
    double3 AQ = Q - vA;
    double3 BQ = Q - vB;
    double3 CQ = Q - vC;
    double s = 0.5 * length(cross(AB, BC));
    double s1 = 0.5 * length(cross(AQ, BQ));
    double s2 = 0.5 * length(cross(BQ, CQ));
    double s3 = 0.5 * length(cross(CQ, AQ));
    if (normalOverlap >= 0.)
    {
        if (s1 + s2 < s && s1 + s3 < s && s2 + s3 < s)
        {
            faceContact = true;
        }
    }
    return faceContact;
}

__device__ __forceinline__ bool getEdgeSphereContactInfo(double& normalOverlap, double3& contactNormal, double3& contactPoint,
    double3 vA,
    double3 vB,
    double rad,
    double3 pos)
{
    bool edgeContact = false;
    double3 AB = vB - vA;
    double3 AP = pos - vA;
    double t1 = dot(AP, AB) / dot(AB, AB);
    contactPoint = vA + t1 * AB;
    double3 PQ = contactPoint - pos;
    contactNormal = normalize(PQ);
    normalOverlap = rad - length(PQ);
    if (normalOverlap >= 0.)
    {
        if (t1 > 0 && t1 < 1)
        {
            edgeContact = true;
        }
    }
    return edgeContact;
}

__device__ __forceinline__ bool getVertexSphereContactInfo(double& normalOverlap, double3& contactNormal, double3& contactPoint,
    double3 v,
    double rad,
    double3 pos)
{
    bool vertexContact = false;
    contactNormal = normalize(v - pos);
    normalOverlap = rad - length(v - pos);
    contactPoint = pos + (rad - normalOverlap) * contactNormal;
    if (normalOverlap >= 0.)
    {
        vertexContact = true;
    }
    return vertexContact;
}

__device__ __forceinline__ void initializeFaceSearchRange(int3& minSearch, int3& maxSearch,
    double3 minBound,
    double3 cellSize,
    double3 vA,
    double3 vB,
    double3 vC)
{
    int3 vAG = calculateGridPosition(vA, minBound, cellSize);
    int3 vBG = calculateGridPosition(vB, minBound, cellSize);
    int3 vCG = calculateGridPosition(vC, minBound, cellSize);

    minSearch.x = vAG.x < vBG.x ? vAG.x : vBG.x;
    minSearch.x = minSearch.x < vCG.x ? minSearch.x : vCG.x;
    minSearch.y = vAG.y < vBG.y ? vAG.y : vBG.y;
    minSearch.y = minSearch.y < vCG.y ? minSearch.y : vCG.y;
    minSearch.z = vAG.z < vBG.z ? vAG.z : vBG.z;
    minSearch.z = minSearch.z < vCG.z ? minSearch.z : vCG.z;

    maxSearch.x = vAG.x > vBG.x ? vAG.x : vBG.x;
    maxSearch.x = maxSearch.x > vCG.x ? maxSearch.x : vCG.x;
    maxSearch.y = vAG.y > vBG.y ? vAG.y : vBG.y;
    maxSearch.y = maxSearch.y > vCG.y ? maxSearch.y : vCG.y;
    maxSearch.z = vAG.z > vBG.z ? vAG.z : vBG.z;
    maxSearch.z = maxSearch.z > vCG.z ? maxSearch.z : vCG.z;
}

__device__ __forceinline__ void initializeEdgeSearchRange(int3& minSearch, int3& maxSearch,
    double3 minBound,
    double3 cellSize,
    double3 vA,
    double3 vB)
{
    int3 vAG = calculateGridPosition(vA, minBound, cellSize);
    int3 vBG = calculateGridPosition(vB, minBound, cellSize);

    minSearch.x = vAG.x < vBG.x ? vAG.x : vBG.x;
    minSearch.y = vAG.y < vBG.y ? vAG.y : vBG.y;
    minSearch.z = vAG.z < vBG.z ? vAG.z : vBG.z;

    maxSearch.x = vAG.x > vBG.x ? vAG.x : vBG.x;
    maxSearch.y = vAG.y > vBG.y ? vAG.y : vBG.y;
    maxSearch.z = vAG.z > vBG.z ? vAG.z : vBG.z;
}

__device__ __forceinline__ void fixSearchRange(int3& minSearch, int3& maxSearch,
    int3 gridSize)
{
    int3 newMinSearch = minSearch, newMaxSearch = maxSearch;
    newMinSearch.x = minSearch.x > 0 ? minSearch.x - 1 : 0;
    newMinSearch.y = minSearch.y > 0 ? minSearch.y - 1 : 0;
    newMinSearch.z = minSearch.z > 0 ? minSearch.z - 1 : 0;
    newMaxSearch.x = gridSize.x - 1 > maxSearch.x + 1 ? maxSearch.x + 1 : gridSize.x - 1;
    newMaxSearch.y = gridSize.y - 1 > maxSearch.y + 1 ? maxSearch.y + 1 : gridSize.y - 1;
    newMaxSearch.z = gridSize.z - 1 > maxSearch.z + 1 ? maxSearch.z + 1 : gridSize.z - 1;
    minSearch = newMinSearch;
    maxSearch = newMaxSearch;
}

__device__ __forceinline__ bool setFace2SphereSpring(BasicInteraction faceSphI, int posWrite, int faceIndex, int sphIndex, double3 contactNormal, int* face2Wall, InteractionRange faceRange)
{
    double tolerance = sin(5. * pi() / 180.);
    if (faceRange.start[sphIndex] != 0xFFFFFFFF)
    {
        for (int i = faceRange.start[sphIndex]; i < faceRange.end[sphIndex]; i++)
        {
            int iContactPrev = faceSphI.hash.index[i];
            int faceIndexPrev = faceSphI.prev.objectPointed[iContactPrev];
            if (faceIndex == faceIndexPrev)
            {
                faceSphI.slidingSpring[posWrite] = faceSphI.prev.slidingSpring[iContactPrev];
                faceSphI.rollingSpring[posWrite] = faceSphI.prev.rollingSpring[iContactPrev];
                faceSphI.torsionSpring[posWrite] = faceSphI.prev.torsionSpring[iContactPrev];
                return true;
            }
        }
        for (int i = faceRange.start[sphIndex]; i < faceRange.end[sphIndex]; i++)
        {
            int iContactPrev = faceSphI.hash.index[i];
            int faceIndexPrev = faceSphI.prev.objectPointed[iContactPrev];
            double3 contactNormalPrev = faceSphI.prev.contactNormal[iContactPrev];
            if (face2Wall[faceIndex] == face2Wall[faceIndexPrev] &&
                length(cross(contactNormal, contactNormalPrev)) < tolerance &&
                dot(contactNormal, contactNormalPrev) > 0.)
            {
                faceSphI.slidingSpring[posWrite] = faceSphI.prev.slidingSpring[iContactPrev];
                faceSphI.rollingSpring[posWrite] = faceSphI.prev.rollingSpring[iContactPrev];
                faceSphI.torsionSpring[posWrite] = faceSphI.prev.torsionSpring[iContactPrev];
                return true;
            }
        }
    }
    return false;
}

__device__ __forceinline__ bool setEdge2SphereSpring(BasicInteraction edgeSphI, int posWrite, int edgeIndex, int sphIndex, double3 contactNormal, int* edge2Wall, InteractionRange edgeRange)
{
    double tolerance = sin(5. * pi() / 180.);
    if (edgeRange.start[sphIndex] != 0xFFFFFFFF)
    {
        for (int i = edgeRange.start[sphIndex]; i < edgeRange.end[sphIndex]; i++)
        {
            int iContactPrev = edgeSphI.hash.index[i];
            int edgeIndexPrev = edgeSphI.prev.objectPointed[iContactPrev];
            if (edgeIndex == edgeIndexPrev)
            {
                edgeSphI.slidingSpring[posWrite] = edgeSphI.prev.slidingSpring[iContactPrev];
                edgeSphI.rollingSpring[posWrite] = edgeSphI.prev.rollingSpring[iContactPrev];
                edgeSphI.torsionSpring[posWrite] = edgeSphI.prev.torsionSpring[iContactPrev];
                return true;
            }
        }
        for (int i = edgeRange.start[sphIndex]; i < edgeRange.end[sphIndex]; i++)
        {
            int iContactPrev = edgeSphI.hash.index[i];
            int edgeIndexPrev = edgeSphI.prev.objectPointed[iContactPrev];
            double3 contactNormalPrev = edgeSphI.prev.contactNormal[iContactPrev];
            if (edge2Wall[edgeIndex] == edge2Wall[edgeIndexPrev] &&
                length(cross(contactNormal, contactNormalPrev)) < tolerance &&
                dot(contactNormal, contactNormalPrev) > 0.)
            {
                edgeSphI.slidingSpring[posWrite] = edgeSphI.prev.slidingSpring[iContactPrev];
                edgeSphI.rollingSpring[posWrite] = edgeSphI.prev.rollingSpring[iContactPrev];
                edgeSphI.torsionSpring[posWrite] = edgeSphI.prev.torsionSpring[iContactPrev];
                return true;
            }
        }
    }
    return false;
}

__device__ __forceinline__ bool setFaceEdgeVertex2SphereSpring(BasicInteraction elementSphI, int posWrite, int sphIndex, int wallIndex, double3 contactNormal, int* otherElement2Wall, InteractionRange otherElementSphRange, BasicInteraction otherElementSphI)
{
    double tolerance = sin(5. * pi() / 180.);
    if (otherElementSphRange.start[sphIndex] != 0xFFFFFFFF)
    {
        for (int i = otherElementSphRange.start[sphIndex]; i < otherElementSphRange.end[sphIndex]; i++)
        {
            int iContactPrev = otherElementSphI.hash.index[i];
            double3 contactNormalPrev = otherElementSphI.prev.contactNormal[iContactPrev];
            int otherElementIndexPrev = otherElementSphI.prev.objectPointed[iContactPrev];
            if (otherElement2Wall[otherElementIndexPrev] == wallIndex &&
                length(cross(contactNormal, contactNormalPrev)) < tolerance &&
                dot(contactNormal, contactNormalPrev) > 0.)
            {
                elementSphI.slidingSpring[posWrite] = otherElementSphI.prev.slidingSpring[iContactPrev];
                elementSphI.rollingSpring[posWrite] = otherElementSphI.prev.rollingSpring[iContactPrev];
                elementSphI.torsionSpring[posWrite] = otherElementSphI.prev.torsionSpring[iContactPrev];
                return true;
            }
        }
    }
    return false;
}

__global__ void calculateParticleHash(Sphere sph, SpatialGrid SG);

__global__ void setInitialIndices(int* initialIndices, int numObjects);

__global__ void setHashAux(int* hashAux, int* hash, int numObjects);

__global__ void findStartAndEnd(int* start, int* end, int* hash, int* hashAux, int numObjects);

void buildHashSpans(int* start, int* end, int* sortedIndexes, int* hash, int* hashAux, int numObjects, int maxThreadsPerBlock);

__global__ void setSphereSphereInteractions(BasicInteraction I, Sphere sph, SpatialGrid SG, ContactParameter CP, int flag);

__global__ void setFaceSphereInteractions(BasicInteraction faceSphI, TriangleWall TW, BasicInteraction edgeSphI, BasicInteraction vertexSphI, Sphere sph, SpatialGrid SG, int flag);

__global__ void setEdgeSphereInteractions(BasicInteraction edgeSphI, TriangleWall TW, BasicInteraction faceSphI, BasicInteraction vertexSphI, Sphere sph, SpatialGrid SG, int flag);

__global__ void setVertexSphereInteractions(BasicInteraction vertexSphI, TriangleWall TW, BasicInteraction faceSphI, BasicInteraction edgeSphI, Sphere sph, SpatialGrid SG, int flag);

__global__ void setSphereSphereBondedInteractions(BondedInteraction bondedI, BasicInteraction sphSphI, Sphere sph, ContactParameter CP);

void setBondedInteractions(DeviceData& d, int maxThreadsPerBlock);

void updateSphereGridHash(Sphere& sph, SpatialGrid& SG, int maxThreadsPerBlock);

void triangleWallNeighborSearch(DeviceData& d, int maxThreadsPerBlock);

void neighborSearch(DeviceData& d, int maxThreadsPerBlock, int iStep);