#pragma once
#include "Integrate.cuh"
#include "HostDataValidator.h"
#include "input.h"
#include "output.h"
#include <chrono>
#include <cstdio>

void setGPUDevice(int deviceID);

template<typename F>
double timeHostFunc(F func, cudaStream_t stream = 0)
{
    cudaStreamSynchronize(stream);

    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

class Solver
{
public:
    Solver()
    {
        domainOrigin = make_double3(0., 0., 0.);
        domainSize = make_double3(1., 1., 1.);
        gravity = make_double3(0., 0., 0.);
        addBoundaryWallX = false;
        addBoundaryWallY = false;
        addBoundaryWallZ = false;
        boundaryWallXMaterialIndex = 0;
        boundaryWallYMaterialIndex = 0;
        boundaryWallZMaterialIndex = 0;
        timeMax = 1.;
        timeStep = 1.;

        deviceIndex = 0;
        maxThreadsPerBlock = 256;

        stepMax = 1;
        saveAccount = 1;
        iStep = 1;
        double currentTime = 0.;
        computeTime_neighborSearch = 0.;
        computeTime_contact = 0.;
        computeTime_integration = 0.;
    };

    ~Solver()
    {
        release();
    };

    const HostSphere& getHostSphere()
    {
        dev.spheres.uploadState(hos.spheres);
        return hos.spheres;
    };

    const HostTriangleWall& getHostTriangleWall()
    {
        dev.triangleWalls.uploadState(hos.triangleWalls);
        return hos.triangleWalls;
    };

    const HostClump& getHostClump()
    {
        dev.clumps.uploadState(hos.clumps);
        return hos.clumps;
    }

    const HostSPH& getHostSPH()
    {
        dev.SPHParticles.upload(hos.SPHParticles);
        return hos.SPHParticles;
    }

    const HostBasicInteraction& getHostSphereSphereInteraction()
    {
        dev.sphSphInteract.upload(hos.sphSphInteract);
        return hos.sphSphInteract;
    };

    const HostBasicInteraction& getHostFaceSphereInteraction()
    {
        dev.faceSphInteract.upload(hos.faceSphInteract);
        return hos.faceSphInteract;
    };

    const HostBasicInteraction& getHostEdgeSphereInteraction()
    {
        dev.edgeSphInteract.upload(hos.edgeSphInteract);
        return hos.edgeSphInteract;
    };

    const HostBasicInteraction& getHostVertexSphereInteraction()
    {
        dev.vertexSphInteract.upload(hos.vertexSphInteract);
        return hos.vertexSphInteract;
    };

    const HostBondedInteraction& getHostBondedInteraction()
    {
        dev.sphSphBondedInteract.upload(hos.sphSphBondedInteract);
        return hos.sphSphBondedInteract;
    }

    void solve();

protected:
    void setDomainGravity(double3 origin, double3 size, double3 g)
    {
        domainOrigin = origin;
        if (size.x > 0. && size.y > 0. && size.z > 0.)domainSize = size;
        gravity = g;
    };

    void setBoundaryWallX(int materialIndex)
    {
        addBoundaryWallX = true;
        boundaryWallXMaterialIndex = materialIndex;
    }

    void setBoundaryWallY(int materialIndex)
    {
        addBoundaryWallY = true;
        boundaryWallYMaterialIndex = materialIndex;
    }

    void setBoundaryWallZ(int materialIndex)
    {
        addBoundaryWallZ = true;
        boundaryWallZMaterialIndex = materialIndex;
    }

    void setTimeMaxTimeStepPrintNumber(double tMax, double dt, int n)
    {
        if (tMax > 0.) timeMax = tMax;
        if (dt > 0.) timeStep = dt;
        int numSteps = (timeMax - currentTime) / timeStep + 1;
        if (numSteps > 0) stepMax = iStep + numSteps;
        saveAccount = numSteps / n;
        if (saveAccount < 1)  saveAccount = 1;
    };

    void setGPUParameter(int deviceID, int maxThread)
    {
        if (deviceID >= 0) deviceIndex = deviceID;
        if (maxThread > 0) maxThreadsPerBlock = maxThread;
    };

    const double& getTimeStep() const
    {
        return timeStep;
    }

    const double& getTimeMax() const
    {
        return timeMax;
    }

    const double& getCurrentTime() const
    {
        return currentTime;
    };

    const int& getCurrentStep() const
    {
        return iStep;
    };

    const int& getStepMax() const
    {
        return stepMax;
    };

    Sphere& getDeviceSphere()
    {
        return dev.spheres;
    };

    DynamicState& getDeviceSphereState()
    {
        return dev.spheres.state;
    };

    DynamicState& getDeviceTriangleWallState()
    {
        return dev.triangleWalls.state;
    };

    virtual void buildHostBasicInteraction()
    {
        if (hos.SPHParticles.num > 0)
        {
            hos.sphSphInteract = HostBasicInteraction(50 * hos.spheres.num);
        }
        else
        {
            hos.sphSphInteract = HostBasicInteraction(6 * hos.spheres.num);
        }
        if (hos.triangleWalls.num > 0)
        {
            hos.faceSphInteract = HostBasicInteraction(hos.spheres.num);
            hos.edgeSphInteract = HostBasicInteraction(hos.spheres.num);
            hos.vertexSphInteract = HostBasicInteraction(hos.triangleWalls.vertex.num);
        }
    };

    virtual void loadHostData(HostData& h) {};

    virtual void handleDataBeforeContact() {};

    virtual void handleDataAfterContact() {};

    virtual void outputData(int frame) {};

private:
    double3 domainOrigin;
    double3 domainSize;
    double3 gravity;
	bool addBoundaryWallX;
	bool addBoundaryWallY;
	bool addBoundaryWallZ;
	int boundaryWallXMaterialIndex;
	int boundaryWallYMaterialIndex;
	int boundaryWallZMaterialIndex;
    double timeMax;
    double timeStep;

    int deviceIndex;
    int maxThreadsPerBlock;

    int stepMax;
    int saveAccount;
    int iStep;
    double currentTime;
    double computeTime_neighborSearch;
    double computeTime_contact;
    double computeTime_integration;

    HostData hos;
    DeviceData dev;

    void buildHostSpatialGrid()
    {
        double maxSphereSize = *std::max_element(hos.spheres.radii.begin(), hos.spheres.radii.end()) * 2.;
        double maxContactGap = *std::max_element(hos.contactPara.Bond.maxContactGap.begin(), hos.contactPara.Bond.maxContactGap.end());
        double cellSizeOneDim = maxSphereSize + maxContactGap;
        hos.spatialGrids.minBound = domainOrigin;
        hos.spatialGrids.maxBound = domainOrigin + domainSize;
        hos.spatialGrids.gridSize.x = domainSize.x > cellSizeOneDim ? int(domainSize.x / cellSizeOneDim) : 1;
        hos.spatialGrids.gridSize.y = domainSize.y > cellSizeOneDim ? int(domainSize.y / cellSizeOneDim) : 1;
        hos.spatialGrids.gridSize.z = domainSize.z > cellSizeOneDim ? int(domainSize.z / cellSizeOneDim) : 1;
        hos.spatialGrids.cellSize.x = domainSize.x / double(hos.spatialGrids.gridSize.x);
        hos.spatialGrids.cellSize.y = domainSize.y / double(hos.spatialGrids.gridSize.y);
        hos.spatialGrids.cellSize.z = domainSize.z / double(hos.spatialGrids.gridSize.z);
        hos.spatialGrids.num = hos.spatialGrids.gridSize.x * hos.spatialGrids.gridSize.y * hos.spatialGrids.gridSize.z + 1;
    };

    bool buildDeviceData()
    {
        if (!validateHostData(hos)) return false;
        release();
        dev.copyFromHost(hos);
        if (addBoundaryWallX)
        {
            dev.boundaryWallX.alloc(hos.spheres.num);
            if (boundaryWallXMaterialIndex < hos.contactPara.material.num) 
                dev.boundaryWallX.materialIndex = boundaryWallXMaterialIndex;
        }
        if (addBoundaryWallY)
        {
            dev.boundaryWallY.alloc(hos.spheres.num);
            if (boundaryWallYMaterialIndex < hos.contactPara.material.num) 
                dev.boundaryWallY.materialIndex = boundaryWallYMaterialIndex;
        }
        if (addBoundaryWallZ)
        {
            dev.boundaryWallZ.alloc(hos.spheres.num);
            if (boundaryWallZMaterialIndex < hos.contactPara.material.num) 
                dev.boundaryWallZ.materialIndex = boundaryWallZMaterialIndex;
        }
        return true;
    };

    virtual void setBond(DeviceData& D, int maxThread) {};

    virtual void calculateParticleContactForceTorque(DeviceData& D, double dt, int maxThread, int i) {};

    void release()
    {
        dev.release();
    }
};

class DEMSolver : public Solver
{
public:
    DEMSolver() : Solver()
    {
    }

private:
    void setBond(DeviceData& D, int maxThread)override
    {
        setBondedInteractions(D, maxThread);
    }

    void calculateParticleContactForceTorque(DeviceData& D, double dt, int maxThread, int i)override
    {
		calculateContactForceTorqueDEM(D, dt, maxThread, i);
    }
};

class SPHSolver : public Solver
{
public:
    SPHSolver() : Solver()
    {
    }

private:
    void calculateParticleContactForceTorque(DeviceData& D, double dt, int maxThread, int i)override
    {
        calculateContactForceTorqueSPH(D, dt, maxThread, i);
    }
};

class DEMSPHSolver : public Solver
{
public:
    DEMSPHSolver() : Solver()
    {
    }

private:
    void setBond(DeviceData& D, int maxThread)override
    {
        setBondedInteractions(D, maxThread);
    }

    void calculateParticleContactForceTorque(DeviceData& D, double dt, int maxThread, int i)override
    {
        calculateContactForceTorqueDEM(D, dt, maxThread, i);
		calculateContactForceTorqueSPH(D, dt, maxThread, i);
        calculateContactForceTorqueDEMSPH(D, maxThread);
    }
};


