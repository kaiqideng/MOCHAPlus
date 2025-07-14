#pragma once
#include "Integrate.cuh"
#include "HostDataValidator.h"
#include "input.h"
#include "output.h"
#include <chrono>
#include <cstdio>

class DEMSolver 
{
public:
    
    double computeTime_neighborSearch;
    double computeTime_contact;
    double computeTime_integration;
    
    DEMSolver()
    {
        computeTime_neighborSearch = 0.;
        computeTime_contact = 0.;
        computeTime_integration = 0.;
    }

    ~DEMSolver()
    { 
        release(); 
    }

    void uploadSphereState()
    {
		dev.spheres.upload(hos.spheres);
    }

    void uploadSPHState()
    {
		dev.SPHParticles.upload(hos.SPHParticles);
    }

    void uploadClumpState()
    {
		dev.clumps.upload(hos.clumps);
    }

    void uploadTriangleWallState()
    {
		dev.triangleWalls.upload(hos.triangleWalls);
    }

    void uploadBasicInteraction()
    {
		dev.sphSphInteract.upload(hos.sphSphInteract);
		dev.faceSphInteract.upload(hos.faceSphInteract);
		dev.edgeSphInteract.upload(hos.edgeSphInteract);
		dev.vertexSphInteract.upload(hos.vertexSphInteract);
    }

	void uploadBondedInteraction()
	{
		dev.sphSphBondedInteract.upload(hos.sphSphBondedInteract);
	}

    void downloadSphereState()
    {
        dev.spheres.download(hos.spheres);
    }

	void downloadTriangleWallState()
	{
		dev.triangleWalls.download(hos.triangleWalls);
	}

	void resetSpatialGrid()
	{
		hos.buildSpatialGrid();
		if (!validateSpatialGrid(hos.spatialGrids)) return;
        dev.spatialGrids.copy(hos.spatialGrids);
	}

    virtual void loadHostData() {};

    virtual void handleDataBeforeContact() {};

    virtual void handleDataAfterContact() {};

    virtual void outputData(int frame, int step) {};

	HostData& getHostData()
	{
		return hos;
	}

    DeviceData& getDeviceData()
    {
        return dev;
    }

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

    void solve();

private:
    HostData hos;
    DeviceData dev;

    bool buildDeviceData()
    {
        if (!validateHostData(hos)) return false;
        release();
        dev.copyFromHost(hos);
        return true;
    }

    void release()
    {
        dev.release();
    }
};