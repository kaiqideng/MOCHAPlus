#pragma once
#include "Integrate.cuh"
#include "HostDataValidator.h"
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

    void buildDeviceData(const HostData& h)
    {
        release();                      
        dev.copyFromHost(h);
    }

    void upload2Host()
    {
        dev.upload(hos);
    }

    virtual void loadHostData(HostData& h) {};

    virtual void handleDataBeforeContact(HostData& h, DeviceData& d) {};

    virtual void handleDataAfterContact(HostData& h, DeviceData& d) {};

    virtual void outputData(const HostData& h, int frame, int step) {};

	HostData& getHostData()
	{
		return hos;
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

    void release()
    {
        dev.release();
    }
};