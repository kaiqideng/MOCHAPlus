#include "Solver.h"

void setGPUDevice(int deviceID)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(deviceID);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Setting CUDA device failed!  Do you have a CUDA-capable GPU installed?");
    }
    else
    {
        std::cout << "Set CUDA device successfully. \n";
    }
}

void Solver::solve()
{
    loadHostData(hos);
	buildHostSpatialGrid();
    buildHostBasicInteraction();

    std::cout << "------ Simulation Parameters ------" << std::endl;
    std::cout << "       Domain Origin: " << domainOrigin.x << ", " << domainOrigin.y << ", " << domainOrigin.z << std::endl;
    std::cout << "       Domain Size: " << domainSize.x << ", " << domainSize.y << ", " << domainSize.z << std::endl;
    std::cout << "       Gravity: " << gravity.x << ", " << gravity.y << ", " << gravity.z << std::endl;
    std::cout << "       Time Step: " << timeStep << std::endl;
    std::cout << "       Time Max: " << timeMax << std::endl;
    std::cout << "       Number of Spheres: " << hos.spheres.num << std::endl;
    std::cout << "       Number of Triangle Walls: " << hos.triangleWalls.num << std::endl;
    if (hos.triangleWalls.num > 0)
    {
        std::cout << "       Number of Triangle Faces: " << hos.triangleWalls.face.num << std::endl;
        std::cout << "       Number of Triangle Edges: " << hos.triangleWalls.edge.num << std::endl;
        std::cout << "       Number of Triangle Vertices: " << hos.triangleWalls.vertex.num << std::endl;
    }
    std::cout << "       GPU device: " << deviceIndex << std::endl;

    setGPUDevice(deviceIndex);
    if (!buildDeviceData()) return;

    neighborSearch(dev, maxThreadsPerBlock, 0);
    this->setBond(dev, maxThreadsPerBlock);
    int iFrame = 0;
    outputData(iFrame);
    while (iStep <= stepMax)
    {
        computeTime_neighborSearch += timeHostFunc([&]() { neighborSearch(dev, maxThreadsPerBlock, iStep); });
        computeTime_integration += timeHostFunc([&]() { integrateBeforeContact(dev, gravity, timeStep, maxThreadsPerBlock); });
        handleDataBeforeContact();
        computeTime_contact += timeHostFunc([&]() { this->calculateParticleContactForceTorque(dev, timeStep, maxThreadsPerBlock, iStep); });
        handleDataAfterContact();
        computeTime_contact += timeHostFunc([&]() { accumulateForceTorque(dev, maxThreadsPerBlock);});
        computeTime_contact += timeHostFunc([&]() { this->boundaryForceTorque(dev, timeStep, maxThreadsPerBlock); });
        computeTime_integration += timeHostFunc([&]() { integrateAfterContact(dev, gravity, timeStep, maxThreadsPerBlock);});
        if (iStep % saveAccount == 0)
        {
            iFrame++;
            std::cout << "------ Frame: " << iFrame << " ------" << std::endl;
            std::cout << "       Simulation Time(s): " << currentTime << std::endl;
            std::cout << "       Total Computational Time for Neighbor Search(s): " << computeTime_neighborSearch / 1000. << std::endl;
            std::cout << "       Total Computational Time for Contact(s): " << computeTime_contact / 1000. << std::endl;
            std::cout << "       Total Computational Time for Integration(s): " << computeTime_integration / 1000. << std::endl;
            outputData(iFrame);
        }
		currentTime += timeStep;
		iStep++;
    }
}
