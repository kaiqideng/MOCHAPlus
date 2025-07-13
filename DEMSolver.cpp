#include "DEMSolver.h"

void DEMSolver:: solve()
{
    loadHostData(hos);
    hos.handleHostDataAfterLoading();
    if (!validateHostData(hos)) return;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(hos.simulation.deviceNumber);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Setting CUDA device failed!  Do you have a CUDA-capable GPU installed?");
    }
    else
    {
        std::cout << "Set CUDA device successfully. \n";
    }

    buildDeviceData(hos);
    int numSteps = int((hos.simulation.timeMax - hos.simulation.currentTime) / hos.simulation.timeStep) + 1;
    int saveAccount = numSteps / hos.simulation.nPrint;
    if (saveAccount == 0)  saveAccount = 1;
    computeTime_neighborSearch += timeHostFunc([&]() { neighborSearch(dev, hos.simulation.maxThreadsPerBlock, 0); });
    computeTime_neighborSearch += timeHostFunc([&]() { setBondedInteractions(dev, hos.simulation.maxThreadsPerBlock); });
    upload2Host();
    int iFrame = 0;
    outputData(hos, iFrame, 0);
    for (int iStep = 1; iStep <= numSteps; iStep++)
    {
        computeTime_neighborSearch += timeHostFunc([&]() { neighborSearch(dev, hos.simulation.maxThreadsPerBlock, iStep); });
        computeTime_integration += timeHostFunc([&]() { integrateBeforeContact(dev, hos.simulation.gravity, hos.simulation.timeStep, hos.simulation.maxThreadsPerBlock); });
        handleDataBeforeContact(hos, dev);
        computeTime_contact += timeHostFunc([&]() { calculateContactForceTorque(dev, hos.simulation.timeStep, hos.simulation.maxThreadsPerBlock, iStep); });
        handleDataAfterContact(hos, dev);
        computeTime_integration += timeHostFunc([&]() { integrateAfterContact(dev, hos.simulation.gravity, hos.simulation.timeStep, hos.simulation.maxThreadsPerBlock);});
        if (iStep % saveAccount == 0)
        {
            upload2Host();
            iFrame++;
            std::cout << "------ Frame: " << iFrame << " ------" << std::endl;
            std::cout << "       Simulation Time(s): " << hos.simulation.currentTime << std::endl;
            std::cout << "       Total Computational Time for Neighbor Search(s): " << computeTime_neighborSearch / 1000. << std::endl;
            std::cout << "       Total Computational Time for Contact(s): " << computeTime_contact / 1000. << std::endl;
            std::cout << "       Total Computational Time for Integration(s): " << computeTime_integration / 1000. << std::endl;
            outputData(hos, iFrame, iStep);
        }
        hos.simulation.currentTime += hos.simulation.timeStep;
    }
}