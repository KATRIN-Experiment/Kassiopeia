#include "KOpenCLInterface.hh"

#include "KEMCoreMessage.hh"
#include "KOpenCLData.hh"

#include <sstream>
#include <stdlib.h>


#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif

#ifndef KEMFIELD_DEFAULT_GPU_ID
#define KEMFIELD_DEFAULT_GPU_ID 0
#endif

#ifndef KEMFIELD_USE_DOUBLE_PRECISION
#pragma error "KEMField can't be used with OpenCL on a device that cannot provide double precision. Single precision is not good enough for charge calculations and will result in limited accuracy. You might be able to manually enable double precision with your device, see also: https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-double-precision-emulation-fp64 ."
#endif

#ifndef DEFAULT_KERNEL_DIR
#define DEFAULT_KERNEL_DIR "."
#endif /* !DEFAULT_KERNEL_DIR */

namespace KEMField
{
KOpenCLInterface* KOpenCLInterface::fOpenCLInterface = nullptr;

KOpenCLInterface::KOpenCLInterface() :
    fKernelPath(DEFAULT_KERNEL_DIR),
    fPlatforms(),
    fDevices(),
    fCLDeviceID(0),
    fContext(nullptr),
    fQueues(),
    fActiveData(nullptr)
{
    Initialize();
}

KOpenCLInterface::~KOpenCLInterface()
{
    for (std::vector<cl::CommandQueue*>::iterator it = fQueues.begin(); it != fQueues.end(); ++it) {
        if (*it)
            delete *it;
    }
}

/**
   * Interface to accessing KOpenCLInterface.
   */
KOpenCLInterface* KOpenCLInterface::GetInstance()
{
    if (fOpenCLInterface == nullptr)
        fOpenCLInterface = new KOpenCLInterface();
    return fOpenCLInterface;
}

/**
   * Queries the host for available OpenCL platforms, and constructs a Context.
   */
void KOpenCLInterface::Initialize()
{
    // Disable CUDA caching, since it doesn't check if included .cl files have
    // changed
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    // Get available platforms
    cl::Platform::get(&fPlatforms);

    if (fPlatforms.size() == 0) {
        kem_cout(eError) << "There are no OpenCL platforms available on this system." << eom;
        return;
    }
    else if (fPlatforms.size() <= KEMFIELD_OPENCL_PLATFORM) {
        kem_cout(eError) << "Cannot select platform ID # " << KEMFIELD_OPENCL_PLATFORM
                         << ", since there are only " << fPlatforms.size() << " platforms available." << eom;
        return;
    }

    cl::string name = fPlatforms[KEMFIELD_OPENCL_PLATFORM].getInfo<CL_PLATFORM_NAME>();
    kem_cout(eInfo) << "Selecting platform ID # " << KEMFIELD_OPENCL_PLATFORM << " (" << name << ") of " << fPlatforms.size()
                    << " available platforms." << eom;

    // Select the default platform and create a context using this platform and
    // the GPU
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties)(fPlatforms[KEMFIELD_OPENCL_PLATFORM])(),
                                    0};

    int deviceType = KEMFIELD_OPENCL_DEVICE_TYPE;

    if (deviceType == -1)
    {
        fContext = new cl::Context(CL_DEVICE_TYPE_ALL, cps);
    }
    if (deviceType == 0)  //we have a GPU
    {
        fContext = new cl::Context(CL_DEVICE_TYPE_GPU, cps);
    }
    if (deviceType == 1)  //we have a CPU
    {
        fContext = new cl::Context(CL_DEVICE_TYPE_CPU, cps);
    }
    if (deviceType == 2)  //we have an accelerator device
    {
        fContext = new cl::Context(CL_DEVICE_TYPE_ACCELERATOR, cps);
    }

    if (! fContext) {
        kem_cout(eError) << "Failed to create OpenCL context for device type " << deviceType << "." << eom;
        return;
    }

    CL_VECTOR_TYPE<cl::Device> availableDevices = fContext->getInfo<CL_CONTEXT_DEVICES>();

    fDevices.clear();
    fDevices = availableDevices;
    fQueues.resize(fDevices.size(), NULL);

    if (fDevices.size() == 0) {
        kem_cout(eError) << "There are no OpenCL devices available on this platform." << eom;
        return;
    }

    fCLDeviceID = 0;
    fKernelPath = DEFAULT_KERNEL_DIR;

#ifdef KEMFIELD_USE_MPI
    //assign devices according to the number available and local process rank
    unsigned int proc_id = KMPIInterface::GetInstance()->GetProcess();
    int n_dev = GetNumberOfDevices();
    int dev_id = proc_id % n_dev;  //fallback to global process rank if local is unavailable
    int local_rank = KMPIInterface::GetInstance()->GetLocalRank();
    if (local_rank != -1) {
        if (KMPIInterface::GetInstance()->SplitMode()) {
            dev_id = (local_rank / 2) % n_dev;
        }
        else {
            dev_id = (local_rank) % n_dev;
        }
    }
    SetGPU(dev_id);
#else
    SetGPU(KEMFIELD_DEFAULT_GPU_ID);
#endif

}

/**
   * Selects a device for use in OpenCL calculations.
   */
void KOpenCLInterface::SetGPU(unsigned int i)
{
    if (i >= fDevices.size()) {
        kem_cout(eWarning) << "Cannot set GPU device to ID # " << i
                           << ", since there are only " << fDevices.size() << " devices available." << eom;
        return;
    }
#ifdef KEMFIELD_USE_DOUBLE_PRECISION
    cl::string extensions = fDevices[i].getInfo<CL_DEVICE_EXTENSIONS>();
    if ((std::strstr(extensions.c_str(), "cl_khr_fp64") == nullptr) &&
        (std::strstr(extensions.c_str(), "cl_amd_fp64") == nullptr)) {
        kem_cout(eWarning) << "Cannot set GPU device to ID # " << i
                           << ", since it does not support double precision (and this program was built with double precision enabled)." << eom;
        return;
    }
#endif /* KEMFIELD_USE_DOUBLE_PRECISION */


    cl::string name = fDevices[i].getInfo<CL_DEVICE_NAME>();

#ifdef KEMFIELD_USE_MPI

    int process_id = KMPIInterface::GetInstance()->GetProcess();
    std::stringstream msg;
    msg << "Process #" << process_id << ": Setting GPU device to ID # " << i << " (" << name << ") of "
        << fDevices.size() << " available devices on host: " << KMPIInterface::GetInstance()->GetHostName();
#ifdef KEMFIELD_USE_DOUBLE_PRECISION
    msg << " (double precision enabled)." << std::endl;
#else
    msg << "." << std::endl;
#endif
    KMPIInterface::GetInstance()->PrintMessage(msg.str());

#else

    kem_cout() << "Setting GPU device to ID # " << i << " (" << name << ") of " << fDevices.size()
               << " available devices";
#ifdef KEMFIELD_USE_DOUBLE_PRECISION
    kem_cout() << " (double precision enabled)." << eom;
#else
    kem_cout() << "." << eom;
#endif

#endif

    fCLDeviceID = i;
}

void KOpenCLInterface::SetActiveData(KOpenCLData* data)
{
    fActiveData = data;
}

KOpenCLData* KOpenCLInterface::GetActiveData() const
{
    return fActiveData;
}

cl::CommandQueue& KOpenCLInterface::GetQueue(int i) const
{
    int deviceID = (i == -1 ? fCLDeviceID : i);

    if (!fQueues.at(deviceID))
        fQueues[deviceID] = new cl::CommandQueue(GetContext(), GetDevice());
    return *fQueues[deviceID];
}
}  // namespace KEMField
